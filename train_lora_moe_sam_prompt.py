#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from datetime import datetime

# 导入 CLIP 和 SAM
import clip
from segment_anything import sam_model_registry

# ==========================================
# 1. 语义映射器与 MoE 组件
# ==========================================
class SemanticCLIPMapper(nn.Module):
    def __init__(self, clip_dim=512, sam_dim=256, num_tokens=4):
        super().__init__()
        # 核心：可学习的查询令牌
        self.learnable_queries = nn.Parameter(torch.randn(1, num_tokens, sam_dim))
        
        # 文本特征转换网络
        self.text_net = nn.Sequential(
            nn.Linear(clip_dim, sam_dim),
            nn.LayerNorm(sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, sam_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=sam_dim, num_heads=8, batch_first=True)
        
    def forward(self, text_features):
        B = text_features.shape[0]
        # 转换并增强文本特征
        text_emb = self.text_net(text_features).unsqueeze(1) # [B, 1, 256]
        # 使用学习令牌提取语义信息
        queries = self.learnable_queries.expand(B, -1, -1)
        semantic_prompts, _ = self.cross_attn(query=queries, key=text_emb, value=text_emb)
        return semantic_prompts # [B, 4, 256]

class MoE_Adapter(nn.Module):
    def __init__(self, original_linear, num_experts=3, rank=16):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features 
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_features, rank, bias=False),
                nn.Linear(rank, self.out_features, bias=False) 
            ) for _ in range(num_experts)
        ])
        
        for exp in self.experts:
            nn.init.kaiming_uniform_(exp[0].weight, a=math.sqrt(5))
            nn.init.zeros_(exp[1].weight)
            
        self.router = nn.Sequential(
            nn.Linear(self.in_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        with torch.no_grad():
            base_out = self.original_linear(x)
        gate_weights = self.router(x) 
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)
        moe_out = (expert_outs * gate_weights.unsqueeze(-2)).sum(dim=-1)
        return base_out + moe_out

class SAM_MoE_Forgery(nn.Module):
    def __init__(self, model_type, checkpoint):
        super().__init__()
        # 1. 加载并注入 MoE-LoRA
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        for blk in sam.image_encoder.blocks:
            blk.attn.qkv = MoE_Adapter(blk.attn.qkv)
        self.sam = sam
        
        # 2. 加载冻结的 CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 3. 语义映射器 (可学习)
        self.semantic_mapper = SemanticCLIPMapper(clip_dim=512, sam_dim=256)
        
        # 4. 更新训练权限
        for name, param in self.named_parameters():
            if any(k in name for k in ["experts", "router", "semantic_mapper", "mask_decoder"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, image, text_tokens):
        # image: [B, 3, 1024, 1024], text_tokens: [B, 77]
        
        # A. Encoder 特征提取
        image_embeddings = self.sam.image_encoder(image)
        
        # B. CLIP 语义特征提取
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).float()
            
        # C. 转化为可学习的语义提示 (sparse)
        semantic_sparse_embeddings = self.semantic_mapper(text_features)
        

        _, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # D. Decoder 分割
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=semantic_sparse_embeddings, 
            dense_prompt_embeddings=dense_embeddings,           
            multimask_output=False,
        )
        return low_res_masks



# ==========================================
# 2. 工具函数
# ==========================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    if not logger.handlers:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(output_dir, f'train_{timestamp}.log')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(logging.Formatter(log_format))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(log_format))
        logger.addHandler(sh)
        print(f"Logging to: {log_path}")
    return logger

# ==========================================
# 3.  Dataset
# ==========================================
class AutomatedForgeryDataset(Dataset):
    def __init__(self, root_dir, img_size=1024, subset_ratio=1.0):
        self.tp_dir = os.path.join(root_dir, 'Tp')
        self.gt_dir = os.path.join(root_dir, 'Gt')
        self.file_list = sorted([f for f in os.listdir(self.tp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        
        if subset_ratio < 1.0:
            self.file_list = self.file_list[:int(len(self.file_list) * subset_ratio)]
            
        self.sam_input_size = img_size 
        self.gt_size = img_size // 4  

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.raw_text = "This image has a tampered region"
        self.text_tokens = clip.tokenize([self.raw_text])[0] 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        forgery = cv2.imread(os.path.join(self.tp_dir, fname))
        forgery = cv2.cvtColor(forgery, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(os.path.join(self.gt_dir, fname), cv2.IMREAD_GRAYSCALE)

        if self.sam_input_size is not None:
            forgery = cv2.resize(forgery, (self.sam_input_size, self.sam_input_size), cv2.INTER_AREA)
            gt_mask = cv2.resize(gt_mask, (self.gt_size, self.gt_size), cv2.INTER_NEAREST)

        forgery = forgery.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(forgery).permute(2, 0, 1)
        img_tensor = self.normalize(img_tensor)
        gt_mask = np.where(gt_mask > 127, 1.0, 0.0).astype(np.float32)
        mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0)
        
        return img_tensor, self.text_tokens, mask_tensor, fname

class ForgeryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * target).sum()
        dice = 1 - (2. * inter + 1e-5) / (pred_sig.sum() + target.sum() + 1e-5)
        return bce + dice

def visualize_at_256(img_1024, pred_256, gt_256, save_path):
    img_256 = F.interpolate(img_1024.unsqueeze(0), (256, 256), mode='bilinear').squeeze(0)
    img_np = img_256.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    pred_np = (torch.sigmoid(pred_256).detach().squeeze().cpu().numpy() > 0.5).astype(np.float32) 
    gt_np = gt_256.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Input (Resized 256)")
    axes[1].imshow(pred_np, cmap='gray', interpolation='nearest'); axes[1].set_title("Pred")
    axes[2].imshow(gt_np, cmap='gray', interpolation='nearest'); axes[2].set_title("GT")
    plt.savefig(save_path); plt.close()

# ==========================================
# 4. 训练主脚本
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./runs/final_moe')
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--sam_ckpt', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vis_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--mixed_precision', action='store_true')
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    logger.info("================ Parameters Configuration ================")
    for arg, value in vars(args).items(): logger.info(f"{arg}: {value}")
    logger.info("==========================================================")

    device = torch.device('cuda')
    model = SAM_MoE_Forgery("vit_h", args.sam_ckpt).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = ForgeryLoss().to(device)
    scaler = GradScaler() if args.mixed_precision else None
    
    dataset = AutomatedForgeryDataset(args.train_root, img_size=args.img_size, subset_ratio=args.subset_ratio)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, text_tokens, gts, _ in pbar:
            imgs = imgs.to(device)
            gts = gts.to(device)
            text_tokens = text_tokens.to(device) 

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                preds = model(imgs, text_tokens)
                loss = criterion(preds, gts)
            
            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            curr_loss = loss.item()
            pbar.set_postfix({'loss': f'{curr_loss:.4f}'})
            epoch_loss += curr_loss
            
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.vis_interval == 0:
            os.makedirs(f"{args.output_dir}/vis", exist_ok=True)
            model.eval()
            with torch.no_grad():
                # 传入 preds[0].detach() 确保可视化时不会报错
                visualize_at_256(imgs[0], preds[0].detach(), gts[0], f"{args.output_dir}/vis/ep{epoch+1}.png")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/model_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()