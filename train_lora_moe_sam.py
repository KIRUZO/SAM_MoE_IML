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

# 导入 SAM 官方库
from segment_anything import sam_model_registry

# ==========================================
# 1. MoE-LoRA 组件 (保持不变)
# ==========================================
# class LoRAExpert(nn.Module):
#     def __init__(self, dim, rank=16):
#         super().__init__()
#         self.lora_A = nn.Linear(dim, rank, bias=False)
#         self.lora_B = nn.Linear(rank, dim, bias=False)
#         nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B.weight)

#     def forward(self, x):
#         return self.lora_B(self.lora_A(x))

import os
import logging
from datetime import datetime
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    
    if not logger.handlers:
        # 1. 生成时间戳后缀，例如: 20260106_153045
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'train_{timestamp}.log'
        
        # 2. 修改 FileHandler 的路径
        log_path = os.path.join(output_dir, log_filename)
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(logging.Formatter(log_format))
        logger.addHandler(fh)
        
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(log_format))
        logger.addHandler(sh)
        
        # 打印一下当前的日志文件路径，方便确认
        print(f"Logging to: {log_path}")
        
    return logger

class MoE_Adapter(nn.Module):
    def __init__(self, original_linear, num_experts=3, rank=16):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features # 获取原始层的输出维度 (3840)
        
        # 专家组：输出维度必须与 original_linear 一致
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_features, rank, bias=False),
                nn.Linear(rank, self.out_features, bias=False) # 修改此处：in_features -> out_features
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
        # print("Input x shape to MoE_Adapter:", x.shape)
        # Input x shape to MoE_Adapter: torch.Size([25, 14, 14, 1280])
        # Input x shape to MoE_Adapter: torch.Size([25, 14, 14, 1280])
        # Input x shape to MoE_Adapter: torch.Size([25, 14, 14, 1280])
        # Input x shape to MoE_Adapter: torch.Size([25, 14, 14, 1280])
        # Input x shape to MoE_Adapter: torch.Size([25, 14, 14, 1280])
        # Input x shape to MoE_Adapter: torch.Size([25, 14, 14, 1280])
        # Input x shape to MoE_Adapter: torch.Size([25, 14, 14, 1280])
        # Input x shape to MoE_Adapter: torch.Size([1, 64, 64, 1280])
        # 1. 原始路径输出 [B, H, W, 3840]
        with torch.no_grad():
            base_out = self.original_linear(x)
            
        # 2. MoE 路径
        # gate_weights: [B, H, W, num_experts]
        gate_weights = self.router(x) # 由router决定每个样本的专家权重分布
        
        # expert_outs: [B, H, W, 3840, num_experts] 
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)
        
        # moe_out: [B, H, W, 3840] # 各个专家的响应权重weights加权求和
        moe_out = (expert_outs * gate_weights.unsqueeze(-2)).sum(dim=-1)
        
        # 3. 此时维度匹配：3840 + 3840
        # 对于SAM Vit-h qkv 的输出维度是 1280 * 3 = 3840
        return base_out + moe_out

class SAM_MoE_Forgery(nn.Module):
    def __init__(self, model_type, checkpoint):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        # print("The structure of SAM:",sam)
        i= 0
        for blk in sam.image_encoder.blocks:
            # print("Modifying block:", blk)

            # Modifying block: Block(
            # (norm1): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
            # (attn): Attention(
            #     (qkv): Linear(in_features=1280, out_features=3840, bias=True)
            #     (proj): Linear(in_features=1280, out_features=1280, bias=True)
            # )
            # (norm2): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
            # (mlp): MLPBlock(
            #     (lin1): Linear(in_features=1280, out_features=5120, bias=True)
            #     (lin2): Linear(in_features=5120, out_features=1280, bias=True)
            #     (act): GELU(approximate='none')
            # )
            # )
            # i += 1
            # print(i) # 32层 transformer block

            # print('=======================================================')
            # print('=======================================================')
            # print('=======================================================')
            dim = blk.attn.qkv.in_features
            # print("blk.attn.qkv",blk.attn.qkv) # blk.attn.qkv Linear(in_features=1280, out_features=3840, bias=True)
            # 1. 输入到blk.attn.qkv的是原始的original_linear
            # 2. 我们用MoE_Adapter来包装它
            blk.attn.qkv = MoE_Adapter(blk.attn.qkv)
            
        for name, param in sam.named_parameters():
            if any(k in name for k in ["experts", "router", "mask_decoder"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.sam = sam # 这里已经是替换后的sam了，加了MoE组件

    def forward(self, image):
        # image: [B, 3, 1024, 1024]
        image_embeddings = self.sam.image_encoder(image) # [B, 256, 64, 64]
        # print("Image embeddings shape:", image_embeddings.shape) # Image embeddings shape: torch.Size([1, 256, 64, 64])
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        # 现在没有特定的点或框指引，根据图像特征自行判断

        # SAM 原生输出分辨率为 256x256
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks # [B, 1, 256, 256]

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
            
        self.sam_input_size = img_size # 1024
        self.gt_size = img_size // 4   # 256

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        
        # 加载图像
        forgery = cv2.imread(os.path.join(self.tp_dir, fname))
        forgery = cv2.cvtColor(forgery, cv2.COLOR_BGR2RGB)
        
        gt_mask = cv2.imread(os.path.join(self.gt_dir, fname), cv2.IMREAD_GRAYSCALE)

        # 核心对齐：Image 1024 (AREA), GT 256 (NEAREST)
        if self.sam_input_size is not None:
            forgery = cv2.resize(forgery, (self.sam_input_size, self.sam_input_size), cv2.INTER_AREA)
            gt_mask = cv2.resize(gt_mask, (self.gt_size, self.gt_size), cv2.INTER_NEAREST)

        # 处理 Image
        forgery = forgery.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(forgery).permute(2, 0, 1)
        img_tensor = self.normalize(img_tensor)

        # 处理 GT
        gt_mask = np.where(gt_mask > 127, 1.0, 0.0).astype(np.float32)
        mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0)
        
        return img_tensor, mask_tensor, fname

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
# ==========================================
# 4. 训练主脚本
# ==========================================
def visualize_at_256(img_1024, pred_256, gt_256, save_path):
    # 将 1024 的图缩放到 256 以便对齐显示
    img_256 = F.interpolate(img_1024.unsqueeze(0), (256, 256), mode='bilinear').squeeze(0)
    img_np = img_256.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    
    # pred_np = (torch.sigmoid(pred_256).squeeze().cpu().numpy() > 0.5).astype(np.float32)
    # 【关键修改】：增加 .detach()
    pred_np = (torch.sigmoid(pred_256).detach().squeeze().cpu().numpy() > 0.5).astype(np.float32) # 这里是大于0.5的二值化结果
    # 后续可以提高阈值看是否能够打压非篡改区域的参与

    gt_np = gt_256.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Input (Resized 256)")
    axes[1].imshow(pred_np, cmap='gray', interpolation='nearest')
    axes[1].set_title("Pred (Native 256)")
    axes[2].imshow(gt_np, cmap='gray', interpolation='nearest')
    axes[2].set_title("GT (Native 256)")
    plt.savefig(save_path)
    plt.close()

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
    
    # ==========================================
    # 1. 记录所有参数 (核心改进)
    # ==========================================
    logger.info("================ Parameters Configuration ================")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("==========================================================")




    device = torch.device('cuda')
    model = SAM_MoE_Forgery("vit_h", args.sam_ckpt).to(device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # 此处省略 ForgeryLoss 定义，见前文 (BCE + Dice)
    criterion = ForgeryLoss().to(device)
    scaler = GradScaler() if args.mixed_precision else None
    
    dataset = AutomatedForgeryDataset(args.train_root, img_size=args.img_size, subset_ratio=args.subset_ratio)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, gts, _ in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            optimizer.zero_grad()
            
            # with autocast(enabled=args.mixed_precision):
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                # preds 形状为 [B, 1, 256, 256]
                # gts 形状为 [B, 1, 256, 256]
                preds = model(imgs)
                loss = criterion(preds, gts)
            
            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            curr_loss = loss.item()
            pbar.set_postfix({'loss': curr_loss})
            epoch_loss += curr_loss
            
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.vis_interval == 0:
            os.makedirs(f"{args.output_dir}/vis", exist_ok=True)
            visualize_at_256(imgs[0], preds[0], gts[0], f"{args.output_dir}/vis/ep{epoch+1}.png")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/model_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()