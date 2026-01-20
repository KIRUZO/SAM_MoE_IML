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

# 导入 SAM 官方库
from segment_anything import sam_model_registry
# 导入 CLIP 库
import clip


class PrototypicalMemoryBank(nn.Module):
    def __init__(self, feat_dim=256, num_protos=16):
        super().__init__()
        # 1. 存储篡改原型 (Forgery Prototypes) 和 真实原型 (Authentic Prototypes)
        # 每组 16 个，覆盖不同的噪声和边缘模式
        # 用register_buffer定义一个不参与梯度下降的、但属于模型状态的一部分的持久化张量
        self.register_buffer("forgery_protos", torch.randn(num_protos, feat_dim)) # 建立一个容器并起名
        self.register_buffer("authentic_protos", torch.randn(num_protos, feat_dim))

        # 初始标准化
        self.forgery_protos = F.normalize(self.forgery_protos, dim=-1) # 调整容器里内容的数值
        self.authentic_protos = F.normalize(self.authentic_protos, dim=-1)
        
        # 记忆库更新的动量 (Momentum)
        self.momentum = 0.9

    def get_similarity_guidance(self, x):
        """
        检索阶段：计算当前图像特征与库中原型的相似度
        x: [B, 256, 64, 64]
        """
        B, C, H, W = x.shape
        # x_flat: [B*H*W, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C) # 调整形状以便计算形状为[B*H*W, C] 4096 * 256
        x_norm = F.normalize(x_flat, dim=-1)

        # 计算与所有原型的余弦相似度
        # sim_f: 与篡改原型的相似度, sim_a: 与真实原型的相似度
        # 原型的形状为 [num_protos, C] 16 * 256
        # 做完 matmul 后形状为 [B*H*W, num_protos] 4096 * 16
        # 代表对于每一个像素点，全图4096个点，计算它与16个原型的相似度（每个像素点都得到了16个分值）
        sim_f = torch.matmul(x_norm, self.forgery_protos.t()) # [Total_Tokens, num_protos]
        sim_a = torch.matmul(x_norm, self.authentic_protos.t()) # [Total_Tokens, num_protos]

        # 后续可以给上述sim做可视化
        
        # 取最大相似度作为该点是“假”还是“真”的证据
        # 这里体现了 RAG 的“检索”思想
        evidence_f, _ = sim_f.max(dim=-1) # 既然每一个像素点有16个分值，取最大的那个作为该点的最终分值
        evidence_a, _ = sim_a.max(dim=-1)
        
        # 差值图：代表了该像素偏离正常规律的程度
        # 形状还原回 [B, 1, 64, 64]
        guidance = (evidence_f - evidence_a).view(B, 1, H, W) # 计算相对差异

        # 如果值大于 0，说明该点长得更像“假”的。
        # 如果值小于 0，说明该点长得更像“真”的。
        return guidance

    @torch.no_grad()
    def update_memory(self, x, mask):
        """
        学习阶段：根据当前 Batch 的真实特征动态更新记忆库中的专家槽位
        x: [B, 256, 64, 64] - 图像特征
        mask: [B, 1, 1024, 1024] - 原始GT
        """
        B, C, H, W = x.shape
        
        # 1. 下采样 mask 以匹配特征图尺寸 (64x64)
        mask_small = F.interpolate(mask, size=(H, W), mode='nearest') # [B, 1, 64, 64]

        # 2. 将特征图 x 调整为 [总像素数, 通道数]
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C) # [Batch*4096, 256]

        # 3. 将 mask 调整为 [总像素数]
        mask_flat = mask_small.view(-1) # [Batch*4096]

        # 4. 分离当前 Batch 的篡改像素和真实像素特征
        forgery_pixels = x_flat[mask_flat > 0.5]      # [N_f, 256]
        authentic_pixels = x_flat[mask_flat <= 0.5]   # [N_a, 256]

        # 5. 动态更新记忆库
        
        # --- A. 更新篡改原型 (Forgery Prototypes) ---
        if forgery_pixels.numel() > 0:
            # 计算当前 Batch 篡改特征的中心
            batch_f_center = F.normalize(forgery_pixels.mean(0, keepdim=True), dim=-1) # [1, 256]
            
            # 【核心改进】：寻找库中与当前特征最相似的“专家槽位”
            # 计算当前中心与 16 个原型的余弦相似度
            sim_to_protos = torch.matmul(batch_f_center, self.forgery_protos.t()) # [1, 16]
            nearest_idx = torch.argmax(sim_to_protos) # 找到最匹配的专家索引
            
            # 只对该最匹配的专家进行动量更新，使其在该领域更专业
            self.forgery_protos[nearest_idx : nearest_idx + 1] = \
                self.momentum * self.forgery_protos[nearest_idx : nearest_idx + 1] + \
                (1 - self.momentum) * batch_f_center
            
        # --- B. 更新真实原型 (Authentic Prototypes) ---
        if authentic_pixels.numel() > 0:
            # 计算当前 Batch 真实特征的中心
            batch_a_center = F.normalize(authentic_pixels.mean(0, keepdim=True), dim=-1) # [1, 256]
            
            # 寻找最相似的真实背景专家槽位
            sim_to_protos_a = torch.matmul(batch_a_center, self.authentic_protos.t()) # [1, 16]
            nearest_a_idx = torch.argmax(sim_to_protos_a)
            
            # 动量更新该槽位
            self.authentic_protos[nearest_a_idx : nearest_a_idx + 1] = \
                self.momentum * self.authentic_protos[nearest_a_idx : nearest_a_idx + 1] + \
                (1 - self.momentum) * batch_a_center

        # 6. 全局归一化，确保所有原型保持在单位超球面上，方便下次计算相似度
        self.forgery_protos = F.normalize(self.forgery_protos, dim=-1)
        self.authentic_protos = F.normalize(self.authentic_protos, dim=-1)
# ==========================================
# 1. 语义映射器与 DeepSeekMoE 组件
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
    """
    Inspired by DeepSeekMoE 架构 (纯视觉驱动):
    1. Shared Experts: 始终开启，提取全局通用篡改痕迹
    2. Routed Experts: 细粒度拆分 + Top-K 激活，捕获局部特定伪影
    """
    def __init__(self, original_linear, num_shared=1, num_routed=6, top_k=2, rank=16):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.top_k = top_k

        # (1) 共享专家：始终参与计算
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_features, rank, bias=False),
                nn.Linear(rank, self.out_features, bias=False)
            ) for _ in range(num_shared)
        ])

        # (2) 路由专家：细粒度拆分
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_features, rank, bias=False),
                nn.Linear(rank, self.out_features, bias=False)
            ) for _ in range(num_routed)
        ])
        
        # 初始化
        for exp_list in [self.shared_experts, self.routed_experts]:
            for exp in exp_list:
                nn.init.kaiming_uniform_(exp[0].weight, a=math.sqrt(5))
                nn.init.zeros_(exp[1].weight)
            
        # 路由器
        self.router = nn.Linear(self.in_features, num_routed)

    def forward(self, x):
        with torch.no_grad():
            base_out = self.original_linear(x)
            
        shared_out = 0
        for expert in self.shared_experts:
            shared_out += expert(x)
        
        logits = self.router(x)
        gate_weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        zeros = torch.zeros_like(gate_weights)
        sparse_weights = zeros.scatter(-1, topk_indices, topk_weights)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        routed_expert_outs = torch.stack([e(x) for e in self.routed_experts], dim=-1)
        routed_out = (routed_expert_outs * sparse_weights.unsqueeze(-2)).sum(dim=-1)
        
        return base_out + shared_out + routed_out

class SAM_MoE_Forgery(nn.Module):
    def __init__(self, model_type, checkpoint, device, n_ctx=16):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        
        # 1. 注入 DeepSeekMoE 适配器
        for blk in sam.image_encoder.blocks:
            blk.attn.qkv = MoE_Adapter(blk.attn.qkv, num_shared=1, num_routed=6, top_k=3, rank=16)
            
        # 2. 加载冻结的 CLIP 用于语义映射
        # self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.clip_model, _ = clip.load("ViT-B/32", device=device) # 使用传入的 device
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        clip_embed_dim = self.clip_model.token_embedding.weight.shape[1] # 512
        
        # 3. 定义可学习的 Context (Learnable Prompt)
        self.n_ctx = n_ctx
        self.learnable_ctx = nn.Parameter(torch.randn(n_ctx, clip_embed_dim))
        nn.init.normal_(self.learnable_ctx, std=0.02)
        
        # 4. 语义映射器
        self.semantic_mapper = SemanticCLIPMapper(clip_dim=512, sam_dim=256)

        # 5. 原型记忆库
        self.memory_bank = PrototypicalMemoryBank(feat_dim=256, num_protos=16)

        # 6. 权限管理
        for name, param in sam.named_parameters():
            if any(k in name for k in ["experts", "router", "mask_decoder", "learnable_ctx", "semantic_mapper"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 显式开启 CLIP 相关的可训练参数
        self.learnable_ctx.requires_grad = True
        for param in self.semantic_mapper.parameters():
            param.requires_grad = True

        self.sam = sam 

    def forward(self, image, gt_mask=None):
        B = image.shape[0]
        device = image.device

        # A. 视觉路径: Image Encoder
        image_embeddings = self.sam.image_encoder(image) 


        # B. RAG 检索：利用记忆库生成“伪造概率指引”
        # 这个 guidance 是一张 64x64 的热力图，标注了哪些像素像“记忆中的伪造”
        forgery_guidance = self.memory_bank.get_similarity_guidance(image_embeddings)

        # 在训练模式且提供 GT 时，更新记忆库的“经验值”
        if self.training and gt_mask is not None:
            self.memory_bank.update_memory(image_embeddings.detach(), gt_mask)
        
        # C. 语言路径: 纯 Learnable Prompt 流
        sos_id = torch.tensor([49406], device=device)
        eos_id = torch.tensor([49407], device=device)
        
        sos_emb = self.clip_model.token_embedding(sos_id).type(self.clip_model.dtype) # [1, 512]
        eos_emb = self.clip_model.token_embedding(eos_id).type(self.clip_model.dtype) # [1, 512]
        ctx = self.learnable_ctx.unsqueeze(0).expand(B, -1, -1).type(self.clip_model.dtype) # [B, n_ctx, 512]
        
        # 拼接序列: [SOS] + [Learnable Ctx] + [EOS]
        prefix = sos_emb.unsqueeze(0).expand(B, -1, -1)
        suffix = eos_emb.unsqueeze(0).expand(B, -1, -1)
        embeddings = torch.cat([prefix, ctx, suffix], dim=1) # [B, n_ctx+2, 512]
        
        # 补全到 CLIP 长度 77
        padding = torch.zeros((B, 77 - embeddings.shape[1], 512), device=device, dtype=self.clip_model.dtype)
        full_embeddings = torch.cat([embeddings, padding], dim=1)
        
        # 手动推导 CLIP Transformer
        x = full_embeddings + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        # 提取特征 (EOS 位置)
        text_features = x[:, self.n_ctx + 1, :] @ self.clip_model.text_projection
        
        # 映射为动态稀疏 Prompt
        semantic_sparse_embeddings = self.semantic_mapper(text_features.float())

        

        # C. 解码路径
        _, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        # [关键 Trick]: 将检索到的 forgery_guidance 直接加到 dense_embeddings 上
        # 这样 Decoder 在看图像特征时，会被强制带上记忆库的“偏见”
        enhanced_dense_embeddings = dense_embeddings + forgery_guidance

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=semantic_sparse_embeddings, # 注入可学习语义提示
            dense_prompt_embeddings=dense_embeddings, # 注入 RAG 指引
            multimask_output=False,
        )
        return low_res_masks

# ==========================================
# 2. 数据、Loss、可视化 (保持不变)
# ==========================================

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    if not logger.handlers:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(output_dir, f'train_{timestamp}.log')
        fh = logging.FileHandler(log_path, mode='w'); fh.setFormatter(logging.Formatter(log_format)); logger.addHandler(fh)
        sh = logging.StreamHandler(); sh.setFormatter(logging.Formatter(log_format)); logger.addHandler(sh)
        print(f"Logging to: {log_path}")
    return logger

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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        forgery = cv2.imread(os.path.join(self.tp_dir, fname)); forgery = cv2.cvtColor(forgery, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(os.path.join(self.gt_dir, fname), cv2.IMREAD_GRAYSCALE)
        if self.sam_input_size is not None:
            forgery = cv2.resize(forgery, (self.sam_input_size, self.sam_input_size), cv2.INTER_AREA)
            gt_mask = cv2.resize(gt_mask, (self.gt_size, self.gt_size), cv2.INTER_NEAREST)
        img_tensor = self.normalize(torch.from_numpy(forgery.astype(np.float32) / 255.0).permute(2, 0, 1))
        mask_tensor = torch.from_numpy(np.where(gt_mask > 127, 1.0, 0.0).astype(np.float32)).unsqueeze(0)
        return img_tensor, mask_tensor, fname

class ForgeryLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.bce = nn.BCEWithLogitsLoss()
    def forward(self, pred, target):
        bce = self.bce(pred, target); pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * target).sum(); dice = 1 - (2. * inter + 1e-5) / (pred_sig.sum() + target.sum() + 1e-5)
        return bce + dice

def visualize_at_256(img_1024, pred_256, gt_256, save_path):
    img_256 = F.interpolate(img_1024.unsqueeze(0), (256, 256), mode='bilinear').squeeze(0)
    img_np = np.clip(img_256.permute(1, 2, 0).cpu().numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    pred_np = (torch.sigmoid(pred_256).detach().squeeze().cpu().numpy() > 0.5).astype(np.float32) 
    gt_np = gt_256.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Input (Resized 256)")
    axes[1].imshow(pred_np, cmap='gray', interpolation='nearest'); axes[1].set_title("Pred (Native 256)")
    axes[2].imshow(gt_np, cmap='gray', interpolation='nearest'); axes[2].set_title("GT (Native 256)")
    plt.savefig(save_path); plt.close()

# ==========================================
# 3. 训练主流程 (保持参数一致)
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
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--vis_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use') 
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    logger.info("================ Parameters Configuration ================")
    for arg, value in vars(args).items(): logger.info(f"{arg}: {value}")
    logger.info("==========================================================")

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # model = SAM_MoE_Forgery("vit_h", args.sam_ckpt).to(device)
    model = SAM_MoE_Forgery("vit_h", args.sam_ckpt, device=device).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # try the new SAM optimizer
    # base_optimizer = optim.AdamW
    # optimizer = SAMOptimizer(
    #     filter(lambda p: p.requires_grad, model.parameters()), 
    #     base_optimizer, 
    #     rho=0.05, # 扰动半径，0.05 是通用推荐值
    #     lr=args.lr,
    #     weight_decay=1e-4
    # )
    criterion = ForgeryLoss().to(device)
    scaler = GradScaler() if args.mixed_precision else None
    
    dataset = AutomatedForgeryDataset(args.train_root, img_size=args.img_size, subset_ratio=args.subset_ratio)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    for epoch in range(args.epochs):
        epoch_loss = 0.0; model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, gts, _ in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            
            # --- 第一步：寻找最坏邻居 ---
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                preds = model(imgs, gt_mask=gts) # 传入 GT 以更新记忆库
                loss = criterion(preds, gts)
            
            if args.mixed_precision:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            
            curr_loss = loss.item(); pbar.set_postfix({'loss': f'{curr_loss:.4f}'}); epoch_loss += curr_loss
            
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.vis_interval == 0:
            os.makedirs(f"{args.output_dir}/vis", exist_ok=True)
            model.eval()
            with torch.no_grad():
                visualize_at_256(imgs[0], preds[0].detach(), gts[0], f"{args.output_dir}/vis/ep{epoch+1}.png")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/model_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()