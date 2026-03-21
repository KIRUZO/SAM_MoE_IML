import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
from PIL import Image
import argparse
import math
import json
import clip
import sys

# if needed...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_val import test_dataset
from segment_anything import sam_model_registry

warnings.filterwarnings("ignore")


class PrototypicalMemoryBank(nn.Module):
    def __init__(self, feat_dim=256, num_protos=16):
        super().__init__()
        self.register_buffer("forgery_protos", torch.randn(num_protos, feat_dim))
        self.register_buffer("authentic_protos", torch.randn(num_protos, feat_dim))
        self.forgery_protos = F.normalize(self.forgery_protos, dim=-1)
        self.authentic_protos = F.normalize(self.authentic_protos, dim=-1)
        self.momentum = 0.9

    def get_similarity_guidance(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        x_norm = F.normalize(x_flat, dim=-1)
        sim_f = torch.matmul(x_norm, self.forgery_protos.t())
        sim_a = torch.matmul(x_norm, self.authentic_protos.t())
        evidence_f, _ = sim_f.max(dim=-1)
        evidence_a, _ = sim_a.max(dim=-1)
        guidance = (evidence_f - evidence_a).view(B, 1, H, W)
        return guidance

    @torch.no_grad()
    def update_memory(self, x, mask):
        # 评估阶段不调用此函数
        pass

class SemanticCLIPMapper(nn.Module):
    def __init__(self, clip_dim=512, sam_dim=256, num_tokens=4):
        super().__init__()
        self.learnable_queries = nn.Parameter(torch.randn(1, num_tokens, sam_dim))
        self.text_net = nn.Sequential(
            nn.Linear(clip_dim, sam_dim),
            nn.LayerNorm(sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, sam_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=sam_dim, num_heads=8, batch_first=True)
        
    def forward(self, text_features):
        B = text_features.shape[0]
        text_emb = self.text_net(text_features).unsqueeze(1)
        queries = self.learnable_queries.expand(B, -1, -1)
        semantic_prompts, _ = self.cross_attn(query=queries, key=text_emb, value=text_emb)
        return semantic_prompts

class MoE_Adapter(nn.Module):
    def __init__(self, original_linear, num_shared=1, num_routed=6, top_k=2, rank=16):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.top_k = top_k
        
        # 共享专家 (始终计算)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(self.in_features, rank, bias=False), nn.Linear(rank, self.out_features, bias=False)) 
            for _ in range(num_shared)
        ])
        
        # 路由专家 (稀疏计算)
        self.routed_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(self.in_features, rank, bias=False), nn.Linear(rank, self.out_features, bias=False)) 
            for _ in range(num_routed)
        ])
        
        self.router = nn.Linear(self.in_features, num_routed)

    def forward(self, x):
        # 1. 基础路径 (无梯度，保持原样)
        with torch.no_grad(): 
            base_out = self.original_linear(x)
        
        # 2. 共享专家路径 (始终计算)
        shared_out = 0
        for expert in self.shared_experts: 
            shared_out += expert(x)
        
        # ============================================================
        # 实现稀疏路由计算
        # ============================================================
        
        # A. 计算路由分数并获取 Top-K
        logits = self.router(x)
        gate_weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1) 
        # topk_indices shape: [B, Seq_Len, top_k]
        
        B, Seq_Len, _ = x.shape
        device = x.device
        
        # B. 准备容器存储路由专家的输出
        routed_out = torch.zeros_like(x) # [B, Seq_Len, Out_Features]
        
        # C. 稀疏计算核心逻辑：
        # 遍历每一个被选中的 "槽位" (0 到 top_k-1)
        # 对于每个槽位 k，我们只计算该槽位对应的专家，避免计算未选中的专家
        for k in range(self.top_k):
            # 获取当前槽位 k 选中的专家索引 [B, Seq_Len]
            current_expert_indices = topk_indices[:, :, k]
            
            # 获取对应的权重 [B, Seq_Len, 1]
            current_weights = topk_weights[:, :, k:k+1]
            
            # --- 批量 gather 输入并分发给对应专家 ---
            # 由于不同位置可能选择不同的专家，直接向量化比较困难。
            
            # 真正的稀疏需要重排 (Reorder) 数据。
            # 1. 将 (Batch, Seq) 展平为 Total_Tokens
            # 2. 根据 expert_index 排序
            # 3. 批量计算每个专家的任务
            # 4. 还原顺序
            
            flat_x = x.reshape(-1, self.in_features) # [Total_Tokens, In]
            flat_indices = current_expert_indices.reshape(-1) # [Total_Tokens]
            flat_weights = current_weights.reshape(-1) # [Total_Tokens]
            
            total_tokens = flat_x.shape[0]
            
            # 初始化当前槽位的输出
            current_slot_out = torch.zeros_like(flat_x)
            
            # 对每个专家 ID (0 到 num_routed-1)，只计算属于它的 token
            # 这一步确保了：如果一个专家没被任何 token 选中，它完全不被计算
            for expert_id in range(self.num_routed):
                # 找到所有选择了当前 expert_id 的 token 索引
                mask = (flat_indices == expert_id)
                if not mask.any():
                    continue
                
                selected_tokens = flat_x[mask]
                
                # 只对这部分的 selected_tokens 调用 expert
                expert_output = self.routed_experts[expert_id](selected_tokens)
                current_slot_out[mask] = expert_output * flat_weights[mask].unsqueeze(-1)
              
            routed_out += current_slot_out.reshape(B, Seq_Len, self.out_features)

        
        return base_out + shared_out + routed_out

class SAM_MoE_Forgery(nn.Module):
    def __init__(self, model_type, checkpoint, device, n_ctx=16):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        # 注入修改后的 MoE Adapter
        for blk in sam.image_encoder.blocks:
            blk.attn.qkv = MoE_Adapter(blk.attn.qkv, num_shared=1, num_routed=6, top_k=3, rank=16)
        self.sam = sam
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters(): param.requires_grad = False
        self.n_ctx = n_ctx
        self.learnable_ctx = nn.Parameter(torch.randn(n_ctx, 512))
        self.semantic_mapper = SemanticCLIPMapper(clip_dim=512, sam_dim=256)
        self.memory_bank = PrototypicalMemoryBank(feat_dim=256, num_protos=16)

    def forward(self, image, gt_mask=None):
        B = image.shape[0]; device = image.device
        image_embeddings = self.sam.image_encoder(image) 
        forgery_guidance = self.memory_bank.get_similarity_guidance(image_embeddings)
        
        sos_id = torch.tensor([49406], device=device); eos_id = torch.tensor([49407], device=device)
        sos_emb = self.clip_model.token_embedding(sos_id).type(self.clip_model.dtype)
        eos_emb = self.clip_model.token_embedding(eos_id).type(self.clip_model.dtype)
        ctx = self.learnable_ctx.unsqueeze(0).expand(B, -1, -1).type(self.clip_model.dtype)
        prefix = sos_emb.unsqueeze(0).expand(B, -1, -1); suffix = eos_emb.unsqueeze(0).expand(B, -1, -1)
        embeddings = torch.cat([prefix, ctx, suffix], dim=1)
        padding = torch.zeros((B, 77 - embeddings.shape[1], 512), device=device, dtype=self.clip_model.dtype)
        full_embeddings = torch.cat([embeddings, padding], dim=1)
        x = full_embeddings + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2); x = self.clip_model.transformer(x); x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        text_features = x[:, self.n_ctx + 1, :] @ self.clip_model.text_projection
        semantic_sparse_embeddings = self.semantic_mapper(text_features.float())

        _, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        enhanced_dense_embeddings = dense_embeddings + forgery_guidance
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings, image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=semantic_sparse_embeddings, dense_prompt_embeddings=enhanced_dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks


def pil_to_save(array, name, target_size, save_path):
    array = (array * 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    pil_image.save(os.path.join(save_path, name))

def metric(premask, groundtruth):
    premask = (premask > 0.5).astype(np.bool_)
    groundtruth = (groundtruth > 0.5).astype(np.bool_)
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    iou = np.sum(np.logical_and(premask, groundtruth)) / (np.sum(np.logical_or(premask, groundtruth)) + 1e-6)
    return f1, iou

def get_gt_path(file, path_gt):
    if "NC16" in path_gt: return path_gt + file
    elif "C1" in path_gt: return path_gt + file[:-4] + '_gt.png'
    elif "Coverage" in path_gt: return path_gt + file[:-5] + 't.tif'
    elif "Columbia" in path_gt: return path_gt + file[:-4] + '.png'
    elif "Imd" in path_gt: return path_gt + file[:-4] + '.png'
    elif "Korus" in path_gt: return path_gt + file[:-4] + '.PNG'
    elif "CocoGlide" in path_gt: return path_gt + file[:-4] + '_mask.png' 
    elif "In-the-Wild" in path_gt: return path_gt + file[:-4] + '.png'
    elif "DSO" in path_gt: return path_gt + file[:-4] + '.png'
    else: return path_gt + file

def process_single_image(args):
    file, path_pre, path_gt = args
    try:
        pre = cv2.imread(path_pre + '/' + file, cv2.IMREAD_GRAYSCALE)
        gt_path = get_gt_path(file, path_gt); gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pre is None or gt is None: return None
        H, W = gt.shape
        if pre.shape[0] != H or pre.shape[1] != W: pre = cv2.resize(pre, (W, H))
        y_true = (gt > 127).flatten().astype(int)
        y_scores = pre.flatten() / 255.0
        auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else None
        f1, iou = metric(pre / 255.0, gt / 255.0)
        return (auc, f1, iou)
    except: return None

def evaluate_dataset(path_pre, path_gt, dataset_name, record_txt, epoch, model_name):
    flist = sorted(os.listdir(path_pre))
    args_list = [(f, path_pre, path_gt) for f in flist]
    with Pool(processes=min(16, cpu_count())) as pool:
        results = list(tqdm(pool.imap(process_single_image, args_list), total=len(args_list), desc=f'Eval {dataset_name}'))
    aucs, f1s, ious = [], [], []
    for r in results:
        if r:
            a, f, i = r
            if a is not None: aucs.append(a)
            f1s.append(f); ious.append(i)
    m_auc, m_f1, m_iou = np.mean(aucs), np.mean(f1s), np.mean(ious)
    res_str = f'{datetime.now()} | {model_name} | {dataset_name} | Ep:{epoch} | AUC:{m_auc:.4f}, F1:{m_f1:.4f}, IoU:{m_iou:.4f}\n'
    print(res_str)
    with open(record_txt, "a") as f: f.write(res_str)
    return m_auc, m_f1, m_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to your model_epX.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--sam_ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='/data/zireal/W1_manipulate/data/val_all_set')
    parser.add_argument('--output_dir', type=str, default='./results/rag_moe_sam/')
    parser.add_argument('--gpu', type=str, default='3')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = SAM_MoE_Forgery(args.sam_type, args.sam_ckpt, device=device, n_ctx=16)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
    model = model.to(device).eval()

    epoch = args.ckpt.split('ep')[-1].split('.')[0] if 'ep' in args.ckpt else "final"
    model_name = "SAM_RAG_MoE_Learnable_SparseInfer"
    
    all_results_data = {
        "meta": {"checkpoint": args.ckpt, "epoch": epoch, "timestamp": str(datetime.now())},
        "datasets": {}
    }

    record_txt = os.path.join(args.output_dir, f"val_results_ep{epoch}.txt")
    record_json = os.path.join(args.output_dir, f"val_results_ep{epoch}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    test_datasets = ['C1', 'Coverage', 'Columbia', 'NC16', 'Imd', 'Korus', 'In-the-Wild', 'DSO', 'CocoGlide']
    f1_in, f1_out = [], []

    for i, dataset in enumerate(test_datasets):
        img_dir = os.path.join(args.data_root, dataset, 'Tp/')
        gt_dir = os.path.join(args.data_root, dataset, 'Gt/')
        if not os.path.exists(img_dir): continue

        loader = test_dataset(image_root=img_dir, gt_root=gt_dir, testsize=1024)
        recent_save_path = os.path.join(args.output_dir, "masks", dataset)
        os.makedirs(recent_save_path, exist_ok=True)

        with torch.no_grad():
            for _ in tqdm(range(loader.size), desc=f'Inference {dataset}'):
                image, _, gt, name, _ = loader.load_data()
                orig_size = (gt.size[1], gt.size[0])
                res = model(image.to(device))
                res = torch.sigmoid(res).cpu().numpy().squeeze()
                pil_to_save(res, name, orig_size, recent_save_path)

        avg_auc, avg_f1, avg_iou = evaluate_dataset(recent_save_path, gt_dir, dataset, record_txt, epoch, model_name)
        
        all_results_data["datasets"][dataset] = {
            "AUC": round(float(avg_auc), 4), "F1": round(float(avg_f1), 4), "IoU": round(float(avg_iou), 4)
        }
        if i < 4: f1_in.append(avg_f1)
        else: f1_out.append(avg_f1)

    m_f1_in = np.mean(f1_in) if f1_in else 0
    m_f1_out = np.mean(f1_out) if f1_out else 0
    all_results_data["summary"] = {"f1_in_avg": round(float(m_f1_in), 4), "f1_out_avg": round(float(m_f1_out), 4)}

    with open(record_txt, "a") as f:
        f.write("-" * 60 + "\n")
        f.write(f"SUMMARY FOR EPOCH {epoch}\n")
        f.write(f"In-Distribution F1 (First 4): {m_f1_in:.4f}\n")
        f.write(f"Out-of-Distribution F1 (Others): {m_f1_out:.4f}\n")
        f.write("-" * 60 + "\n\n")

    with open(record_json, "w") as jf:
        json.dump(all_results_data, jf, indent=4)
    
    print(f"Eval Done. TXT: {record_txt} | JSON: {record_json}")
