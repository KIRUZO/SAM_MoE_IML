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

from utils.data_val import test_dataset
from segment_anything import sam_model_registry
import clip
warnings.filterwarnings("ignore")

# ==========================================
# 1. 架构定义 (必须与 train 文件完全一致)
# ==========================================
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

class SAM_MoE_Forgery(nn.Module):
    def __init__(self, model_type, checkpoint):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        for blk in sam.image_encoder.blocks:
            blk.attn.qkv = MoE_Adapter(blk.attn.qkv)
        self.sam = sam
        
        # === 必须添加：加载冻结的 CLIP ===
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # === 必须添加：语义映射器 ===
        self.semantic_mapper = SemanticCLIPMapper(clip_dim=512, sam_dim=256)

    def forward(self, image, text_tokens):
        # A. Encoder 特征提取
        image_embeddings = self.sam.image_encoder(image)
        
        # B. CLIP 语义特征提取
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).float()
            
        # C. 转化为语义提示
        semantic_sparse_embeddings = self.semantic_mapper(text_features)
        
        # D. 获取默认的 Dense Embeddings (对应之前修复的 TypeError)
        _, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # E. Decoder 分割
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=semantic_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, # 传入默认的密集嵌入
            multimask_output=False,
        )
        return low_res_masks

# ==========================================
# 2. 评估核心函数
# ==========================================
def pil_to_save(array, name, target_size, save_path):
    # array 为 0-1 之间的概率
    array = (array * 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    # resize 回原图尺寸
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
    
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    
    return f1, iou

def get_gt_path(file, path_gt):
    # 保持你原来的路径匹配逻辑
    if "NC16" in path_gt: return path_gt + file
    elif "C1" in path_gt: return path_gt + file[:-4] + '_gt.png'
    elif "Coverage" in path_gt: return path_gt + file[:-5] + 't.tif'
    elif "Columbia" in path_gt: return path_gt + file[:-4] + '.png'
    elif "Imd" in path_gt: return path_gt + file[:-4] + '.png'
    elif "Korus" in path_gt: return path_gt + file[:-4] + '.PNG'
    elif "Coco" in path_gt: return path_gt + file[:-4] + '_mask.png' 
    elif "In-the-Wild" in path_gt: return path_gt + file[:-4] + '.png'
    elif "DSO" in path_gt: return path_gt + file[:-4] + '.png'
    else: return path_gt + file

def process_single_image(args):
    file, path_pre, path_gt = args
    try:
        pre = cv2.imread(path_pre + '/' + file, cv2.IMREAD_GRAYSCALE)
        gt_path = get_gt_path(file, path_gt)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pre is None or gt is None: return None

        H, W = gt.shape
        if pre.shape[0] != H or pre.shape[1] != W:
            pre = cv2.resize(pre, (W, H))

        # AUC 计算 (使用概率图)
        y_true = (gt > 127).flatten().astype(int)
        y_scores = pre.flatten() / 255.0
        auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else None

        # F1, IoU 计算 (使用二值化)
        f1, iou = metric(pre / 255.0, gt / 255.0)
        return (auc, f1, iou)
    except Exception as e:
        return None

def evaluate_dataset(path_pre, path_gt, dataset_name, record_txt, epoch, model_name):
    flist = sorted(os.listdir(path_pre))
    args_list = [(f, path_pre, path_gt) for f in flist]

    with Pool(processes=min(16, cpu_count())) as pool:
        results = list(tqdm(pool.imap(process_single_image, args_list), total=len(args_list), desc=f'Evaluating {dataset_name}'))

    aucs, f1s, ious = [], [], []
    for r in results:
        if r is not None:
            a, f, i = r
            if a is not None: aucs.append(a)
            f1s.append(f)
            ious.append(i)

    m_auc, m_f1, m_iou = np.mean(aucs), np.mean(f1s), np.mean(ious)
    res_str = f'{datetime.now()} | Model: {model_name} | Dataset: {dataset_name} | Epoch: {epoch} | AUC: {m_auc:.4f}, F1: {m_f1:.4f}, IoU: {m_iou:.4f}\n'
    print(res_str)
    with open(record_txt, "a") as f:
        f.write(res_str)
        
    # 修改这里：返回所有指标
    return m_auc, m_f1, m_iou

# ==========================================
# 3. Main Eval 脚本
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to MoE-SAM checkpoint')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--sam_ckpt', type=str, default='/data/zireal/W1_manipulate/baselines/BoxPromtIML/sam_vit_h_4b8939.pth')
    parser.add_argument('--data_root', type=str, default='/data/zireal/W1_manipulate/data/val_all_set')
    parser.add_argument('--output_dir', type=str, default='./masks/')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    model = SAM_MoE_Forgery(args.sam_type, args.sam_ckpt)
    # 加载微调后的权重
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 解析 Epoch 和记录路径
    epoch = args.ckpt.split('ep')[-1].split('.')[0]
    model_name = "SAM_MoE_LoRA"
    
    # 结果存储字典
    all_results_data = {
        "meta": {
            "model": model_name,
            "epoch": epoch,
            "timestamp": str(datetime.now()),
            "checkpoint": args.ckpt
        },
        "datasets": {}
    }

    record_txt = os.path.join(os.path.dirname(args.output_dir), "val_results_moe_sam.txt")
    record_json = os.path.join(os.path.dirname(args.output_dir), "val_results_moe_sam.json") # 新增 JSON 路径

    # test_datasets = ['C1', 'Coverage', 'Columbia', 'NC16', 'Imd', 'Korus', 'In-the-Wild', 'DSO', 'CocoGlide']
    test_datasets = ['C1']
    
    f1_in, f1_out = [], []

    for i, dataset in enumerate(test_datasets):
        img_dir = os.path.join(args.data_root, dataset, 'Tp/')
        gt_dir = os.path.join(args.data_root, dataset, 'Gt/')
        
        if not os.path.exists(img_dir): continue

        # 数据加载器：必须设为 1024 对齐 SAM
        loader = test_dataset(image_root=img_dir, gt_root=gt_dir, testsize=1024)

        save_path = os.path.join(args.output_dir, model_name, f'epoch_{epoch}', dataset)
        os.makedirs(save_path, exist_ok=True)

        # 加入初始化的text
        fixed_text = clip.tokenize(["This image has a tampered region"]).to(device)

        # A. 生成 Masks
        with torch.no_grad():
            for _ in tqdm(range(loader.size), desc=f'Inference {dataset}'):
                # image: [1, 3, 1024, 1024], gt: PIL, name: str
                image, _, gt, name, _ = loader.load_data()
                orig_size = (gt.size[1], gt.size[0]) # (H, W)
                
                image = image.to(device)
                # 模型输出 [1, 1, 256, 256]
                res = model(image, fixed_text)
                
                # 获取概率图并转为 Numpy
                res = torch.sigmoid(res).cpu().numpy().squeeze() # [256, 256]
                
                # 保存为图片（内部会插值回 orig_size）
                pil_to_save(res, name, orig_size, save_path)

        # B. 评估指标
        avg_auc, avg_f1, avg_iou = evaluate_dataset(save_path, gt_dir, dataset, record_txt, epoch, model_name)

# 存入字典用于 JSON
        all_results_data["datasets"][dataset] = {
            "AUC": round(float(avg_auc), 4),
            "F1": round(float(avg_f1), 4),
            "IoU": round(float(avg_iou), 4)
        }

        if i < 4: f1_in.append(avg_f1)
        else: f1_out.append(avg_f1)

    # 计算平均分
    m_f1_in = np.mean(f1_in) if f1_in else 0
    m_f1_out = np.mean(f1_out) if f1_out else 0
    all_results_data["summary"] = {
        "f1_in_avg": round(float(m_f1_in), 4),
        "f1_out_avg": round(float(m_f1_out), 4)
    }

    # === 记录 TXT (格式美化) ===
    with open(record_txt, "a") as f:
        f.write("-" * 60 + "\n")
        f.write(f"SUMMARY FOR EPOCH {epoch}\n")
        f.write(f"In-Distribution F1 (First 4): {m_f1_in:.4f}\n")
        f.write(f"Out-of-Distribution F1 (Others): {m_f1_out:.4f}\n")
        f.write("-" * 60 + "\n\n")

    # === 记录 JSON (结构化存储) ===
    # 如果 JSON 已存在，读取并追加或直接保存为新文件
    with open(record_json, "w") as jf:
        json.dump(all_results_data, jf, indent=4)
    
    print(f"Evaluation finished. Results saved to {record_txt} and {record_json}")


# python eval_moe_sam.py \
#   --ckpt ./runs/final_moe/model_ep50.pth \
#   --sam_ckpt /data/zireal/W1_manipulate/baselines/BoxPromtIML/sam_vit_h_4b8939.pth \
#   --data_root /data/zireal/W1_manipulate/data/val_all_set
