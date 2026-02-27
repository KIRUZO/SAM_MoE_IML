import os
import csv
import shutil
import cv2
import numpy as np


csv_path = r" " # NIST16 CSV

# ==========================================

# 1. 自动反推数据根目录
# CSV 在: .../reference/manipulation/CSV
ref_dir = os.path.dirname(csv_path)       # .../manipulation
ref_parent = os.path.dirname(ref_dir)     # .../reference
dataset_root = os.path.dirname(ref_parent)# .../NC2016_Test0613 (根目录)

# 2. 设置输出目录 (存放在数据集同级目录下)
output_root = os.path.join(os.path.dirname(dataset_root), "NIST16_BoxPromptIML")
target_tp_dir = os.path.join(output_root, "Tp")
target_gt_dir = os.path.join(output_root, "Gt")

# 检查文件是否存在
if not os.path.exists(csv_path):
    print(f"❌ 错误：找不到文件！请检查路径是否完全正确：\n{csv_path}")
    exit()

print(f"✅ 找到索引文件: {os.path.basename(csv_path)}")
print(f"📂 推断根目录: {dataset_root}")
print(f"💾 输出目录: {output_root}")

os.makedirs(target_tp_dir, exist_ok=True)
os.makedirs(target_gt_dir, exist_ok=True)

# 3. 开始提取
count = 0
skip_count = 0

try:
    # 尝试 utf-8 读取，如果报错则尝试 latin-1
    f = open(csv_path, 'r', encoding='utf-8')
    reader = csv.DictReader(f, delimiter='|')
except:
    f = open(csv_path, 'r', encoding='latin-1')
    reader = csv.DictReader(f, delimiter='|')

print("🚀 开始匹配并提取数据...")

for row in reader:
    # 清理空格
    row = {k.strip(): v.strip() for k, v in row.items()}
    
    # 筛选：只提取 IsTarget = True 的 (篡改图)
    # CSV里可能是 'Y' 或者 'True'，都兼容一下
    is_target = row.get('IsTarget', 'N')
    if is_target not in ['True', 'Y', 'TRUE']:
        continue

    file_id = row['ProbeFileID']
    
    # 获取相对路径 (去掉开头的 / 或 \)
    img_rel = row['ProbeFileName'].lstrip('/').lstrip('\\') 
    mask_rel = row['ProbeMaskFileName'].lstrip('/').lstrip('\\')
    
    # 拼接绝对路径
    src_img = os.path.join(dataset_root, img_rel)
    src_mask = os.path.join(dataset_root, mask_rel)
    
    # 检查原图是否存在
    if not os.path.exists(src_img):
        # 调试信息：如果大量找不到，可能是根目录推断错了
        if skip_count < 3: 
            print(f"[跳过] 原图不存在: {src_img}")
        skip_count += 1
        continue

    # 目标文件名
    dst_img_name = file_id + ".jpg"
    dst_mask_name = file_id + ".png"
    
    # === 1. 复制原图 (Tp) ===
    shutil.copy(src_img, os.path.join(target_tp_dir, dst_img_name))
    
    # === 2. 处理 Mask (Gt) ===
    # 读取图片 (保持原格式读取)
    mask_data = cv2.imread(src_mask, cv2.IMREAD_UNCHANGED)
    
    if mask_data is None:
        print(f"[警告] Mask文件损坏或路径错误: {src_mask}")
        continue
        
    # 如果是多通道（RGB），转灰度
    if len(mask_data.shape) == 3:
        mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGR2GRAY)
    
    # 二值化 (非黑即白)
    # NIST Mask 有时候会有杂色，大于1的都算篡改(255)
    _, binary_mask = cv2.threshold(mask_data, 1, 255, cv2.THRESH_BINARY)
    
    # 保存为 PNG
    cv2.imwrite(os.path.join(target_gt_dir, dst_mask_name), binary_mask)
    
    count += 1
    if count % 50 == 0:
        print(f"已处理 {count} 张...")

f.close()
print("-" * 30)
print(f"🎉 提取完成！")
print(f"成功提取: {count} 对")
if skip_count > 0:
    print(f"跳过缺失文件: {skip_count} 个")
print(f"数据已保存在: {output_root}")