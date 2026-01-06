import os
import shutil
import cv2
import numpy as np

# ================= 配置路径 =================
# 1. 你的 IMD2020 原始文件夹路径 (包含几百个子文件夹的那个目录)
source_root = r"E:\tsinghua_projects\RAs\work1_AI_manipulation_detection\Datasets\IMD2020" 

# 2. 目标输出路径
output_root = r"E:\tsinghua_projects\RAs\work1_AI_manipulation_detection\Datasets\A_Processed\val_all_set\IMD"
target_tp_dir = os.path.join(output_root, "Tp")
target_gt_dir = os.path.join(output_root, "Gt")
# ===========================================

# 创建目标目录
os.makedirs(target_tp_dir, exist_ok=True)
os.makedirs(target_gt_dir, exist_ok=True)

print(f"正在扫描 IMD2020 数据集: {source_root}")

pair_count = 0
skip_count = 0

# 遍历所有子文件夹 (os.walk 会自动钻进所有子目录)
for root, dirs, files in os.walk(source_root):
    # 筛选出当前文件夹下的所有图片文件
    # 忽略隐藏文件
    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')) and not f.startswith('.')]
    
    # === 第一步：分离角色 ===
    # 1. 忽略原图 (_orig)
    # 2. 找出 Mask (_mask)
    # 3.剩下的就是 篡改图 (Tp)
    
    candidates_tp = []
    candidates_mask = {} # 用字典存储，方便查找: {'文件名不带mask': '完整mask文件名'}
    
    for img_name in images:
        name_no_ext = os.path.splitext(img_name)[0]
        
        # 跳过原图
        if "_orig" in name_no_ext:
            continue
            
        # 收集 Mask
        if "_mask" in name_no_ext:
            # 提取属于哪个篡改图的 key。例如 'abc_0_mask' -> key是 'abc_0'
            key = name_no_ext.replace("_mask", "")
            candidates_mask[key] = img_name
        else:
            # 这大概率是篡改图
            candidates_tp.append(img_name)
            
    # === 第二步：配对处理 ===
    for tp_filename in candidates_tp:
        tp_name_no_ext = os.path.splitext(tp_filename)[0]
        
        # 检查有没有对应的 Mask
        if tp_name_no_ext in candidates_mask:
            mask_filename = candidates_mask[tp_name_no_ext]
            
            # 构建完整路径
            src_tp_path = os.path.join(root, tp_filename)
            src_gt_path = os.path.join(root, mask_filename)
            
            # === 第三步：转移和重命名 ===
            
            # 1. 处理 Tp (直接复制)
            dst_tp_name = tp_filename # 保持原名，例如 c8tt7fg_0.jpg
            shutil.copy(src_tp_path, os.path.join(target_tp_dir, dst_tp_name))
            
            # 2. 处理 Gt (读取 -> 二值化 -> 重命名保存)
            # BoxPromptIML 要求 Gt 文件名和 Tp 一致 (后缀可以不同)
            # 所以我们要把 c8tt7fg_0_mask.png 改名为 c8tt7fg_0.png
            dst_gt_name = tp_name_no_ext + ".png"
            dst_gt_path = os.path.join(target_gt_dir, dst_gt_name)
            
            # 读取 Mask
            mask_img = cv2.imread(src_gt_path, cv2.IMREAD_UNCHANGED)
            
            if mask_img is None:
                print(f"[警告] 无法读取 Mask: {src_gt_path}")
                continue
                
            # 转灰度
            if len(mask_img.shape) == 3:
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            
            # 二值化 (确保只有 0 和 255)
            # IMD2020 的 Mask 有时候边缘有抗锯齿，需要二值化
            _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            
            cv2.imwrite(dst_gt_path, binary_mask)
            
            pair_count += 1
            if pair_count % 100 == 0:
                print(f"已处理 {pair_count} 对...")
        else:
            # 只有篡改图，没有 Mask (IMD2020里很少见，但可能有)
            # print(f"[跳过] 孤立的篡改图: {tp_filename}")
            skip_count += 1

print("-" * 30)
print(f"处理完成！")
print(f"✅ 成功提取 Pairs: {pair_count}")
print(f"❌ 跳过无 Mask 图片: {skip_count}")
print(f"数据保存在: {output_root}")