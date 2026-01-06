import os
import shutil

# ================= 配置路径 =================
# 原始下载解压后的路径
source_image_dir = r"E:\tsinghua_projects\RAs\work1_AI_manipulation_detection\Datasets\COVERAGE\image" 
source_mask_dir = r"E:\tsinghua_projects\RAs\work1_AI_manipulation_detection\Datasets\COVERAGE\mask" 

# 你想存放整理好数据的路径
target_root = r"E:\tsinghua_projects\RAs\work1_AI_manipulation_detection\Datasets\A_Processed/COVERAGE"
target_tp_dir = os.path.join(target_root, "Tp")
target_gt_dir = os.path.join(target_root, "Gt")

# ===========================================

os.makedirs(target_tp_dir, exist_ok=True)
os.makedirs(target_gt_dir, exist_ok=True)

print(f"正在处理 COVERAGE 数据集...")

# 遍历所有图片
cnt = 0
for filename in os.listdir(source_image_dir):
    # 1. 只筛选篡改图 (文件名以 't' 结尾，例如 '1t.tif')
    # 注意：有的文件名可能是 '1t.jpg' 或 '1t.tif'，根据实际情况调整
    name_no_ext, ext = os.path.splitext(filename)
    
    if name_no_ext.endswith('t'):
        # 提取数字ID，例如 '1t' -> '1'
        img_id = name_no_ext[:-1] 
        
        # === 处理图片 (Tp) ===
        src_img_path = os.path.join(source_image_dir, filename)
        # 为了方便，我们可以统一重命名为 1.tif, 2.tif... 或者保持 1t.tif
        # 这里建议保持 1t.tif 以免和原图混淆
        dst_img_path = os.path.join(target_tp_dir, filename)
        shutil.copy(src_img_path, dst_img_path)
        
        # === 处理掩膜 (Gt) ===
        # 在 mask 文件夹里找对应的 GT
        # COVERAGE 的 mask 命名通常是 '1forged.tif' 或者 '1paste.tif'
        # 我们优先找 forged
        mask_candidates = [
            f"{img_id}forged.tif", 
            f"{img_id}paste.tif",
            f"{img_id}forged.jpg",
            f"{img_id}paste.jpg"
        ]
        
        found_mask = False
        for mask_name in mask_candidates:
            src_mask_path = os.path.join(source_mask_dir, mask_name)
            if os.path.exists(src_mask_path):
                # 关键：Mask 的文件名必须和 Tp 的文件名一致（扩展名可以不同，最好一致）
                # Tp叫 '1t.tif'，Gt 也得叫 '1t.tif' (或者 1t.png)
                dst_mask_name = filename 
                dst_mask_path = os.path.join(target_gt_dir, dst_mask_name)
                
                shutil.copy(src_mask_path, dst_mask_path)
                found_mask = True
                break
        
        if found_mask:
            cnt += 1
            if cnt % 10 == 0:
                print(f"已处理 {cnt} 对: {filename} <-> {mask_name}")
        else:
            print(f"[警告] 找不到对应的 Mask: {filename} (ID: {img_id})")

print(f"完成！共整理了 {cnt} 对数据存放到 {target_root}")