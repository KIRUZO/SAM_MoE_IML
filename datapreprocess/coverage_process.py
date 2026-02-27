import os
import shutil


source_image_dir = r" " # COVERAGE Image
source_mask_dir = r" " # COVERAGE Mask

target_root = r" " # 输出根目录，里面会自动创建 Tp 和 Gt 两个子文件夹
target_tp_dir = os.path.join(target_root, "Tp")
target_gt_dir = os.path.join(target_root, "Gt")


os.makedirs(target_tp_dir, exist_ok=True)
os.makedirs(target_gt_dir, exist_ok=True)

print(f"正在处理 COVERAGE 数据集...")

# 遍历所有图片
cnt = 0
for filename in os.listdir(source_image_dir):

    name_no_ext, ext = os.path.splitext(filename)
    
    if name_no_ext.endswith('t'):

        img_id = name_no_ext[:-1] 
        
        src_img_path = os.path.join(source_image_dir, filename)

        dst_img_path = os.path.join(target_tp_dir, filename)
        shutil.copy(src_img_path, dst_img_path)
        
        # === 处理掩膜 (Gt) ===
        # 在 mask 文件夹里找对应的 GT
        # COVERAGE 的 mask 命名通常是 '1forged.tif' 或者 '1paste.tif'
        # 优先找 forged
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