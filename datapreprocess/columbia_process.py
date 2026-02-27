import os
import cv2
import numpy as np
import shutil


target_tp_dir = r"" # Tp
source_mask_dir = r" " # Columbia Mask


target_gt_dir = r"" # Output Gt
# ===========================================


if os.path.exists(target_gt_dir):
    shutil.rmtree(target_gt_dir)
os.makedirs(target_gt_dir, exist_ok=True)

print(f"基准图片目录: {target_tp_dir}")
print(f"正在进行严格匹配转换 (Target: 180)...")

success_count = 0
missing_count = 0

# === 核心逻辑：只遍历 Tp 文件夹 ===
tp_files = sorted(os.listdir(target_tp_dir))

for img_filename in tp_files:
    # 忽略非图片文件
    if not img_filename.lower().endswith(('.jpg', '.tif', '.png', '.bmp')):
        continue

    # 获取不带后缀的文件名 (例如 'canong3_canon_sub_01')
    name_no_ext = os.path.splitext(img_filename)[0]
    
    # 拼凑 Columbia 数据集的标准 Mask 文件名
    # Columbia 的规则通常是：原名 + _edgemask + .bmp
    mask_filename = name_no_ext + "_edgemask.bmp"
    mask_path = os.path.join(source_mask_dir, mask_filename)
    
    # 如果找不到 .bmp，尝试找一下 .jpg 或 .tif (防止数据集版本不同)
    if not os.path.exists(mask_path):
        for ext in ['.jpg', '.tif', '.png']:
            temp_path = os.path.join(source_mask_dir, name_no_ext + "_edgemask" + ext)
            if os.path.exists(temp_path):
                mask_path = temp_path
                break
    
    if os.path.exists(mask_path):
        img = cv2.imread(mask_path)
        
        if img is None:
            print(f"[错误] 无法读取文件: {mask_path}")
            continue


        blue_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        
        # 创建二值 Mask
        binary_mask = np.zeros_like(blue_channel)
        # 只要绿色或蓝色通道 > 100，就认为是篡改
        binary_mask[(green_channel > 100) | (blue_channel > 100)] = 255

        save_name = name_no_ext + ".png" 
        save_path = os.path.join(target_gt_dir, save_name)
        
        cv2.imwrite(save_path, binary_mask)
        success_count += 1

        
    else:
        print(f"[警告] 找不到对应的 Mask: {img_filename} (预期: {mask_filename})")
        missing_count += 1

print("-" * 30)
print(f"处理完毕！")
print(f"Tp 图片总数: {len(tp_files)}")
print(f"生成 Mask 总数: {success_count}")
print(f"缺失 Mask 总数: {missing_count}")

if success_count == len(tp_files):
    print("完美！Tp 和 Gt 数量完全一致。")
else:
    print("注意：数量仍不匹配，请检查上方输出的[警告]信息。")