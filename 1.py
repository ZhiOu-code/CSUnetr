import nibabel as nib
import numpy as np

label_path = "M:/Output/1/label0070.nii.gz"
label_nii = nib.load(label_path)

# 1. 查看标签的体素维度
print("标签 shape (X, Y, Z):", label_nii.shape)

# 2. 查看 affine（空间方向、位置、分辨率）
print("标签 affine 矩阵:\n", label_nii.affine)

# 3. 获取原始数据内容（物理空间中是 X, Y, Z）
label_data = label_nii.get_fdata()
