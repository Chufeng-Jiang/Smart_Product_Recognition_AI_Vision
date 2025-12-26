# convert_embedding.py
import numpy as np
import os

# 加载原始数据
input_path = "embeddings/data_embedding.npy"
output_npy = "embeddings/embedding_safe.npy"
output_npz = "embeddings/embedding_safe.npz"
output_txt = "embeddings/embedding_shape.txt"

print("转换 embedding 文件...")

try:
    # 方法1：重新保存为安全的 .npy 格式
    data = np.load(input_path, allow_pickle=True)
    print(f"原始数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    
    # 保存为安全的 float32 格式
    data_safe = data.astype(np.float32)
    np.save(output_npy, data_safe)
    print(f"✅ 已保存安全格式: {output_npy}")
    
    # 方法2：保存为 .npz（压缩格式，更安全）
    np.savez_compressed(output_npz, embedding=data_safe)
    print(f"✅ 已保存压缩格式: {output_npz}")
    
    # 保存形状信息
    with open(output_txt, 'w') as f:
        f.write(f"shape: {data_safe.shape}\n")
        f.write(f"dtype: {data_safe.dtype}\n")
        f.write(f"size: {data_safe.size}\n")
    
    print("转换完成！")
    
except Exception as e:
    print(f"❌ 转换失败: {e}")