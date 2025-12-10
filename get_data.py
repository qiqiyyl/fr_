import kagglehub
import os
import shutil

# 1. 用 kagglehub 下载（会放到它自己的 cache 下）
cache_path = kagglehub.dataset_download("yakhyokhuja/webface-112x112")
print("Cache path:", cache_path)

# 2. 你想要的目标目录
target_root = "./data"
target_path = os.path.join(target_root, "webface-112x112")

os.makedirs(target_root, exist_ok=True)

# 3. 如果目标目录已存在，按需要决定是否删除 / 覆盖
if os.path.exists(target_path):
    print(f"Target path {target_path} already exists, removing it...")
    shutil.rmtree(target_path)

# 4. 把整个数据集目录复制过去
shutil.copytree(cache_path, target_path)

print("Copied dataset to:", target_path)
