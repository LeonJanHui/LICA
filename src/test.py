import os

def count_unique_dirs(folder_path: str):
    dir_count = 0
    # 遍历一级子项
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            dir_count += 1
    return dir_count


if __name__ == "__main__":
    target_dir = r"/home/dldx/Boltz-EGNN-Affinity/data/curated_pdbbind"
    num = count_unique_dirs(target_dir)
    print(f"当前目录下一级文件夹总数：{num}")