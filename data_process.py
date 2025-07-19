import os

# 设置文件夹路径
base_dir = 'documents'
folders = [
    'FDM Process Parameters',
    'General Knowledge of PLA and Fillers',
    'General Knowledge of PLA Itself'
]

# 收集每个子文件夹下的pdf文件名
files_dict = {}
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    files_dict[folder] = set(pdf_files)

# 取出各自的文件集
fdm_files = files_dict['FDM Process Parameters']
filler_files = files_dict['General Knowledge of PLA and Fillers']
itself_files = files_dict['General Knowledge of PLA Itself']

# FDM 和 Filler 共同的文件
fdm_filler_common = fdm_files & filler_files

# FDM 和 Itself 共同的文件
fdm_itself_common = fdm_files & itself_files

# Filler 和 Itself 共同的文件
filler_itself_common = filler_files & itself_files

# 三者共同的文件
all_three_common = fdm_files & filler_files & itself_files

print('FDM 和 Filler 共同的文件:')
print(fdm_filler_common, "length:", len(fdm_filler_common))
print('\nFDM 和 Itself 共同的文件:')
print(fdm_itself_common, "length:", len(fdm_itself_common))
print('\nFiller 和 Itself 共同的文件:')
print(filler_itself_common, "length:", len(filler_itself_common))

###
print('\n三者共同的文件:')
print(all_three_common)


import os
import pandas as pd

base_dir = 'documents'
folders = [
    'FDM Process Parameters',
    'General Knowledge of PLA and Fillers',
    'General Knowledge of PLA Itself'
]
# 文件类别前缀映射
prefix_map = {
    'FDM Process Parameters': 'FDM',
    'General Knowledge of PLA and Fillers': 'PLA_Filler',
    'General Knowledge of PLA Itself': 'PLA_Itself'
}

# 收集每个文件在哪些文件夹出现
file_locations = dict()
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.pdf'):
            continue
        if fname.lower() in ['review_1.pdf', 'review_2.pdf', 'review_3.pdf']:
            continue  # 忽略 review_x 文件
        file_locations.setdefault(fname, set()).add(folder)

# 统计每种组合
category_count = dict()
for v in file_locations.values():
    key = tuple(sorted([prefix_map[f] for f in v]))
    category_count[key] = category_count.get(key, 0) + 1

# 生成新文件名映射
counters = dict()  # 各类计数器
filename_map = dict()

for fname, folders_set in file_locations.items():
    categories = sorted([prefix_map[f] for f in folders_set])
    prefix = '_'.join(categories)
    # 新文件名计数
    counters.setdefault(prefix, 0)
    counters[prefix] += 1
    new_name = f"{prefix}_{counters[prefix]}.pdf"
    filename_map[fname] = new_name

# 转为DataFrame方便查看
df = pd.DataFrame([
    {'original_name': k, 'new_name': v}
    for k, v in filename_map.items()
])
print(df)

# 可选：保存映射表
df.to_csv('filename_mapping.csv', index=False, encoding='utf-8-sig')





####################################################################################################
import os

base_dir = 'documents'
folders = [
    'FDM Process Parameters',
    'General Knowledge of PLA and Fillers',
    'General Knowledge of PLA Itself'
]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"{folder} 文件夹下 PDF 文件数量：{len(pdf_files)}")
total = 0
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"{folder} 文件夹下 PDF 文件数量：{len(pdf_files)}")
    total += len(pdf_files)
print(f"三大文件夹总PDF数量：{total}")


####################################################################################################



base_dir = 'documents'
folders = [
    'FDM Process Parameters',
    'General Knowledge of PLA and Fillers',
    'General Knowledge of PLA Itself'
]

# 忽略的文件
ignore_files = {'review_1.pdf', 'review_2.pdf', 'review_3.pdf'}

# 实际重命名
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.pdf'):
            continue
        if fname.lower() in ignore_files:
            continue
        if fname not in filename_map:
            print(f"未在映射表中找到：{fname}，跳过")
            continue
        old_path = os.path.join(folder_path, fname)
        new_name = filename_map[fname]
        new_path = os.path.join(folder_path, new_name)
        # 如果新文件名已存在，先删除或改名（极少见，除非映射冲突）
        if os.path.exists(new_path):
            print(f"警告：{new_name} 已存在，删除后重命名")
            os.remove(new_path)
        print(f"重命名：{old_path} -> {new_path}")
        os.rename(old_path, new_path)


########################################################################################################################################################################################################

import os

base_dir = 'documents'
folders = [
    'FDM Process Parameters',
    'General Knowledge of PLA and Fillers',
    'General Knowledge of PLA Itself'
]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"{folder} 文件夹下 PDF 文件数量：{len(pdf_files)}")
total = 0
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"{folder} 文件夹下 PDF 文件数量：{len(pdf_files)}")
    total += len(pdf_files)
print(f"三大文件夹总PDF数量：{total}")


