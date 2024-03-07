import os
import pandas as pd
from collections import defaultdict

# 假设top_directory是你的顶级目录路径
top_directory = 'H:/pyglcv/result_glo'

# 初始化一个字典来保存统计结果
sample_stats = defaultdict(lambda: defaultdict(int))

# 遍历顶级目录下的每个样本文件夹
for sample_name in os.listdir(top_directory):
    sample_path = os.path.join(top_directory, sample_name)
    if os.path.isdir(sample_path):
        # 遍历样本文件夹下的每张图片
        for image_file in os.listdir(sample_path):
            # 解析文件名以获取图片类型
            image_parts = image_file.split('_')
            if len(image_parts) >= 4:
                image_type = image_parts[0]
                sample_stats[sample_name][image_type] += 1

# 创建一个空的DataFrame，列为样本名称和所有可能的图片类型
all_types = set()
for stats in sample_stats.values():
    all_types.update(stats.keys())
columns = ['样本名称'] + sorted(list(all_types))

# 准备数据填充DataFrame
data = []
for sample_name, types in sample_stats.items():
    row = [sample_name] + [types.get(image_type, 0) for image_type in columns[1:]]
    data.append(row)

df = pd.DataFrame(data, columns=columns)

# 保存DataFrame到Excel文件
excel_path = 'H:/pyglcv/stats.xlsx'  # 指定Excel文件保存路径
df.to_excel(excel_path, index=False, engine='openpyxl')

print(f"统计结果已保存到 '{excel_path}'")
