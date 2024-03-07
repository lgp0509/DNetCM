import pandas as pd
import os
import numpy as np


def calculate_combined_statistics(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 如果存在无关的列，可以在这里删除
    # 例如：df.drop(columns=['Unnamed: 0'], inplace=True)

    # 合并所有列的数据
    combined_data = pd.concat([df[col] for col in df.columns if col.startswith('mes')], axis=0)

    # 计算统计数据
    mean = combined_data.mean()
    std_dev = combined_data.std()
    q1 = combined_data.quantile(0.25)
    median = combined_data.median()
    q3 = combined_data.quantile(0.75)

    # 打印统计结果
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Q1: {q1}")
    print(f"Median (M): {median}")
    print(f"Q3: {q3}")
    return  [mean,std_dev,median,q1,q3]

def list_direct_files(directory):
    list = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            print(item)
            list.append(item)
    return list
def list_directories(directory):
    directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return directories

# 假设文件路径如下，根据实际情况进行修改

stats = pd.DataFrame(columns = ['Mean', 'SD', 'M', 'Q1', 'Q3'])
for file_path in list_directories(r'H:\pyglcv\mes'):
    df = pd.DataFrame()
    for i, item in enumerate(list_direct_files(file_path)):
        file = file_path + "/" + item
        statlist = calculate_combined_statistics(file)
        df[f'glo{i}'] = statlist
    df = df.T
    df.columns = ['Mean', 'SD', 'M', 'Q1', 'Q3']
    df.to_excel(file_path + '.xlsx', index=True)

    slide_mean = df['Mean'].mean()
    slide_sd = df['Mean'].std()
    slide_q1 = df['Mean'].quantile(0.25)
    slide_median = df['Mean'].median()
    slide_q3 = df['Mean'].quantile(0.75)
    s = pd.Series([slide_mean,slide_sd,slide_median,slide_q1,slide_q3],index=df.columns ,name= file_path)
    stats = stats.append(s,ignore_index=False)
stats.to_excel("H:\pyglcv\output\statsmes.xlsx")