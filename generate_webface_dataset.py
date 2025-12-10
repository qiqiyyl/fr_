#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 webface_112x112 数据集转换为 demo.txt 格式
格式：./webface_112x112/id_XXXX/XXXX_*.jpg XXXX
"""

import os
import glob
import re
from pathlib import Path


def natural_sort_key(text):
    """
    自然排序键函数，用于正确排序包含数字的字符串
    例如：'0_2.jpg' 会在 '0_10.jpg' 之前
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', text)]


def generate_dataset_file(webface_dir='./archive/webface_112x112', output_file='./dataset/webface.txt'):
    """
    生成数据集文件
    
    Args:
        webface_dir: webface_112x112 文件夹路径
        output_file: 输出文件路径
    """
    webface_path = Path(webface_dir)
    
    if not webface_path.exists():
        print(f"错误：文件夹 {webface_dir} 不存在")
        return
    
    # 获取所有 id_* 文件夹
    id_folders = sorted([d for d in webface_path.iterdir() if d.is_dir() and d.name.startswith('id_')])
    
    print(f"找到 {len(id_folders)} 个ID文件夹")
    
    # 创建输出目录
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 收集所有图片路径和标签（使用元组以便排序）
    dataset_items = []
    total_images = 0
    
    for id_folder in id_folders:
        # 从文件夹名提取ID（id_123 -> 123）
        folder_name = id_folder.name
        try:
            label_id = int(folder_name.replace('id_', ''))
        except ValueError:
            print(f"警告：无法解析文件夹名 {folder_name}，跳过")
            continue
        
        # 获取该文件夹下的所有jpg图片
        image_files = sorted(glob.glob(str(id_folder / '*.jpg')))
        
        if not image_files:
            print(f"警告：文件夹 {folder_name} 中没有找到jpg图片")
            continue
        
        # 生成相对路径（相对于项目根目录）
        for img_file in image_files:
            img_name = Path(img_file).name
            # 直接构建相对路径：./webface_112x112/id_XXXX/XXXX_*.jpg
            # 将webface_dir转换为相对路径格式
            webface_rel = webface_dir.replace('\\', '/')
            if not webface_rel.startswith('./'):
                webface_rel = './' + webface_rel.lstrip('./')
            
            relative_path = f"{webface_rel}/{id_folder.name}/{img_name}"
            # 存储为元组 (label_id, img_name, line) 以便排序
            dataset_items.append((label_id, img_name, f"{relative_path} {label_id}\n"))
            total_images += 1
        
        if len(id_folders) > 100 and len(dataset_items) % 10000 == 0:
            print(f"已处理 {len(dataset_items)} 张图片...")
    
    # 排序：先按标签ID，再按文件名（使用自然排序）
    print(f"正在排序 {total_images} 条记录...")
    dataset_items.sort(key=lambda x: (x[0], natural_sort_key(x[1])))
    
    # 提取排序后的行
    dataset_lines = [item[2] for item in dataset_items]
    
    # 写入文件
    print(f"正在写入 {total_images} 条记录到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(dataset_lines)
    
    print(f"完成！共生成 {total_images} 条记录")
    print(f"输出文件：{output_file}")


if __name__ == '__main__':
    # 可以修改这些路径
    webface_dir = './archive/webface_112x112'
    output_file = './dataset/webface.txt'
    
    generate_dataset_file(webface_dir, output_file)

