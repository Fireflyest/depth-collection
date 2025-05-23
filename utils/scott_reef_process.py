#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scott Reef 水下双目深度估计处理工具
合并和优化了flat_depth_estimation.py和stereo_depth.py的功能
修复了深度异常值处理问题

使用方法:
    python scott_reef_process.py --dataset ./assets/ScottReef --calib ./assets/ScottReef/calib/ScottReef_20090804_084719.calib --save-3d
"""

import os
import sys
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
import time
from parse_calib import parse_scott_reef_calibration

# 添加Open3D依赖，用于3D点云生成
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("警告: Open3D库未安装，3D点云生成将不可用。通过 'pip install open3d' 安装")
    OPEN3D_AVAILABLE = False

def optimize_image_for_matching(img):
    """
    优化图像以提高立体匹配质量
    
    Parameters:
        img: 输入图像
    
    Returns:
        优化后的图像
    """
    # 对比度受限的自适应直方图均衡化 (CLAHE) - 增强局部对比度
    lab = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 轻微高斯模糊以减少噪声
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0.5)
    
    # 锐化以增强边缘 - 对纹理匮乏的水下场景很有帮助
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def remove_depth_outliers(depth, rate_to_set_mean=0.025, kernel_size=7, abs_threshold=1, max_depth_clamp=10.0):
    """
    移除深度图中的异常值，用周围像素的平均值替代
    
    Parameters:
        depth: 深度图
        rate_to_set_mean: 设置为均值的异常值比例
        kernel_size: 用于计算替代值的周围像素核大小
        abs_threshold: 绝对距离阈值(米)，如果提供，则使用固定距离而不是标准差的倍数
        max_depth_clamp: 强制限制最大深度值(米)，超过此值的深度会被认为是异常值
        
    Returns:
        处理后的深度图
    """
    # 首先应用最大深度限制 (针对水下场景)
    if max_depth_clamp is not None:
        extreme_mask = depth > max_depth_clamp
        if np.any(extreme_mask):
            print(f"预处理: 移除 {np.sum(extreme_mask)} 个极端深度值 (>{max_depth_clamp}m)")
            depth = depth.copy()  # 创建副本以避免修改原数组
            depth[extreme_mask] = 0  # 将极端值设为无效
    
    # 创建深度图副本
    cleaned_depth = depth.copy()
    
    # 处理无效值
    valid_mask = (depth > 0.01) & (depth < max_depth_clamp)
    if np.sum(valid_mask) < 100:
        return depth  # 如果有效点太少，直接返回原图
    
    # 计算有效深度值的全局统计信息
    valid_depths = depth[valid_mask]
    global_mean = np.mean(valid_depths)
    global_std = np.std(valid_depths)
    global_median = np.median(valid_depths)
    
    print(f"深度统计 - 均值: {global_mean:.4f}m, 中位数: {global_median:.4f}m, 标准差: {global_std:.4f}m")
    
    # 多级异常值检测
    outlier_mask = np.zeros_like(depth, dtype=bool)
    
    # 1. 使用基于阈值的检测 (根据平均值和标准差或绝对距离)
    # 使用绝对距离阈值
    print(f"使用绝对距离阈值: {abs_threshold}米")
    outlier_threshold_high = global_median + abs_threshold
    outlier_threshold_low = max(0.01, global_median - abs_threshold)
    threshold_mask = (depth > outlier_threshold_high) | ((depth < outlier_threshold_low) & valid_mask)
    outlier_mask = outlier_mask | threshold_mask
    
    # 2. 强行限制极端值 - 针对水下场景的最大合理深度
    # 改进: 使用更严格的全局阈值，基于中位数的倍数而非固定值
    max_reasonable_depth = min(5.0 * global_median, max_depth_clamp)  # 默认使用中位数的5倍作为极限值
    extreme_outliers = (depth > max_reasonable_depth) & valid_mask
    outlier_mask = outlier_mask | extreme_outliers
    
    # 3. 局部一致性检查 - 深度应当在局部区域内大致平滑
    height, width = depth.shape[:2]
    half_k = kernel_size // 2
    
    # 仅对怀疑区域进行局部一致性检验 (降低计算量)
    suspicious = np.zeros_like(depth, dtype=bool)
    suspicious[valid_mask] = np.abs(depth[valid_mask] - global_median) > global_std * 1.5
    
    y_indices, x_indices = np.where(suspicious)
    for y, x in zip(y_indices, x_indices):
        # 定义局部区域
        y_min, y_max = max(0, y - half_k), min(height, y + half_k + 1)
        x_min, x_max = max(0, x - half_k), min(width, x + half_k + 1)
        
        local_region = depth[y_min:y_max, x_min:x_max]
        local_valid = valid_mask[y_min:y_max, x_min:x_max] & ~suspicious[y_min:y_max, x_min:x_max]
        
        if np.sum(local_valid) >= 4:  # 至少需要4个有效点
            local_median = np.median(local_region[local_valid])
            local_std = np.std(local_region[local_valid])
            
            # 如果点与局部中值差异太大，标记为异常点
            if abs(depth[y, x] - local_median) > max(abs_threshold if abs_threshold else (local_std * 2.0), 0.5):
                outlier_mask[y, x] = True
    
    # 合并所有无效点到一个掩码 - 异常值和原始无效值
    invalid_mask = outlier_mask | ~valid_mask
    outlier_count = np.sum(outlier_mask)
    invalid_count = np.sum(invalid_mask)
    print(f"检测到 {outlier_count} 个异常值和 {invalid_count} 个总无效点 ({invalid_count / depth.size * 100:.2f}%)")
        
    # 使用迭代方法填充无效值
    max_iterations = 5  # 最大迭代次数
    for iteration in range(max_iterations):
        # 迭代前的无效点数量
        prev_invalid_count = np.sum(invalid_mask)
        
        if prev_invalid_count == 0:
            print(f"所有无效值已处理，停止迭代")
            break
            
        print(f"迭代 {iteration+1}/{max_iterations}: 处理 {prev_invalid_count} 个无效点")
        
        # 寻找当前可用于插值的有效点
        valid_points = ~invalid_mask & valid_mask
        if np.sum(valid_points) < 10:  # 如果有效点太少，停止迭代
            print(f"有效点数量不足，停止迭代")
            break
            
        # 获取无效点的坐标
        y_invalid, x_invalid = np.where(invalid_mask)
        
        # 按区域处理，避免处理太大的数组导致内存问题
        chunk_size = 10000  # 每次处理的点数
        for chunk_start in range(0, len(y_invalid), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(y_invalid))
            current_chunk = slice(chunk_start, chunk_end)
            
            y_chunk = y_invalid[current_chunk]
            x_chunk = x_invalid[current_chunk]
            
            # 对每个无效点应用局部插值
            for i, (y, x) in enumerate(zip(y_chunk, x_chunk)):
                # 局部区域
                y_min, y_max = max(0, y - half_k), min(height, y + half_k + 1)
                x_min, x_max = max(0, x - half_k), min(width, x + half_k + 1)
                
                # 局部有效区域
                local_region = depth[y_min:y_max, x_min:x_max]
                local_valid = ~invalid_mask[y_min:y_max, x_min:x_max] & valid_mask[y_min:y_max, x_min:x_max]
                
                # 如果局部区域有足够的有效点，使用中值填充
                if np.sum(local_valid) >= 3:
                    replacement_value = np.median(local_region[local_valid])
                    cleaned_depth[y, x] = replacement_value
                    invalid_mask[y, x] = False  # 标记为已处理
        
        # 检查这轮迭代的效果
        current_invalid_count = np.sum(invalid_mask)
        points_filled = prev_invalid_count - current_invalid_count
        
        print(f"  - 本轮填充了 {points_filled} 个点，剩余 {current_invalid_count} 个无效点")
        
        # 如果太少新的点被填充，尝试扩大搜索范围
        if points_filled < current_invalid_count / 50:
            half_k = min(half_k * 2, 50)  # 增大搜索窗口但不超过合理范围
            print(f"  - 扩大搜索窗口至 {half_k*2+1}x{half_k*2+1}")
    
    # 最终检查，确保所有值都在合理范围内
    if np.sum(invalid_mask) > 0:
        remaining_invalid = np.sum(invalid_mask)
        print(f"警告: 仍有 {remaining_invalid} 个点未能处理")
        
        # 获取有效点的均值
        valid_mask = ~invalid_mask & (depth > 0.01)
        if np.sum(valid_mask) > 0:
            mean_depth = np.mean(depth[valid_mask])
            
            # 随机选择rate_to_set_mean的无效点设为均值
            invalid_y, invalid_x = np.where(invalid_mask)
            num_to_set_mean = int(len(invalid_y) * rate_to_set_mean)
            
            if num_to_set_mean > 0:
                # 随机选择索引并设置为均值
                indices = np.random.choice(len(invalid_y), num_to_set_mean, replace=False)
                for i in indices:
                    cleaned_depth[invalid_y[i], invalid_x[i]] = mean_depth
                
                # 其余设为0
                cleaned_depth[invalid_mask] = 0  # 先全部设为0
                
                print(f"  - 设置 {num_to_set_mean} 个点为均值 {mean_depth:.4f}m")
                print(f"  - 设置 {remaining_invalid - num_to_set_mean} 个点为0")
            else:
                cleaned_depth[invalid_mask] = 0
        else:
            cleaned_depth[invalid_mask] = 0

    return cleaned_depth

def preprocess_underwater_image(img):
    """
    预处理水下图像以提高匹配质量
    
    Parameters:
        img: 输入图像
    
    Returns:
        处理后的图像和灰度图
    """
    # 转换为LAB颜色空间进行处理
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 对L通道进行CLAHE处理增强对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # 对A和B通道进行直方图均衡化，增强颜色
    lab[:,:,1] = cv2.equalizeHist(lab[:,:,1])
    lab[:,:,2] = cv2.equalizeHist(lab[:,:,2])
    
    # 转回BGR空间
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 锐化处理
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 转为灰度图
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # 额外进行自适应直方图均衡化
    gray = clahe.apply(gray)
    
    return enhanced, gray

def compute_underwater_stereo_depth(left_img, right_img, baseline, focal_length, min_disp=0, max_disp=128, max_depth=10.0):
    """
    专为水下场景优化的双目立体匹配深度计算
    
    Parameters:
        left_img: 左相机图像
        right_img: 右相机图像
        baseline: 相机基线距离(米)
        focal_length: 相机焦距(像素)
        min_disp: 最小视差值
        max_disp: 最大视差值
        max_depth: 最大深度限制(米)
        
    Returns:
        深度图(米)和视差图
    """
    # 确保图像尺寸一致
    if left_img.shape != right_img.shape:
        right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
    
    # 预处理图像
    left_enhanced, left_gray = preprocess_underwater_image(left_img)
    right_enhanced, right_gray = preprocess_underwater_image(right_img)
    
    # 调整立体匹配参数
    # 使用更大的窗口尺寸和更强的平滑约束，以应对水下环境的低纹理区域
    window_size = 11  # 增大窗口尺寸以应对水下低纹理区域
    
    # 为水下图像优化的立体匹配参数
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=max_disp - min_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,  # 惩罚系数1
        P2=32 * 3 * window_size**2, # 惩罚系数2，更大的值使视差平滑
        disp12MaxDiff=2,            # 允许更大的左右一致性差异
        uniquenessRatio=5,          # 降低这个值以允许更多匹配
        speckleWindowSize=200,      # 增大窗口尺寸以过滤更大的噪声区域
        speckleRange=2,             # 视差变化阈值
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # 计算视差
    disparity = stereo.compute(left_gray, right_gray)
    disparity = disparity.astype(np.float32) / 16.0
    
    # 使用WLS滤波器进一步优化视差图
    # 使用WLS滤波器改进视差图，尤其对水下低纹理区域有效
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    right_disparity = right_matcher.compute(right_gray, left_gray)
    right_disparity = right_disparity.astype(np.float32) / 16.0
    
    # 创建WLS滤波器
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(8000)      # 平滑度参数 - 值越大视差越平滑
    wls_filter.setSigmaColor(1.5)   # 边缘保留参数
    
    # 应用WLS滤波
    filtered_disparity = wls_filter.filter(disparity, left_gray, disparity_map_right=right_disparity)
    
    # 应用额外的形态学操作填充孔洞和移除噪声
    kernel = np.ones((5, 5), np.uint8)
    filtered_disparity = cv2.morphologyEx(filtered_disparity, cv2.MORPH_CLOSE, kernel)
    
    # 应用中值滤波去除椒盐噪声
    filtered_disparity = cv2.medianBlur(filtered_disparity, 5)
    
    # 计算深度
    valid_mask = (filtered_disparity > 0.1)
    depth = np.zeros_like(filtered_disparity)
    depth[valid_mask] = baseline * focal_length / filtered_disparity[valid_mask]
    
    # 水下环境中通常深度有限，过滤过远和过近的异常值
    # 改进：首先移除明显非水下环境的深度值
    depth[depth > max_depth] = 0  # 应用最大深度限制
    depth[depth < 0.2] = 0 # 过近的点通常是错误匹配
    
    # 后处理优化 - 使用绝对距离阈值
    depth = remove_depth_outliers(depth, abs_threshold=0.5, max_depth_clamp=max_depth)
    
    # 平滑深度
    valid_depth = depth > 0
    if np.sum(valid_depth) > 100:
        # 双边滤波保持边缘的同时平滑深度
        depth_filtered = cv2.bilateralFilter(
            depth.astype(np.float32), d=7, sigmaColor=0.05, sigmaSpace=5.0
        )
        # 保留有效区域
        depth_filtered[~valid_depth] = 0
    else:
        depth_filtered = depth
    
    return depth_filtered, filtered_disparity

def process_stereo_pair(left_img_path, right_img_path, K_left, D_left, K_right, D_right, R, T, baseline, focal_length, max_depth=10.0):
    """
    处理双目图像对，计算深度图，并将结果对齐到原始左图像
    
    Parameters:
        left_img_path: 左图路径
        right_img_path: 右图路径
        K_left, D_left: 左相机内参和畸变系数
        K_right, D_right: 右相机内参和畸变系数
        R, T: 旋转矩阵和平移向量
        baseline: 相机基线距离(米)
        focal_length: 相机焦距(像素)
        max_depth: 最大深度限制(米)
        
    Returns:
        aligned_depth: 与原始左图对齐的深度图
        aligned_rgb: 原始左图
        depth: 校正空间的深度图
        disparity: 视差图
        color_depth: 彩色深度图
        rect_left: 校正后的左图
        rect_right: 校正后的右图
        R1, P1: 立体校正参数，用于对齐和可视化
    """
    # 加载图像
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        raise ValueError(f"Unable to read images: {left_img_path} or {right_img_path}")
    
    # 保存原始左图的副本
    original_left = left_img.copy()
    
    # 获取图像尺寸
    img_size = (left_img.shape[1], left_img.shape[0])
    
    # 计算立体校正参数 - 需要保存R1和P1用于后续将深度图对齐回原始图像
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, 
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, 
        alpha=0)
    
    # 对图像进行校正
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, img_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, img_size, cv2.CV_32FC1)
    
    # 重映射图像
    rect_left = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    print(f"Using parameters - Baseline distance: {baseline} meters, Focal length: {focal_length} pixels, Max depth: {max_depth} meters")
    
    # 计时开始
    start_time = time.time()
    
    # 计算深度图
    depth, disparity = compute_underwater_stereo_depth(rect_left, rect_right, baseline, focal_length, max_depth=max_depth)
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    print(f"Depth computation completed in: {elapsed_time:.3f} seconds")
    
    # 创建彩色深度图
    color_depth = colorize_depth(depth, max_depth=max_depth)
    
    # 将深度图对齐到原始左图
    print("Aligning depth map to original left image...")
    aligned_depth, aligned_rgb = align_depth_to_left_image(depth, rect_left, original_left, K_left, D_left, R1, P1)
    
    # 打印对齐后的深度图统计信息
    valid_aligned = aligned_depth > 0
    if np.sum(valid_aligned) > 0:
        print(f"Aligned depth statistics: Min={aligned_depth[valid_aligned].min():.3f}m, "
              f"Max={aligned_depth[valid_aligned].max():.3f}m, "
              f"Mean={aligned_depth[valid_aligned].mean():.3f}m, "
              f"Valid points={np.sum(valid_aligned)}")
    else:
        print("Warning: No valid values in aligned depth map!")
    
    return aligned_depth, aligned_rgb, depth, disparity, color_depth, rect_left, rect_right, R1, P1

def crop_to_valid_region(rgb_image, depth_map, padding=10):
    """
    裁剪RGB图像和深度图到深度图的有效区域（非零区域）
    
    Parameters:
        rgb_image: RGB图像
        depth_map: 深度图
        padding: 在有效区域周围添加的额外边距（像素）
        
    Returns:
        cropped_rgb: 裁剪后的RGB图像
        cropped_depth: 裁剪后的深度图
        crop_coords: 裁剪坐标 (y_min, y_max, x_min, x_max)
    """
    # 找到深度图中的有效区域（非零值）
    valid_mask = depth_map > 0
    
    # 检查是否有有效区域
    if not np.any(valid_mask):
        print("Warning: No valid depth values found, returning original images")
        return rgb_image, depth_map, (0, depth_map.shape[0], 0, depth_map.shape[1])
    
    # 找到有效区域的边界
    y_indices, x_indices = np.where(valid_mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # 计算额外的内部裁剪量（5%的区域大小）
    height = y_max - y_min
    width = x_max - x_min
    y_trim = int(height * 0.05)
    x_trim = int(width * 0.05)
    
    # 应用内部裁剪和额外边距
    y_min = max(0, y_min + y_trim - padding)
    y_max = min(depth_map.shape[0], y_max - y_trim + padding)
    x_min = max(0, x_min + x_trim - padding)
    x_max = min(depth_map.shape[1], x_max - x_trim + padding)
    
    # 确保裁剪区域至少有最小尺寸
    if (y_max - y_min) < 10 or (x_max - x_min) < 10:
        print("Warning: Crop region too small after trimming, using original valid region")
        y_min, y_max = max(0, np.min(y_indices) - padding), min(depth_map.shape[0], np.max(y_indices) + padding)
        x_min, x_max = max(0, np.min(x_indices) - padding), min(depth_map.shape[1], np.max(x_indices) + padding)
    
    # 执行裁剪
    cropped_depth = depth_map[y_min:y_max, x_min:x_max]
    cropped_rgb = rgb_image[y_min:y_max, x_min:x_max]
    
    # 计算有效像素百分比
    valid_percentage = np.sum(cropped_depth > 0) / cropped_depth.size * 100
    
    print(f"Cropped to valid region with 5% inset: [{x_min}:{x_max}, {y_min}:{y_max}] - " 
          f"Size: {cropped_rgb.shape[1]}x{cropped_rgb.shape[0]}, "
          f"Valid depth pixels: {valid_percentage:.2f}%")
    
    return cropped_rgb, cropped_depth, (y_min, y_max, x_min, x_max)

def load_stereo_calibration(calib_file):
    """
    加载双目相机标定文件
    
    Parameters:
        calib_file: 标定文件路径
        
    Returns:
        标定参数 (内参矩阵, 畸变系数, 旋转矩阵, 平移向量, 基线距离, 焦距)
    """
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file does not exist: {calib_file}")
    
    # 检查文件扩展名，判断是否是Scott Reef格式
    if calib_file.endswith('.calib'):
        print("Detected Scott Reef calibration file format, using specialized parser...")
        try:
            calib_data = parse_scott_reef_calibration(calib_file)
            
            # 提取参数
            if len(calib_data['cameras']) < 2:
                raise ValueError("Insufficient number of cameras in calibration file")
            
            cam1 = calib_data['cameras'][0]
            cam2 = calib_data['cameras'][1]
            
            K_left = cam1['intrinsic']
            D_left = cam1['distortion'].reshape(-1, 1)
            
            K_right = cam2['intrinsic']
            D_right = cam2['distortion'].reshape(-1, 1)
            
            # 如果第二个相机相对于第一个相机有旋转和平移
            R = cam2['rotation']  # 右相机相对于左相机的旋转矩阵
            T = cam2['translation'].reshape(3, 1)  # 右相机相对于左相机的平移向量
            
            baseline = calib_data['baseline']
            focal_length = (cam1['fx'] + cam2['fx']) / 2
            
            print(f"Successfully parsed Scott Reef calibration file")
            print(f"Baseline distance: {baseline} meters, Average focal length: {focal_length} pixels")
            
            return K_left, D_left, K_right, D_right, R, T, baseline, focal_length
            
        except Exception as e:
            print(f"Error parsing Scott Reef calibration file: {e}")
            raise
    
    # 标准OpenCV标定文件格式
    try:
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        
        # 读取左相机内参和畸变系数
        K_left = fs.getNode("K1").mat()
        D_left = fs.getNode("D1").mat()
        
        # 读取右相机内参和畸变系数
        K_right = fs.getNode("K2").mat()
        D_right = fs.getNode("D2").mat()
        
        # 读取两相机之间的旋转矩阵和平移向量
        R = fs.getNode("R").mat()
        T = fs.getNode("T").mat()
        
        # 计算基线距离 (平移向量的长度)
        baseline = abs(T[0, 0])
        
        # 获取焦距 (通常使用两相机平均值)
        focal_length = (K_left[0, 0] + K_right[0, 0]) / 2
        
        fs.release()
        
        return K_left, D_left, K_right, D_right, R, T, baseline, focal_length
    except Exception as e:
        print(f"Error parsing standard calibration file: {e}")
        raise

def rectify_stereo_images(left_img, right_img, K_left, D_left, K_right, D_right, R, T):
    """
    对双目图像对进行畸变校正和立体校正
    
    Parameters:
        left_img: 左相机图像
        right_img: 右相机图像
        K_left, D_left: 左相机内参和畸变系数
        K_right, D_right: 右相机内参和畸变系数
        R, T: 旋转矩阵和平移向量
        
    Returns:
        校正后的左右图像
    """
    # 获取图像尺寸
    img_size = (left_img.shape[1], left_img.shape[0])
    
    # 计算立体校正参数
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, 
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, 
        alpha=0)
    
    # 计算重映射查找表
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, img_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, img_size, cv2.CV_32FC1)
    
    # 重映射图像
    rect_left = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    return rect_left, rect_right

def draw_epipolar_lines(left_img, right_img, interval=50):
    """
    在立体校正后的图像上绘制极线，以验证校正效果
    
    Parameters:
        left_img: 校正后的左图
        right_img: 校正后的右图
        interval: 极线间隔(像素)
        
    Returns:
        带有极线的拼接图像
    """
    h, w = left_img.shape[:2]
    
    # 创建拼接图像
    vis = np.zeros((h, w*2, 3), np.uint8)
    vis[:, :w] = left_img
    vis[:, w:] = right_img
    
    # 绘制极线
    for y in range(0, h, interval):
        color = (0, 255, 0)  # 绿色
        cv2.line(vis, (0, y), (w*2, y), color, 1)
        
    return vis

def colorize_depth(depth, min_depth=0.1, max_depth=5.0):
    """
    将深度图可视化为彩色图像
    
    Parameters:
        depth: 深度图
        min_depth: 最小深度值
        max_depth: 最大深度值
        
    Returns:
        彩色深度图
    """
    # 打印深度统计信息以便调试
    nonzero = depth > 0
    if np.sum(nonzero) > 0:
        print(f"Depth statistics: Min={depth[nonzero].min():.3f}m, Max={depth[nonzero].max():.3f}m, "
              f"Mean={depth[nonzero].mean():.3f}m, Valid points={np.sum(nonzero)}")
    else:
        print("Warning: No valid values in depth map!")
    
    # 应用更宽松的有效深度范围，以适应水下场景
    valid_mask = (depth > min_depth) & (depth < max_depth)
    valid_count = np.sum(valid_mask)
    print(f"Valid points in current depth range [{min_depth}-{max_depth}] meters: {valid_count} ({valid_count/depth.size*100:.2f}%)")
    
    # 即使有效点很少，也尝试生成有意义的可视化（降低阈值）
    if valid_count < 10:  # 极少有效点，生成灰色图像
        print("Warning: Too few valid depth points, generating gray image")
        color_depth = np.ones((depth.shape[0], depth.shape[1], 3), dtype=np.uint8) * 128
        return color_depth
    
    # 提取有效深度范围
    valid_depths = depth[valid_mask]
    d_min = max(min_depth, valid_depths.min())
    d_max = min(max_depth, valid_depths.max())
    print(f"Valid depth range: {d_min:.3f}m - {d_max:.3f}m")
    
    # 归一化深度值到0-1
    depth_norm = np.zeros_like(depth)
    # 避免除以零
    depth_range = d_max - d_min
    if depth_range < 1e-5:
        depth_range = 1.0
    
    depth_norm[valid_mask] = (depth[valid_mask] - d_min) / depth_range
    
    # 创建颜色映射（使用jet色彩映射更容易区分深度）
    cm = plt.get_cmap('jet')  # jet显示深度更加明显
    color_depth = cm(depth_norm)[:, :, :3]  # 取RGB部分
    
    # 转换为8位图像
    color_depth = (color_depth * 255).astype(np.uint8)
    
    # 将无效区域设为黑色
    color_depth[~valid_mask] = [0, 0, 0]
    
    return color_depth

def generate_point_cloud(rgb_image, depth_map, focal_length, cx=None, cy=None, max_depth=5.0):
    """
    从RGB图像和深度图生成点云
    
    Parameters:
        rgb_image: RGB图像
        depth_map: 深度图
        focal_length: 相机焦距(像素)
        cx, cy: 光学中心(如果为None则使用图像中心)
        max_depth: 最大深度值(米)
        
    Returns:
        点云对象(Open3D PointCloud)
    """
    if not OPEN3D_AVAILABLE:
        print("Error: Open3D library is required to generate point cloud")
        return None
        
    height, width = depth_map.shape
    
    # 使用图像中心作为主点(如果未指定)
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    
    # 创建网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 转换图像坐标到相机坐标
    X = (x - cx) * depth_map / focal_length
    Y = (y - cy) * depth_map / focal_length
    Z = depth_map
    
    # 过滤无效深度值
    mask = (Z > 0) & (Z < max_depth)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]
    
    # 从RGB图像提取颜色
    colors = rgb_image[mask]
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([X, Y, Z]))
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 归一化颜色值到[0,1]
    
    return pcd

def create_overlay_visualization(rgb_image, depth_map, max_depth=5.0, alpha=0.5):
    """
    创建RGB图像和深度图叠加的可视化效果图
    
    Parameters:
        rgb_image: RGB图像
        depth_map: 深度图
        max_depth: 最大深度值(米)，用于深度图归一化
        alpha: 叠加透明度，控制深度图在叠加图中的权重
        
    Returns:
        overlay_image: RGB和深度图叠加的效果图
    """
    # 确保图像尺寸一致
    assert rgb_image.shape[:2] == depth_map.shape, "RGB image and depth map must have the same dimensions"
    
    # 创建彩色深度图
    depth_colored = colorize_depth(depth_map, max_depth=max_depth)
    
    # 将BGR转为RGB (如果需要)
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        rgb_vis = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_vis = rgb_image.copy()
    
    # 创建叠加图像 (深度图叠加在RGB图像上)
    overlay = cv2.addWeighted(
        depth_colored, alpha, 
        rgb_vis, 1 - alpha, 
        0
    )
    
    # 创建带有深度轮廓的RGB图像
    # 提取深度边缘
    valid_mask = depth_map > 0
    depth_edges = np.zeros_like(depth_map, dtype=np.uint8)
    if np.sum(valid_mask) > 0:
        # 归一化深度图用于边缘检测
        norm_depth = depth_map.copy()
        norm_depth[valid_mask] = ((norm_depth[valid_mask] - norm_depth[valid_mask].min()) / 
                                (norm_depth[valid_mask].max() - norm_depth[valid_mask].min() + 1e-6) * 255)
        norm_depth = norm_depth.astype(np.uint8)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(norm_depth, 50, 150)
        
        # 膨胀边缘以使其更明显
        kernel = np.ones((2, 2), np.uint8)
        depth_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 在RGB图像上绘制深度边缘
    contour_overlay = rgb_vis.copy()
    contour_overlay[depth_edges > 0] = [255, 0, 0]  # 红色边缘
    
    # 创建完整的可视化效果图 (水平拼接)
    h, w = rgb_image.shape[:2]
    visualization = np.zeros((h, w * 3, 3), dtype=np.uint8)
    visualization[:, :w] = rgb_vis
    visualization[:, w:2*w] = depth_colored
    visualization[:, 2*w:] = overlay
    
    # 添加分隔线
    visualization[:, w-1:w+1] = [255, 255, 255]
    visualization[:, 2*w-1:2*w+1] = [255, 255, 255]
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "RGB", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Depth", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Overlay", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
    
    return visualization, overlay, contour_overlay

def align_depth_to_left_image(depth, rect_left, original_left, K_left, D_left, R1, P1):
    """
    将立体校正后的深度图对齐回原始左图像空间
    
    Parameters:
        depth: 校正空间中的深度图
        rect_left: 校正空间中的左图
        original_left: 原始左图
        K_left, D_left: 左相机内参和畸变系数
        R1, P1: 立体校正参数
        
    Returns:
        aligned_depth: 对齐到原始左图空间的深度图
        aligned_rgb: 对原始左图进行颜色调整的彩色图像
    """
    # 获取图像尺寸
    h, w = depth.shape[:2]
    orig_h, orig_w = original_left.shape[:2]
    
    # 确保输出深度图的大小与原图一致
    if h != orig_h or w != orig_w:
        print(f"调整深度图尺寸从 {h}x{w} 到 {orig_h}x{orig_w}")
    
    # 创建映射网格
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # 这些点在校正空间中
    points_rect = np.stack((x_coords.flatten(), y_coords.flatten()), axis=-1).astype(np.float32)
    
    # 校正矩阵的逆转换
    R1_inv = np.linalg.inv(R1)
    K_inv = np.linalg.inv(K_left)
    
    # 从校正空间到相机空间的转换
    # 注意: 这里我们需要考虑P1中可能包含的平移
    fx = P1[0, 0]
    fy = P1[1, 1]
    cx = P1[0, 2]
    cy = P1[1, 2]
    
    # 创建输出图像
    aligned_depth = np.zeros((orig_h, orig_w), dtype=np.float32)
    
    # 为每个校正空间中的点计算原始图像空间中的位置
    for y in range(h):
        for x in range(w):
            if depth[y, x] <= 0:  # 跳过无效深度值
                continue
                
            # 校正空间中的点
            point_rect = np.array([(x - cx) / fx, (y - cy) / fy, 1.0]) * depth[y, x]
            
            # 将点转换回原始相机空间
            point_cam = R1_inv @ point_rect
            
            # 应用相机模型将3D点映射回2D图像
            point_2d = K_left @ point_cam
            point_2d = point_2d / point_2d[2]
            
            # 考虑镜头畸变
            x_orig, y_orig = cv2.undistortPoints(np.array([[point_2d[0], point_2d[1]]]), K_left, D_left, None, K_left).squeeze()
            
            # 检查点是否在原图范围内
            if 0 <= int(y_orig) < orig_h and 0 <= int(x_orig) < orig_w:
                aligned_depth[int(y_orig), int(x_orig)] = depth[y, x]
    
    # 填充空洞 - 使用中值滤波或附近有效值
    valid_mask = aligned_depth > 0
    if np.sum(valid_mask) > 100:  # 如果有足够多的有效点
        # 使用OpenCV的形态学操作填充小孔洞
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=1)
        holes_mask = dilated_mask > valid_mask.astype(np.uint8)
        
        # 使用最近邻插值填充孔洞
        if np.sum(holes_mask) > 0:
            # 提取有效点的坐标和值
            y_valid, x_valid = np.where(valid_mask)
            valid_points = np.stack((y_valid, x_valid), axis=1)
            valid_depths = aligned_depth[valid_mask]
            
            # 提取需要填充的点的坐标
            y_holes, x_holes = np.where(holes_mask)
            holes_coords = np.stack((y_holes, x_holes), axis=1)
            
            # 使用最近邻插值
            from scipy.interpolate import NearestNDInterpolator
            interpolator = NearestNDInterpolator(valid_points, valid_depths)
            aligned_depth[holes_mask] = interpolator(holes_coords)
    
    # 对齐的RGB图像就是原始左图
    aligned_rgb = original_left.copy()
    
    return aligned_depth, aligned_rgb

def compute_underwater_stereo_depth(left_img, right_img, baseline, focal_length, min_disp=0, max_disp=128, max_depth=10.0):
    """
    专为水下场景优化的双目立体匹配深度计算
    
    Parameters:
        left_img: 左相机图像
        right_img: 右相机图像
        baseline: 相机基线距离(米)
        focal_length: 相机焦距(像素)
        min_disp: 最小视差值
        max_disp: 最大视差值
        max_depth: 最大深度限制(米)
        
    Returns:
        深度图(米)和视差图
    """
    # 确保图像尺寸一致
    if left_img.shape != right_img.shape:
        right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
    
    # 预处理图像
    left_enhanced, left_gray = preprocess_underwater_image(left_img)
    right_enhanced, right_gray = preprocess_underwater_image(right_img)
    
    # 调整立体匹配参数
    # 使用更大的窗口尺寸和更强的平滑约束，以应对水下环境的低纹理区域
    window_size = 11  # 增大窗口尺寸以应对水下低纹理区域
    
    # 为水下图像优化的立体匹配参数
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=max_disp - min_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,  # 惩罚系数1
        P2=32 * 3 * window_size**2, # 惩罚系数2，更大的值使视差平滑
        disp12MaxDiff=2,            # 允许更大的左右一致性差异
        uniquenessRatio=5,          # 降低这个值以允许更多匹配
        speckleWindowSize=200,      # 增大窗口尺寸以过滤更大的噪声区域
        speckleRange=2,             # 视差变化阈值
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # 计算视差
    disparity = stereo.compute(left_gray, right_gray)
    disparity = disparity.astype(np.float32) / 16.0
    
    # 使用WLS滤波器进一步优化视差图
    # 使用WLS滤波器改进视差图，尤其对水下低纹理区域有效
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    right_disparity = right_matcher.compute(right_gray, left_gray)
    right_disparity = right_disparity.astype(np.float32) / 16.0
    
    # 创建WLS滤波器
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(8000)      # 平滑度参数 - 值越大视差越平滑
    wls_filter.setSigmaColor(1.5)   # 边缘保留参数
    
    # 应用WLS滤波
    filtered_disparity = wls_filter.filter(disparity, left_gray, disparity_map_right=right_disparity)
    
    # 应用额外的形态学操作填充孔洞和移除噪声
    kernel = np.ones((5, 5), np.uint8)
    filtered_disparity = cv2.morphologyEx(filtered_disparity, cv2.MORPH_CLOSE, kernel)
    
    # 应用中值滤波去除椒盐噪声
    filtered_disparity = cv2.medianBlur(filtered_disparity, 5)
    
    # 计算深度
    valid_mask = (filtered_disparity > 0.1)
    depth = np.zeros_like(filtered_disparity)
    depth[valid_mask] = baseline * focal_length / filtered_disparity[valid_mask]
    
    # 水下环境中通常深度有限，过滤过远和过近的异常值
    # 改进：首先移除明显非水下环境的深度值
    depth[depth > max_depth] = 0  # 应用最大深度限制
    depth[depth < 0.2] = 0 # 过近的点通常是错误匹配
    
    # 后处理优化 - 使用绝对距离阈值
    for i in range(10):
        depth = remove_depth_outliers(depth, rate_to_set_mean=0.025+(i*0.01), abs_threshold=0.5, max_depth_clamp=max_depth)
    depth = remove_depth_outliers(depth, rate_to_set_mean=1, abs_threshold=0.5, max_depth_clamp=max_depth)

    # 平滑深度
    valid_depth = depth > 0
    if np.sum(valid_depth) > 100:
        # 双边滤波保持边缘的同时平滑深度
        depth_filtered = cv2.bilateralFilter(
            depth.astype(np.float32), d=7, sigmaColor=0.05, sigmaSpace=5.0
        )
        # 保留有效区域
        depth_filtered[~valid_depth] = 0
    else:
        depth_filtered = depth
    
    return depth_filtered, filtered_disparity

def process_stereo_pair(left_img_path, right_img_path, K_left, D_left, K_right, D_right, R, T, baseline, focal_length, max_depth=10.0):
    """
    处理双目图像对，计算深度图，并将结果对齐到原始左图像
    
    Parameters:
        left_img_path: 左图路径
        right_img_path: 右图路径
        K_left, D_left: 左相机内参和畸变系数
        K_right, D_right: 右相机内参和畸变系数
        R, T: 旋转矩阵和平移向量
        baseline: 相机基线距离(米)
        focal_length: 相机焦距(像素)
        max_depth: 最大深度限制(米)
        
    Returns:
        aligned_depth: 与原始左图对齐的深度图
        aligned_rgb: 原始左图
        depth: 校正空间的深度图
        disparity: 视差图
        color_depth: 彩色深度图
        rect_left: 校正后的左图
        rect_right: 校正后的右图
    """
    # 加载图像
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        raise ValueError(f"Unable to read images: {left_img_path} or {right_img_path}")
    
    # 保存原始左图的副本
    original_left = left_img.copy()
    
    # 获取图像尺寸
    img_size = (left_img.shape[1], left_img.shape[0])
    
    # 计算立体校正参数 - 需要保存R1和P1用于后续将深度图对齐回原始图像
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, 
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, 
        alpha=0)
    
    # 对图像进行校正
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, img_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, img_size, cv2.CV_32FC1)
    
    # 重映射图像
    rect_left = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    print(f"Using parameters - Baseline distance: {baseline} meters, Focal length: {focal_length} pixels, Max depth: {max_depth} meters")
    
    # 计时开始
    start_time = time.time()
    
    # 计算深度图
    depth, disparity = compute_underwater_stereo_depth(rect_left, rect_right, baseline, focal_length, max_depth=max_depth)
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    print(f"Depth computation completed in: {elapsed_time:.3f} seconds")
    
    # 创建彩色深度图
    color_depth = colorize_depth(depth, max_depth=max_depth)
    
    # 将深度图对齐到原始左图
    print("Aligning depth map to original left image...")
    aligned_depth, aligned_rgb = align_depth_to_left_image(depth, rect_left, original_left, K_left, D_left, R1, P1)
    
    # 打印对齐后的深度图统计信息
    valid_aligned = aligned_depth > 0
    if np.sum(valid_aligned) > 0:
        print(f"Aligned depth statistics: Min={aligned_depth[valid_aligned].min():.3f}m, "
              f"Max={aligned_depth[valid_aligned].max():.3f}m, "
              f"Mean={aligned_depth[valid_aligned].mean():.3f}m, "
              f"Valid points={np.sum(valid_aligned)}")
    else:
        print("Warning: No valid values in aligned depth map!")
    
    return aligned_depth, aligned_rgb, depth, disparity, color_depth, rect_left, rect_right, R1, P1

def main():
    parser = argparse.ArgumentParser(description='Scott Reef Underwater Stereo Depth Estimation')
    
    # 简化后的参数 - 只需要数据集目录和校准文件
    parser.add_argument('--dataset', '-d', default='./assets/ScottReef', 
                      help='Scott Reef dataset directory, default: ./assets/ScottReef')
    parser.add_argument('--calib', '-c', default='./assets/ScottReef/calib/ScottReef_20090804_084719.calib',
                      help='Path to camera calibration file, default: ./assets/ScottReef/calib/ScottReef_20090804_084719.calib')
    parser.add_argument('--output', '-o', default='./visualizations/scottreff', 
                      help='Output directory, default: ./visualizations/scottreff/')
    parser.add_argument('--max-depth', '-m', type=float, default=5.0, 
                      help='Maximum depth value for 3D point cloud (meters)')
    parser.add_argument('--save-3d', '-s', action='store_true', 
                      help='Save 3D point cloud to PLY file')
    parser.add_argument('--save-temp', '-t', action='store_true',
                      help='Save temporary/intermediate results (depth maps, disparity images, etc.)')
    parser.add_argument('--rgb-depth-path', '-p', default='./rgb_depth_pairs',
                      help='Path to save aligned RGB-depth image pairs, default: ./rgb_depth_pairs/')
    parser.add_argument('--crop-padding', type=int, default=0,
                      help='Padding around valid depth region when cropping (pixels)')
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置RGB-深度图像对输出目录
    rgb_depth_dir = args.rgb_depth_path
    os.makedirs(rgb_depth_dir, exist_ok=True)
    rgb_dir = os.path.join(rgb_depth_dir, 'rgb')
    depth_dir = os.path.join(rgb_depth_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # 从校准文件加载相机参数
    print(f"Loading camera parameters from calibration file: {args.calib}")
    try:
        K_left, D_left, K_right, D_right, R, T, baseline, focal_length = load_stereo_calibration(args.calib)
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return
    
    # 输出相机参数
    print(f"Camera parameters loaded - Baseline: {baseline} meters, Focal length: {focal_length} pixels")
    
    # 查找数据集中的所有图像对
    if not os.path.isdir(args.dataset):
        parser.error(f"Dataset directory does not exist: {args.dataset}")
    
    # 查找匹配的图像对 - 针对Scott Reef格式 (_LC和_RM)
    left_images = []
    right_images = []
    
    # 查找所有png文件
    all_files = [f for f in os.listdir(args.dataset) if f.endswith('.png')]
    
    # 查找左图(_LC)和对应的右图(_RM)
    for file in all_files:
        if '_LC' in file:
            left_path = os.path.join(args.dataset, file)
            # 尝试找到匹配的右图
            right_file = file.replace('_LC', '_RM')
            right_path = os.path.join(args.dataset, right_file)
            
            if os.path.isfile(right_path):
                left_images.append(left_path)
                right_images.append(right_path)
    
    if not left_images:
        print(f"No image pairs found in {args.dataset}")
        return
    
    print(f"Found {len(left_images)} image pairs in {args.dataset}")
    
    # 存储所有处理后的深度图，用于后续处理或分析
    all_depths = {}
    rgb_depth_pairs = []
    
    # 处理每对图像
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"Processing image pair {i+1}/{len(left_images)}: {os.path.basename(left_path)} & {os.path.basename(right_path)}")
        
        try:
            aligned_depth, aligned_rgb, depth, disparity, color_depth, rect_left, rect_right, R1, P1 = process_stereo_pair(
                left_path, right_path, 
                K_left, D_left, K_right, D_right, R, T,
                baseline, focal_length, 
                max_depth=args.max_depth
            )
            
            # 生成基础文件名
            base_name = os.path.splitext(os.path.basename(left_path))[0]
            
            # 裁剪到深度图的有效区域
            print(f"Cropping RGB and depth images to valid depth region...")
            cropped_rgb, cropped_depth, crop_coords = crop_to_valid_region(
                aligned_rgb, aligned_depth, padding=args.crop_padding)
            
            y_min, y_max, x_min, x_max = crop_coords
            print(f"Original dimensions: {aligned_rgb.shape[1]}x{aligned_rgb.shape[0]}, "
                  f"Cropped dimensions: {cropped_rgb.shape[1]}x{cropped_rgb.shape[0]}")
            
            # 存储处理后的深度图，用于后续使用
            all_depths[base_name] = cropped_depth
            
            # 保存对齐并裁剪的RGB和深度图像对
            rgb_file = f"{base_name}.png"
            depth_file = f"{base_name}_depth.exr"  # 使用EXR格式保存浮点深度值
            
            # 保存RGB图像
            rgb_path = os.path.join(rgb_dir, rgb_file)
            cv2.imwrite(rgb_path, cropped_rgb)
            
            # 保存深度图像 (使用EXR格式保存浮点深度值)
            depth_path = os.path.join(depth_dir, depth_file)
            try:
                # 尝试保存为EXR格式
                cv2.imwrite(depth_path, cropped_depth.astype(np.float32))
                print(f"Saved cropped depth as EXR to {depth_path}")
            except Exception as e:
                # 如果EXR保存失败，使用NPY格式
                depth_file = f"{base_name}_depth.npy"
                depth_path = os.path.join(depth_dir, depth_file)
                np.save(depth_path, cropped_depth)
                print(f"Saved cropped depth as NPY to {depth_path}")
            
            # 记录RGB-深度图对的相对路径
            rgb_depth_pairs.append((rgb_file, depth_file))
            
            print(f"Saved aligned and cropped RGB-Depth pair: '{rgb_file}' and '{depth_file}'")
            
            # 根据--save-temp参数决定是否保存临时/中间结果
            if args.save_temp:
                # 定义临时文件路径
                temp_depth_path = os.path.join(output_dir, f"{base_name}_aligned_depth.npy")
                disp_path = os.path.join(output_dir, f"{base_name}_disparity.png")
                color_depth_path = os.path.join(output_dir, f"{base_name}_depth_color.png")
                
                # 保存深度数据和可视化结果
                np.save(temp_depth_path, aligned_depth)
                plt.imsave(disp_path, disparity, cmap='jet')
                cv2.imwrite(color_depth_path, cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR))
                
                # 保存裁剪区域信息
                crop_info_path = os.path.join(output_dir, f"{base_name}_crop_coords.txt")
                with open(crop_info_path, 'w') as f:
                    f.write(f"y_min: {y_min}\ny_max: {y_max}\nx_min: {x_min}\nx_max: {x_max}\n")
                
                print(f"Temporary results and crop coordinates saved to: {output_dir}/{base_name}_*")
            
            # 为了可视化，创建灰度深度图，距离摄像机越近(深度值越小)的物体越白
            valid_mask = cropped_depth > 0
            # 处理无效深度值（0值）并创建可视化深度图
            gray_depth = np.zeros_like(cropped_depth, dtype=np.uint8)
            
            # 检查是否有有效深度值
            valid_mask = cropped_depth > 0
            if np.sum(valid_mask) > 0:
                # 创建深度图副本用于处理
                processed_depth = cropped_depth.copy()
                
                # 计算有效深度的均值和标准差，用于检测异常值
                mean_depth = np.mean(processed_depth[valid_mask])
                
                # 标记偏离均值过大的深度值为无效值（大于0.5米）
                deviation_mask = np.abs(processed_depth - mean_depth) > 0.3
                combined_invalid_mask = (~valid_mask) | (deviation_mask & valid_mask)
                
                # 多轮迭代填补无效区域
                max_iterations = 5  # 最大迭代次数
                for iteration in range(max_iterations):
                    # 当前无效点数量
                    invalid_count = np.sum(combined_invalid_mask)
                        
                    # 重新定义有效点（剔除离群点后）
                    y_valid, x_valid = np.where(~combined_invalid_mask)
                    valid_points = np.column_stack((y_valid, x_valid))
                    valid_values = processed_depth[~combined_invalid_mask]
                    
                    # 找出需要填充的点
                    y_fill, x_fill = np.where(combined_invalid_mask)
                    fill_points = np.column_stack((y_fill, x_fill))
                    
                    if len(valid_points) > 0 and len(fill_points) > 0:
                        # 使用最近邻插值填充无效区域
                        interp = NearestNDInterpolator(valid_points, valid_values)
                        processed_depth[combined_invalid_mask] = interp(fill_points)
                    
                    # 更新无效区域掩码 - 现在只考虑深度为零的区域
                    combined_invalid_mask = processed_depth <= 0
                    
                    # 输出填充进度
                    new_invalid_count = np.sum(combined_invalid_mask)
                    if new_invalid_count == invalid_count:
                        # 如果无效点数量没有减少，停止迭代
                        print(f"深度填充停止于第{iteration+1}轮迭代，剩余{new_invalid_count}个无效点")
                        break
                    else:
                        print(f"深度填充第{iteration+1}轮：已填充{invalid_count-new_invalid_count}个点，"
                              f"剩余{new_invalid_count}个无效点")
                
                # 归一化深度值（倒置，使得近处为白色，远处为黑色）
                valid_mask = processed_depth > 0
                if np.sum(valid_mask) > 0:
                    depth_min = np.min(processed_depth[valid_mask])
                    depth_max = np.min([args.max_depth, np.max(processed_depth[valid_mask])])
                    depth_range = depth_max - depth_min
                    
                    if depth_range > 0:
                        # 归一化整个深度图
                        normalized_depth = np.zeros_like(processed_depth)
                        normalized_depth[valid_mask] = 255 * (1.0 - (processed_depth[valid_mask] - depth_min) / depth_range)
                        
                        # 转换为8位单通道
                        gray_depth = normalized_depth.astype(np.uint8)
                else:
                    gray_depth = np.zeros_like(processed_depth, dtype=np.uint8)
            else:
                # 如果没有有效深度，创建一个空的灰度图
                gray_depth = np.zeros_like(cropped_depth, dtype=np.uint8)
                
            # 转换为三通道灰度图像
            gray_depth = np.repeat(gray_depth[..., np.newaxis], 3, axis=-1)
            
            # 保存灰度深度图到RGB-深度对目录
            viz_path = os.path.join(depth_dir, f"{base_name}_depth_viz.png")
            cv2.imwrite(viz_path, gray_depth)
            
            # 3D点云文件总是单独控制
            if args.save_3d:
                # 生成点云 - 使用裁剪后的RGB和深度图
                pcd = generate_point_cloud(cropped_rgb, cropped_depth, focal_length, max_depth=args.max_depth)
                
                if pcd is not None:
                    # 保存点云
                    pcd_path = os.path.join(output_dir, f"{base_name}_point_cloud.ply")
                    o3d.io.write_point_cloud(pcd_path, pcd)
                    print(f"3D point cloud saved to: {pcd_path}")
            
            # 显示结果 - 总是显示第一个结果，其他仅保存
            if i == 0:
                # 创建六面板可视化图：包括原始和裁剪后的RGB与深度图
                plt.figure(figsize=(18, 12))
                
                plt.subplot(231)
                plt.title("Original RGB Image")
                plt.imshow(cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2RGB))
                
                plt.subplot(232)
                plt.title("Original Aligned Depth Map")
                orig_colored_depth = colorize_depth(aligned_depth, max_depth=args.max_depth)
                plt.imshow(orig_colored_depth)
                
                plt.subplot(233)
                plt.title("Original Disparity")
                plt.imshow(disparity, cmap='jet')
                
                plt.subplot(234)
                plt.title("Cropped RGB Image")
                plt.imshow(cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2RGB))
                
                plt.subplot(235)
                plt.title("Cropped Depth Map")
                colored_cropped_depth = colorize_depth(cropped_depth, max_depth=args.max_depth)
                plt.imshow(colored_cropped_depth)
                
                # 在原始深度图上标记裁剪区域
                plt.subplot(236)
                plt.title("Crop Region Visualization")
                plt.imshow(orig_colored_depth)
                plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                                 edgecolor='red', linewidth=2, fill=False))
                plt.text(x_min, y_min-5, f"Crop: [{x_min}:{x_max}, {y_min}:{y_max}]", 
                         color='red', fontsize=10)
                
                plt.tight_layout()
                
                # 如果需要保存临时文件，则保存可视化结果图
                if args.save_temp:
                    result_path = os.path.join(output_dir, f"{base_name}_result_with_crop.png")
                    plt.savefig(result_path)
                    print(f"Visualization saved to: {result_path}")
                
                plt.show()
                
        except Exception as e:
            print(f"Error processing image pair: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存RGB-深度对的索引文件
    index_path = os.path.join(rgb_depth_dir, 'pairs_index.txt')
    with open(index_path, 'w') as f:
        for rgb_file, depth_file in rgb_depth_pairs:
            f.write(f"{rgb_file} {depth_file}\n")
    
    print(f"\nProcessing summary:")
    print(f"- Cropped RGB-Depth pairs saved to: {rgb_depth_dir}")
    print(f"- Total pairs generated: {len(rgb_depth_pairs)}")
    print(f"- Index file created at: {index_path}")
    
    if args.save_temp:
        print(f"- Temporary visualization results saved to: {output_dir}/")
    
    print("\nDone!")

if __name__ == "__main__":
    main()