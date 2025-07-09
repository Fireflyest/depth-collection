import cv2
from cv2.typing import NumPyArrayNumeric
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# 相机内参矩阵
left_camera_matrix = np.array(
    [
        [908.419604235862, 0, 630.894048815585],
        [0, 630.894048815585, 360.381482752897],
        [0.0, 0.0, 1],
    ]
)
right_camera_matrix = np.array(
    [
        [905.028289028687, 0, 658.695849539017],
        [0, 902.220039325956, 348.026331538046],
        [0.0, 0.0, 1],
    ]
)

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[0.0264730956596808, -0.0442497352397709, 0, 0, 0]])
right_distortion = np.array([[0.0261954683173541, -0.0485380273742757, 0, 0, 0]])

# 旋转矩阵
R = np.array(
    [
        [0.999962437488527, 0.00016774752080323, 0.00866576440786256],
        [-0.000161457995185781, 0.999999723075721, -0.00072648454718833],
        [-0.00866588387408369, 0.000725058101656146, 0.999962187658829],
    ]
)
# 平移矩阵
T = np.array([-53.6537323080632, 0.133445313258266, 0.204463575353323])

class StereoDepthEstimator:
    """立体视觉深度估计器"""
    
    def __init__(self):
        """
        初始化立体视觉深度估计器
        """
        # 预计算立体校正参数
        self.stereo_params = None
        self._prepare_stereo_rectification()
        
        # 创建立体匹配器
        self._create_stereo_matcher()
        
        print(f"立体深度估计器初始化完成")
    
    def _prepare_stereo_rectification(self):
        """预计算立体校正参数"""
        # 使用标准图像尺寸进行校正参数计算
        image_size = (1280, 720)  # 默认尺寸，会在处理时根据实际图像更新
        
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            left_camera_matrix,
            left_distortion,
            right_camera_matrix,
            right_distortion,
            image_size,
            R,
            T
        )
        
        self.stereo_params = {
            'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
            'validPixROI1': validPixROI1, 'validPixROI2': validPixROI2,
            'image_size': image_size
        }
    
    def _create_stereo_matcher(self):
        """创建立体匹配器"""
        # 使用CPU版本的StereoSGBM
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5 * 5,
            P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def update_stereo_rectification(self, image_size: Tuple[int, int]):
        """根据实际图像尺寸更新立体校正参数"""
        if self.stereo_params is None or self.stereo_params['image_size'] != image_size:
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                left_camera_matrix,
                left_distortion,
                right_camera_matrix,
                right_distortion,
                image_size,
                R,
                T
            )
            
            self.stereo_params = {
                'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
                'validPixROI1': validPixROI1, 'validPixROI2': validPixROI2,
                'image_size': image_size
            }
    
    def compute_depth_map(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算深度图
        
        Args:
            left_image: 左图像
            right_image: 右图像
            
        Returns:
            depth_map: 深度图 (以毫米为单位)
            disparity_map: 视差图
        """
        # 确保图像是灰度图
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image.copy()
            
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image.copy()
        
        # 更新立体校正参数
        image_size = (left_gray.shape[1], left_gray.shape[0])
        self.update_stereo_rectification(image_size)
        
        # CPU计算视差图
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # 转换为浮点数并归一化
        disparity = disparity.astype(np.float32) / 16.0
        
        # 过滤视差图中的异常点和无效值
        disparity = self.filter_disparity(disparity)
        
        # CPU重投影计算三维坐标
        points_3d = cv2.reprojectImageTo3D(disparity, self.stereo_params['Q'])
        
        # 提取深度信息 (Z坐标)
        depth_map = points_3d[:, :, 2]
        
        # 深度有效性完全依赖于视差有效性
        # 只进行基本的物理合理性检查（深度为正且不超过合理范围）
        valid_disparity_mask = disparity > 0
        valid_depth_mask = (depth_map > 0) & (depth_map < 10000)  # 10米以内
        
        # 最终有效mask：视差有效且深度物理合理
        final_valid_mask = valid_disparity_mask & valid_depth_mask
        
        # 将无效区域设为0，确保深度和视差的完全一致性
        depth_map[~final_valid_mask] = 0
        disparity[~final_valid_mask] = 0
        
        print(f"深度计算: 视差有效={valid_disparity_mask.sum()}, 深度合理={valid_depth_mask.sum()}, 最终有效={final_valid_mask.sum()}")
        
        return depth_map, disparity
    
    def compute_object_distance(self, depth_map: np.ndarray, mask: np.ndarray) -> float:
        """
        计算物体的距离
        
        Args:
            depth_map: 深度图
            mask: 物体mask
            
        Returns:
            distance: 平均距离 (毫米)
        """
        valid_depths = depth_map[mask & (depth_map > 0)]
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))  # 使用中位数更稳定
        return 0
    
    def compute_object_size(self, points_3d: np.ndarray, mask: np.ndarray) -> float:
        """
        计算物体的三维尺寸
        
        Args:
            points_3d: 三维点云
            mask: 物体mask
            
        Returns:
            max_distance: 物体最大尺寸 (毫米)
        """
        valid_points = []
        
        # 提取有效的三维点
        mask_coords = np.where(mask)
        for y, x in zip(mask_coords[0], mask_coords[1]):
            point = points_3d[y, x]
            if not (math.isinf(point[0]) or math.isinf(point[1]) or math.isinf(point[2])):
                if point[2] > 0 and point[2] < 10000:  # 有效深度范围
                    valid_points.append(point)
        
        if len(valid_points) < 2:
            return 0
        
        valid_points = np.array(valid_points)
        
        # 计算所有点对之间的最大距离
        max_distance = 0
        for i in range(len(valid_points)):
            for j in range(i + 1, min(i + 100, len(valid_points))):  # 限制计算量
                distance = np.linalg.norm(valid_points[i] - valid_points[j])
                if 10 < distance < 1500:  # 合理的尺寸范围
                    max_distance = max(max_distance, distance)
        
        return float(max_distance)

    def filter_disparity(self, disparity: np.ndarray, min_disparity: float = 0.1, max_disparity: float = 200.0) -> np.ndarray:
        """
        简化的视差图异常值过滤 - 只保留两种核心过滤方法
        
        过滤策略：
        1. 上下限阈值过滤：
           - 基本无效值过滤（负值、零值、极大值）
           - 自定义范围过滤（min_disparity ~ max_disparity）
           
        2. 从峰值向两边扩展的密度过滤：
           - 找到视差分布的峰值
           - 从峰值向两边扩展，保留主要分布区间
           - 过滤掉分布边缘的异常值
        
        Args:
            disparity: 原始视差图
            min_disparity: 最小有效视差值（对应最远距离）
            max_disparity: 最大有效视差值（对应最近距离）
            
        Returns:
            filtered_disparity: 过滤后的视差图
        """
        # 创建过滤后的视差图副本
        filtered_disparity = disparity.copy()
        
        # ===== 1. 上下限阈值过滤 =====
        
        # 1.1 基本无效值过滤
        # OpenCV的StereoSGBM输出的无效视差通常是负值或极大值
        invalid_mask = (disparity <= 0) | (disparity >= 1000)
        filtered_disparity[invalid_mask] = 0
        
        # 1.2 用户定义范围过滤
        # 过滤超出用户定义范围的视差值
        range_mask = (filtered_disparity > 0) & ((filtered_disparity < min_disparity) | (filtered_disparity > max_disparity))
        filtered_disparity[range_mask] = 0
        
        print(f"上下限阈值过滤: 无效值={invalid_mask.sum()}, 超出范围={range_mask.sum()}")
        
        # 3. 中值滤波去除椒盐噪声
        # 创建二值mask用于保护边界
        valid_mask = filtered_disparity > 0
        if np.any(valid_mask):
            kernel_size = 5
            # 只对有效区域进行中值滤波
            filtered_valid = filtered_disparity[valid_mask]
            filtered_disparity_uint8 = (filtered_disparity * 255 / filtered_disparity.max()).astype(np.uint8)
            filtered_disparity_uint8 = cv2.medianBlur(filtered_disparity_uint8, kernel_size)
            filtered_disparity = filtered_disparity_uint8.astype(np.float32) * filtered_disparity.max() / 255
            filtered_disparity[~valid_mask] = 0
        
        # 4. 形态学操作去除小的噪声区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 创建有效区域mask
        valid_mask = (filtered_disparity > 0).astype(np.uint8)
        
        if np.any(valid_mask):
            # 形态学开运算：先腐蚀后膨胀，去除小的噪声点
            valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
            
            # 形态学闭运算：先膨胀后腐蚀，填充小的空洞
            valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
            
            # 应用形态学处理后的mask
            filtered_disparity[valid_mask == 0] = 0
        
        # 5. 改进的统计异常值过滤 - 针对长尾分布
        valid_disparities = filtered_disparity[filtered_disparity > 0]
        if len(valid_disparities) > 10:  # 需要足够的样本
            # 方法1：使用百分位数过滤（更适合长尾分布）
            p5 = np.percentile(valid_disparities, 5)   # 第5百分位
            p90 = np.percentile(valid_disparities, 90) # 第90百分位
            p95 = np.percentile(valid_disparities, 95) # 第95百分位
            p99 = np.percentile(valid_disparities, 99) # 第99百分位
            
            # 方法2：使用四分位距(IQR)检测异常值
            q1 = np.percentile(valid_disparities, 25)
            q3 = np.percentile(valid_disparities, 75)
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            
            # 方法3：基于密度的异常检测
            # 计算直方图找到主要分布区间
            hist, bin_edges = np.histogram(valid_disparities, bins=50)
            # 找到直方图中的峰值
            peak_bin = np.argmax(hist)
            peak_value = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
            
            # 从峰值向两边扩展，找到分布的主要区间
            # 向左扩展：找到累积概率达到2%的位置
            cumsum_left = np.cumsum(hist[:peak_bin+1][::-1])[::-1]
            total_count = np.sum(hist)
            left_threshold_idx = peak_bin
            for i in range(peak_bin, -1, -1):
                if cumsum_left[peak_bin - i] / total_count > 0.02:  # 2%阈值
                    left_threshold_idx = i
                    break
            
            # 向右扩展：找到累积概率达到98%的位置
            cumsum_right = np.cumsum(hist[peak_bin:])
            right_threshold_idx = peak_bin
            for i in range(peak_bin, len(hist)):
                if cumsum_right[i - peak_bin] / total_count > 0.98:  # 98%阈值
                    right_threshold_idx = i
                    break
            
            density_lower = bin_edges[left_threshold_idx]
            density_upper = bin_edges[right_threshold_idx + 1]
            
            # 组合多种方法的结果
            # 使用更保守的下限和更严格的上限
            final_lower = max(min_disparity, iqr_lower, p5)
            final_upper = min(iqr_upper, p95, density_upper)
            
            # 如果数据分布很集中，使用更严格的上限
            median_disp = np.median(valid_disparities)
            if p99 > 2 * median_disp:  # 检测长尾
                # 对于长尾分布，使用更严格的上限
                final_upper = min(final_upper, p90 if 'p90' not in locals() else np.percentile(valid_disparities, 90))
                print(f"检测到长尾分布，使用更严格的上限过滤")
            
            # 应用过滤
            outlier_mask = (filtered_disparity < final_lower) | (filtered_disparity > final_upper)
            outlier_count = np.sum(outlier_mask & (filtered_disparity > 0))
            filtered_disparity[outlier_mask] = 0
            
            print(f"改进统计过滤:")
            print(f"  百分位数范围: P5={p5:.1f}, P95={p95:.1f}, P99={p99:.1f}")
            print(f"  IQR范围: [{iqr_lower:.1f}, {iqr_upper:.1f}]")
            print(f"  密度范围: [{density_lower:.1f}, {density_upper:.1f}]")
            print(f"  最终范围: [{final_lower:.1f}, {final_upper:.1f}]")
            print(f"  移除异常值: {outlier_count}")
            
            # 额外的长尾过滤：如果仍有很多远距离异常值
            remaining_valid = filtered_disparity[filtered_disparity > 0]
            if len(remaining_valid) > 0:
                p98 = np.percentile(remaining_valid, 98)
                if p98 > 1.5 * median_disp:
                    # 对超过中位数1.5倍的值进行额外过滤
                    extreme_outlier_mask = filtered_disparity > 1.5 * median_disp
                    extreme_outlier_count = np.sum(extreme_outlier_mask & (filtered_disparity > 0))
                    if extreme_outlier_count < len(remaining_valid) * 0.05:  # 少于5%才过滤
                        filtered_disparity[extreme_outlier_mask] = 0
                        print(f"  额外长尾过滤: 移除>{1.5 * median_disp:.1f}的{extreme_outlier_count}个值")
        
        # 6. 连通域分析，去除小的孤立区域
        valid_disparities = filtered_disparity[filtered_disparity > 0]
        if len(valid_disparities) > 0:
            # 创建二值mask
            binary_mask = (filtered_disparity > 0).astype(np.uint8)
            
            # 连通域分析
            num_labels, labels = cv2.connectedComponents(binary_mask)
            
            # 计算每个连通域的大小
            min_component_size = 50  # 最小连通域大小（像素数）
            removed_components = 0
            
            for label in range(1, num_labels):  # 跳过背景(label=0)
                component_mask = labels == label
                component_size = np.sum(component_mask)
                
                # 去除过小的连通域
                if component_size < min_component_size:
                    filtered_disparity[component_mask] = 0
                    removed_components += 1
            
            print(f"连通域过滤: 总区域={num_labels-1}, 移除小区域={removed_components}")
        
        # 统计最终结果
        final_valid_pixels = np.sum(filtered_disparity > 0)
        total_pixels = filtered_disparity.shape[0] * filtered_disparity.shape[1]
        valid_ratio = final_valid_pixels / total_pixels * 100
                
        print(f"最终视差统计: 有效像素={final_valid_pixels}/{total_pixels} ({valid_ratio:.1f}%)")
        
        return filtered_disparity
    
    def create_disparity_visualization(self, disparity: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        创建视差图的可视化，无效值显示为黑色
        
        Args:
            disparity: 视差图
            colormap: 颜色映射
            
        Returns:
            colored_disparity: 彩色视差图
        """
        # 创建mask分离有效和无效区域
        valid_mask = disparity > 0
        
        if not np.any(valid_mask):
            # 如果没有有效视差，返回全黑图像
            return np.zeros((disparity.shape[0], disparity.shape[1], 3), dtype=np.uint8)
        
        # 只对有效区域进行归一化
        valid_disparities = disparity[valid_mask]
        min_disp = np.min(valid_disparities)
        max_disp = np.max(valid_disparities)
        
        # 创建归一化的视差图
        normalized_disparity = np.zeros_like(disparity, dtype=np.uint8)
        
        if max_disp > min_disp:
            # 将有效视差归一化到0-255范围
            normalized_disparity[valid_mask] = ((disparity[valid_mask] - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)
        else:
            # 如果所有视差值相同，设为中等亮度
            normalized_disparity[valid_mask] = 128
        
        # 应用颜色映射
        colored_disparity = cv2.applyColorMap(normalized_disparity, colormap)
        
        # 将无效区域设为黑色
        colored_disparity[~valid_mask] = [0, 0, 0]
        
        return colored_disparity
    
    def create_depth_visualization(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        创建深度图的可视化，无效值显示为黑色
        
        Args:
            depth_map: 深度图
            colormap: 颜色映射
            
        Returns:
            colored_depth: 彩色深度图
        """
        # 创建mask分离有效和无效区域
        valid_mask = depth_map > 0
        
        if not np.any(valid_mask):
            # 如果没有有效深度，返回全黑图像
            return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        
        # 只对有效区域进行归一化
        valid_depths = depth_map[valid_mask]
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        
        # 创建归一化的深度图
        normalized_depth = np.zeros_like(depth_map, dtype=np.uint8)
        
        if max_depth > min_depth:
            # 将有效深度归一化到0-255范围
            normalized_depth[valid_mask] = ((depth_map[valid_mask] - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        else:
            # 如果所有深度值相同，设为中等亮度
            normalized_depth[valid_mask] = 128
        
        # 应用颜色映射
        colored_depth = cv2.applyColorMap(normalized_depth, colormap)
        
        # 将无效区域设为黑色
        colored_depth[~valid_mask] = [0, 0, 0]
        
        return colored_depth
    
    def create_depth_histogram(self, depth_map: np.ndarray, size: Tuple[int, int] = (320, 240)) -> np.ndarray:
        """
        创建深度分布直方图
        
        Args:
            depth_map: 深度图
            size: 输出图像尺寸
            
        Returns:
            histogram_image: 直方图图像
        """
        # 创建空白图像
        hist_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # 获取有效深度值
        valid_depths = depth_map[depth_map > 0]
        
        if len(valid_depths) == 0:
            # 没有有效深度时显示"No Valid Depth"
            cv2.putText(hist_img, "No Valid Depth", (10, size[1]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return hist_img
        
        # 计算直方图
        min_depth = valid_depths.min()
        max_depth = valid_depths.max()
        
        # 如果深度范围太小，扩展一下
        if max_depth - min_depth < 100:  # 100mm
            center = (min_depth + max_depth) / 2
            min_depth = center - 50
            max_depth = center + 50
        
        # 计算直方图（30个bin）
        hist, bin_edges = np.histogram(valid_depths, bins=30, range=(min_depth, max_depth))
        
        # 归一化直方图到图像高度
        hist_normalized = hist / hist.max() * (size[1] - 60)  # 留60像素给标签
        
        # 绘制直方图
        bin_width = size[0] / len(hist)
        
        for i, h in enumerate(hist_normalized):
            x1 = int(i * bin_width)
            x2 = int((i + 1) * bin_width)
            y1 = size[1] - 40  # 底部留40像素
            y2 = int(y1 - h)
            
            # 根据深度值设置颜色（近的偏红，远的偏蓝）
            depth_ratio = i / len(hist)
            color = (int(255 * (1 - depth_ratio)), int(128), int(255 * depth_ratio))
            
            cv2.rectangle(hist_img, (x1, y2), (x2, y1), color, -1)
            cv2.rectangle(hist_img, (x1, y2), (x2, y1), (255, 255, 255), 1)
        
        # 添加标题
        cv2.putText(hist_img, "Depth Distribution", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加深度范围标签
        cv2.putText(hist_img, f"Min: {min_depth:.0f}mm", (10, size[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(hist_img, f"Max: {max_depth:.0f}mm", (size[0] - 100, size[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加像素统计
        total_pixels = depth_map.shape[0] * depth_map.shape[1]
        valid_ratio = len(valid_depths) / total_pixels * 100
        cv2.putText(hist_img, f"Valid: {valid_ratio:.1f}%", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return hist_img

def process_stereo_video(video_path: str, output_dir: str = "stereo_output", max_frames: int = 100, show_display: bool = False):
    """
    处理立体视频，提取深度信息
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        max_frames: 最大处理帧数
        show_display: 是否显示实时窗口
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化立体深度估计器
    estimator = StereoDepthEstimator()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    frame_count = 0
    
    print(f"开始处理立体视频: {video_path}")
    print(f"将处理最多 {max_frames} 帧")
    
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            print(f"读取帧 {frame_count} 失败，视频结束")
            break
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 分割左右图像
        frame_left = frame[0:frame_height, 0:(frame_width // 2)]
        frame_right = frame[0:frame_height, (frame_width // 2):frame_width]
        
        # 计算深度图
        depth_map, disparity = estimator.compute_depth_map(frame_left, frame_right)
        
        # 每帧都保存拼接结果
        if frame_count % 5 == 0:  # 每5帧保存一次
            # 创建拼接图像
            # 调整所有图像到相同尺寸
            target_width, target_height = 640, 480
            
            # 调整原始图像
            left_resized = cv2.resize(frame_left, (target_width, target_height))
            right_resized = cv2.resize(frame_right, (target_width, target_height))
            
            # 创建深度图可视化 - 使用新的可视化方法
            depth_colored = estimator.create_depth_visualization(depth_map, cv2.COLORMAP_JET)
            depth_resized = cv2.resize(depth_colored, (target_width, target_height))
            
            # 创建视差图可视化 - 使用新的可视化方法
            disparity_colored = estimator.create_disparity_visualization(disparity, cv2.COLORMAP_PLASMA)
            disparity_resized = cv2.resize(disparity_colored, (target_width, target_height))
            
            # 创建深度分布直方图
            depth_histogram = estimator.create_depth_histogram(depth_map, (target_width, target_height))
            
            # 添加标题文本
            cv2.putText(left_resized, "Left Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(right_resized, "Right Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(depth_resized, "Depth Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(disparity_resized, "Disparity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 添加深度范围信息
            valid_depths = depth_map[depth_map > 0]
            valid_disparities = disparity[disparity > 0]
            
            if len(valid_depths) > 0:
                depth_range_text = f"Depth: {valid_depths.min():.0f}-{valid_depths.max():.0f}mm"
                cv2.putText(depth_resized, depth_range_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(valid_disparities) > 0:
                disp_range_text = f"Disp: {valid_disparities.min():.1f}-{valid_disparities.max():.1f}"
                cv2.putText(disparity_resized, disp_range_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                valid_ratio = len(valid_disparities) / (disparity.shape[0] * disparity.shape[1]) * 100
                valid_text = f"Valid: {valid_ratio:.1f}%"
                cv2.putText(disparity_resized, valid_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 创建统计信息图
            stats_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            cv2.putText(stats_img, "Statistics", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            y_offset = 70
            if len(valid_depths) > 0:
                cv2.putText(stats_img, f"Total Pixels: {depth_map.shape[0] * depth_map.shape[1]}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 30
                cv2.putText(stats_img, f"Valid Pixels: {len(valid_depths)}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 30
                cv2.putText(stats_img, f"Valid Ratio: {len(valid_depths)/(depth_map.shape[0] * depth_map.shape[1])*100:.1f}%", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 40
                
                cv2.putText(stats_img, f"Min Depth: {valid_depths.min():.1f}mm", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 30
                cv2.putText(stats_img, f"Max Depth: {valid_depths.max():.1f}mm", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                y_offset += 30
                cv2.putText(stats_img, f"Mean Depth: {valid_depths.mean():.1f}mm", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                y_offset += 30
                cv2.putText(stats_img, f"Median Depth: {np.median(valid_depths):.1f}mm", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                y_offset += 40
                
                if len(valid_disparities) > 0:
                    cv2.putText(stats_img, f"Min Disparity: {valid_disparities.min():.1f}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 128), 1)
                    y_offset += 30
                    cv2.putText(stats_img, f"Max Disparity: {valid_disparities.max():.1f}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 255), 1)
            
            # 2x3拼接布局
            top_row = np.hstack([left_resized, right_resized, depth_resized])
            bottom_row = np.hstack([disparity_resized, depth_histogram, stats_img])
            combined = np.vstack([top_row, bottom_row])
            
            # 保存拼接结果
            output_path = f"{output_dir}/stereo_result_frame_{frame_count:04d}.jpg"
            cv2.imwrite(output_path, combined)
            
            print(f"已处理帧 {frame_count}, 深度范围: {valid_depths.min():.1f} - {valid_depths.max():.1f} mm")
            print(f"保存结果: {output_path}")
            
            # 可选的实时显示
            if show_display:
                try:
                    display_combined = cv2.resize(combined, (1536, 640))  # 调整显示尺寸适应2x3布局
                    cv2.imshow("Stereo Processing", display_combined)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"显示窗口错误: {e}")
                    show_display = False  # 禁用显示
        
        frame_count += 1
        
        # 每10帧显示进度
        if frame_count % 10 == 0:
            print(f"进度: {frame_count}/{max_frames} 帧")
    
    cap.release()
    if show_display:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print(f"处理完成，共处理 {frame_count} 帧")
    print(f"结果保存在: {output_dir}")

def demo_stereo_depth():
    """演示立体深度计算"""
    video_path = "assets/fish/stereo_fish.avi"
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    # 处理视频 - 禁用显示以避免Qt问题
    process_stereo_video(video_path, "visualizations/stereo_output", max_frames=50, show_display=False)

if __name__ == "__main__":
    demo_stereo_depth()
