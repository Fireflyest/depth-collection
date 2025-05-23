#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FLSea Dataset Visualization Script

Loads and visualizes samples from the FLSea underwater dataset with both original
and processed images along with their corresponding depth maps.

Usage:
    python flsea_dataset.py --data-root /path/to/FLSea --num-samples 10
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import rasterio
import random
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# Suppress rasterio NotGeoreferencedWarning
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

def load_image(path):
    """Load an image and convert to RGB if necessary."""
    with rasterio.open(path) as src:
        img = src.read()
        if img.shape[0] >= 3:
            img = img[:3].transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0).transpose(1, 2, 0)
    return img

def load_depth(path):
    """Load a depth map from tif file."""
    with rasterio.open(path) as src:
        depth = src.read(1)  # Read first band
        return depth

class FLSeaDataset(Dataset):
    """FLSea underwater dataset with both original and processed images."""
    
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root (str): Path to FLSea dataset
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_root = data_root
        self.transform = transform
        
        # Define dataset paths
        self.processed_img_dir = os.path.join(data_root, 'seaErra')
        self.orig_img_dir = os.path.join(data_root, 'imgs')
        self.depth_dir = os.path.join(data_root, 'depth')
        
        # Use original images as the primary index
        self.orig_img_files = glob.glob(os.path.join(self.orig_img_dir, '*.tiff'))
        self.orig_img_files = sorted(self.orig_img_files)
        
        # Find matching processed images and depth maps
        self.valid_samples = []
        for orig_img_path in self.orig_img_files:
            basename = os.path.splitext(os.path.basename(orig_img_path))[0]
            
            # Look for processed image (format: <basename>_SeaErra.tiff)
            processed_img_path = os.path.join(self.processed_img_dir, f"{basename}_SeaErra.tiff")
            if not os.path.exists(processed_img_path):
                processed_img_path = orig_img_path  # Fall back to original if processed doesn't exist
            
            # Look for depth map (format: <basename>_SeaErra_abs_depth.tif)
            depth_path = os.path.join(self.depth_dir, f"{basename}_SeaErra_abs_depth.tif")
            if not os.path.exists(depth_path):
                continue  # Skip samples without depth maps
                
            # Add valid sample
            self.valid_samples.append((processed_img_path, orig_img_path, depth_path))
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        processed_img_path, orig_img_path, depth_path = self.valid_samples[idx]
        
        # Load images and depth
        processed_img = load_image(processed_img_path)
        orig_img = load_image(orig_img_path)
        depth = load_depth(depth_path)
        
        return {
            'processed_image': processed_img,
            'original_image': orig_img,
            'depth': depth,
            'processed_image_path': processed_img_path
        }

def visualize_sample(original_image, processed_image, depth, file_prefix, output_dir):
    """Visualize original image, processed image, and depth map in a clean layout."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy if needed
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.permute(1, 2, 0).cpu().numpy()
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
    
    if isinstance(processed_image, torch.Tensor):
        processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
        if processed_image.max() <= 1.0:
            processed_image = (processed_image * 255).astype(np.uint8)
    
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    
    # Handle the case where depth is None
    if depth is None:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title('Processed Image (seaErra)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_images_only.png"), dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create masks for valid pixels
    depth_valid = ~np.isnan(depth) & (depth > 0) & np.isfinite(depth)
    
    # Normalize valid depth for visualization
    depth_viz = np.zeros_like(depth, dtype=np.float32)
    if np.any(depth_valid):
        depth_valid_values = depth[depth_valid]
        min_depth, max_depth = np.percentile(depth_valid_values, [5, 95])
        print(f"Dataset visualization depth range: {min_depth:.2f} - {max_depth:.2f}")
        depth_viz[depth_valid] = np.clip((depth[depth_valid] - min_depth) / (max_depth - min_depth + 1e-8), 0, 1)
    
    # Apply colormap for depth visualization
    depth_colored = plt.cm.jet(1.0 - depth_viz)  # 1.0 - for near=red, far=blue
    
    # 将无效值设置为黑色 (RGB: 0,0,0)
    depth_rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)  # 先创建全黑图像
    depth_rgb[depth_valid] = (depth_colored[depth_valid, :3] * 255).astype(np.uint8)  # 只为有效区域上色
    
    # Create visualization
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image (seaErra)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(depth_rgb)
    plt.title('Depth Map (Near=Red, Far=Blue, Invalid=Black)')
    plt.axis('off')
    
    # Add histogram of valid depth values
    plt.subplot(2, 2, 4)
    if np.any(depth_valid):
        plt.hist(depth[depth_valid].flatten(), bins=50)
        plt.title(f'Depth Histogram (Range: {min_depth:.2f} - {max_depth:.2f} m)')
        plt.xlabel('Depth (m)')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No valid depth values', ha='center', va='center')
        plt.title('Depth Histogram')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_sample.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save depth statistics to file
    if np.any(depth_valid):
        stats = {
            'min_depth': float(depth_valid_values.min()),
            'max_depth': float(depth_valid_values.max()),
            'mean_depth': float(depth_valid_values.mean()),
            'median_depth': float(np.median(depth_valid_values)),
            'std_depth': float(depth_valid_values.std()),
            'valid_pixels_pct': float(depth_valid.sum() / depth_valid.size * 100),
        }
        
        with open(os.path.join(output_dir, f"{file_prefix}_stats.txt"), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description='FLSea Dataset Visualization')
    
    parser.add_argument('--data-root', type=str, default='assets/FLSea/red_sea/pier_path',
                        help='Path to FLSea dataset')
    parser.add_argument('--output-dir', type=str, default='visualizations/flsea_dataset',
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--random-samples', action='store_true',
                        help='Select random samples instead of the first N')
    
    args = parser.parse_args()
    
    # Create FLSea dataset
    dataset = FLSeaDataset(args.data_root)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"FLSea dataset loaded with {len(dataset)} valid samples")
    
    # Get indices of samples to visualize
    if args.random_samples:
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    else:
        indices = list(range(min(args.num_samples, len(dataset))))
    
    # Visualize selected samples
    print(f"Visualizing {len(indices)} samples...")
    depth_values = []
    valid_pixel_ratios = []
    
    for idx in tqdm(indices):
        sample = dataset[idx]
        
        # Extract data
        original_image = sample['original_image']
        processed_image = sample['processed_image'] 
        depth = sample['depth']
        processed_img_path = sample['processed_image_path']
        
        # Extract basename for output file
        basename = os.path.splitext(os.path.basename(processed_img_path))[0]
        
        # Visualize sample (convert tensors to uint8 for visualization)
        visualize_sample(
            original_image,
            processed_image,
            depth,
            f"{idx:03d}_{basename}",
            args.output_dir
        )
        
        # Collect statistics
        if depth is not None:
            depth_np = depth if isinstance(depth, np.ndarray) else depth.numpy()
            valid_mask = ~np.isnan(depth_np) & (depth_np > 0) & np.isfinite(depth_np)
            if np.any(valid_mask):
                depth_values.extend(depth_np[valid_mask].flatten().tolist())
                valid_pixel_ratios.append(valid_mask.sum() / valid_mask.size)
    
    # Generate overall dataset statistics
    if depth_values:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.hist(depth_values, bins=50)
        plt.title('Overall Depth Distribution')
        plt.xlabel('Depth (m)')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 1, 2)
        plt.hist(valid_pixel_ratios, bins=20)
        plt.title('Valid Depth Pixel Ratio Distribution')
        plt.xlabel('Ratio of Valid Pixels')
        plt.ylabel('Number of Images')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "dataset_statistics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        with open(os.path.join(args.output_dir, "dataset_summary.txt"), 'w') as f:
            f.write(f"Total samples: {len(dataset)}\n")
            f.write(f"Visualized samples: {len(indices)}\n\n")
            
            f.write("Depth Statistics:\n")
            f.write(f"  Min depth: {min(depth_values):.4f} m\n")
            f.write(f"  Max depth: {max(depth_values):.4f} m\n")
            f.write(f"  Mean depth: {np.mean(depth_values):.4f} m\n")
            f.write(f"  Median depth: {np.median(depth_values):.4f} m\n")
            f.write(f"  Std depth: {np.std(depth_values):.4f} m\n\n")
            
            f.write("Valid Pixel Ratio Statistics:\n")
            f.write(f"  Min ratio: {min(valid_pixel_ratios):.4f}\n")
            f.write(f"  Max ratio: {max(valid_pixel_ratios):.4f}\n")
            f.write(f"  Mean ratio: {np.mean(valid_pixel_ratios):.4f}\n")
            f.write(f"  Median ratio: {np.median(valid_pixel_ratios):.4f}\n")
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()