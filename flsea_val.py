#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified FLSea Dataset Validation Script
Evaluates model performance on both original and processed underwater images from the FLSea dataset
Supports multiple depth estimation models: DepthAnything v2 and ZoeDepth

Usage:
    # For DepthAnything v2
    python flsea_val.py --model-type depthanything --encoder vitl --data-root assets/FLSea/red_sea/pier_path --num-samples 10
    
    # For ZoeDepth
    python flsea_val.py --model-type zoedepth --zoedepth-type N --data-root assets/FLSea/red_sea/pier_path --num-samples 10
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random
import warnings
from typing import Optional, Tuple

# Import model wrapper and characteristics
from models import create_model, get_model_name, ModelOutputCharacteristics

# Import FLSea dataset
from flsea_dataset import FLSeaDataset

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def align_depth_scale(pred, gt, method='median'):
    """
    Align the scale of predicted depth to ground truth depth
    
    Args:
        pred: predicted depth map
        gt: ground truth depth map
        method: alignment method ('median', 'mean', 'least_squares')
    
    Returns:
        scale factor and aligned prediction
    """
    eps = 1e-6
    
    if method == 'median':
        scale = np.median(gt) / (np.median(pred) + eps)
    elif method == 'mean':
        scale = np.mean(gt) / (np.mean(pred) + eps)
    elif method == 'least_squares':
        # Least squares scale alignment: minimize ||s*pred - gt||^2
        scale = np.sum(pred * gt) / (np.sum(pred * pred) + eps)
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    return scale, pred * scale

def compute_depth_metrics(pred, gt, mask=None, align_method='median', is_gt_disparity=True, is_pred_disparity=False):
    """
    Compute depth evaluation metrics for relative depth estimation
    
    Args:
        pred: predicted depth/disparity map
        gt: ground truth map (disparity or depth)
        mask: optional mask for valid pixels
        align_method: method for scale alignment ('median', 'mean', 'least_squares')
        is_gt_disparity: whether the ground truth is a disparity map (inverse depth)
        is_pred_disparity: whether the prediction is a disparity map (inverse depth)
                          Note: For non-metric models, this includes:
                          - Absolute disparity (1/depth with known scale)
                          - Relative/normalized disparity (inverse proportional to depth)
                          - Other inverse-depth-like representations
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays if they are tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.squeeze().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy().astype(bool)
    
    # Create valid value masks - exclude zeros, nan, and negative values
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    
    valid_mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred) & mask
    
    # Apply valid mask
    if valid_mask.sum() == 0:
        # Return zeros if no valid pixels
        return {
            'scale_factor': 0,
            'rmse': 0, 'rmse_log': 0, 'abs_rel': 0, 'sq_rel': 0,
            'delta1': 0, 'delta2': 0, 'delta3': 0, 'log10': 0,
            'silog': 0, 'pearson_corr': 0, 'spearman_corr': 0
        }
        
    pred = pred[valid_mask]
    gt = gt[valid_mask].copy()  # Make a copy to avoid modifying the original
    
    # Avoid division by zero
    eps = 1e-6
    
    # Convert to same domain for fair comparison
    # Strategy: Convert everything to depth domain for metrics calculation
    if is_gt_disparity:
        gt = 1.0 / (gt + eps)  # GT: disparity → depth
    
    if is_pred_disparity:
        pred = 1.0 / (pred + eps)  # Pred: disparity → depth
    
    # Now both pred and gt are in depth domain (near=small, far=large)
    
    # Align scale - critical for comparing relative depth with metric depth
    scale_factor, pred_aligned = align_depth_scale(pred, gt, method=align_method)
    
    # Threshold accuracy metrics: δ < 1.25^n (scale-invariant)
    thresh = np.maximum((gt / (pred_aligned + eps)), ((pred_aligned + eps) / (gt + eps)))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()
    
    # Error metrics (after scale alignment)
    rmse = np.sqrt(((gt - pred_aligned) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt + eps) - np.log(pred_aligned + eps)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred_aligned) / (gt + eps))
    sq_rel = np.mean(((gt - pred_aligned) ** 2) / (gt + eps))
    
    # Log accuracy
    log10 = np.mean(np.abs(np.log10(gt + eps) - np.log10(pred_aligned + eps)))
    
    # Scale-invariant log error (SILog) - important for relative depth
    log_diff = np.log(pred_aligned + eps) - np.log(gt + eps)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2)) * 100
    
    # Correlation metrics (should use aligned values for fair comparison)
    from scipy.stats import pearsonr, spearmanr
    try:
        pearson_corr, _ = pearsonr(pred_aligned.flatten(), gt.flatten())
        spearman_corr, _ = spearmanr(pred_aligned.flatten(), gt.flatten())
    except:
        pearson_corr = 0
        spearman_corr = 0
    
    return {
        'scale_factor': scale_factor,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
        'log10': log10,
        'silog': silog,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    }

def visualize_comparison_combined(orig_img, proc_img, gt_depth, orig_pred_depth, proc_pred_depth, 
                                file_prefix, output_dir, model_characteristics: ModelOutputCharacteristics, 
                                conversion_info=None):
    """
    Create a combined visualization comparing original and processed images with their depth predictions
    
    Args:
        orig_img: Original RGB image
        proc_img: Processed (SeaErra) RGB image
        gt_depth: Ground truth depth map
        orig_pred_depth: Predicted depth map from original image
        proc_pred_depth: Predicted depth map from processed image
        file_prefix: Prefix for saved files
        output_dir: Directory to save visualizations
        model_characteristics: ModelOutputCharacteristics describing the model output
        conversion_info: Information about unit conversion applied
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare for visualization - handle invalid values
    gt_viz = np.zeros_like(gt_depth, dtype=np.float32)
    orig_pred_viz = np.zeros_like(orig_pred_depth, dtype=np.float32)
    proc_pred_viz = np.zeros_like(proc_pred_depth, dtype=np.float32)
    
    # Create masks for valid pixels
    gt_valid = ~np.isnan(gt_depth) & ~np.isinf(gt_depth) & (gt_depth > 0)
    orig_pred_valid = ~np.isnan(orig_pred_depth) & ~np.isinf(orig_pred_depth) & (orig_pred_depth > 0)
    proc_pred_valid = ~np.isnan(proc_pred_depth) & ~np.isinf(proc_pred_depth) & (proc_pred_depth > 0)
    
    # Create a combined mask of valid pixels across all depth maps for consistent normalization
    gt_valid_depths = []
    pred_valid_depths = []
    
    if np.any(gt_valid):
        gt_valid_depths.append(gt_depth[gt_valid])
    if np.any(orig_pred_valid):
        pred_valid_depths.append(orig_pred_depth[orig_pred_valid])
    if np.any(proc_pred_valid):
        pred_valid_depths.append(proc_pred_depth[proc_pred_valid])
    
    # 分别为ground truth和预测深度确定范围
    if gt_valid_depths:
        gt_valid_values = np.concatenate(gt_valid_depths)
        gt_min_depth, gt_max_depth = np.percentile(gt_valid_values, [2, 98])
        gt_range_text = f"GT Depth: {gt_min_depth:.4f} - {gt_max_depth:.4f} (meters)"
        
        # 为gt单独归一化
        if np.any(gt_valid):
            gt_viz[gt_valid] = np.clip((gt_depth[gt_valid] - gt_min_depth) / (gt_max_depth - gt_min_depth + 1e-8), 0, 1)
    else:
        gt_range_text = "GT Depth: N/A"
    
    # 为预测深度确定范围
    if pred_valid_depths:
        pred_valid_values = np.concatenate(pred_valid_depths)
        pred_min_depth, pred_max_depth = np.percentile(pred_valid_values, [5, 95])
        
        # Create range text based on model characteristics
        pred_range_text = f"Pred {model_characteristics.display_name}: {pred_min_depth:.4f} - {pred_max_depth:.4f}"
        
        # 归一化预测深度
        if np.any(orig_pred_valid):
            orig_pred_viz[orig_pred_valid] = np.clip((orig_pred_depth[orig_pred_valid] - pred_min_depth) / (pred_max_depth - pred_min_depth + 1e-8), 0, 1)
        if np.any(proc_pred_valid):
            proc_pred_viz[proc_pred_valid] = np.clip((proc_pred_depth[proc_pred_valid] - pred_min_depth) / (pred_max_depth - pred_min_depth + 1e-8), 0, 1)
    else:
        pred_range_text = "Pred: N/A"
    
    # Apply colormap with custom handling for invalid values
    cm_jet = plt.colormaps['jet']  # Modern way to get colormap
    
    # Ground truth (depth map: small values=near, large values=far)
    # For depth: near=small value=red, far=large value=blue
    # Note: jet colormap goes from blue(0) to red(1), so we need to INVERT for depth
    gt_colored = np.zeros((*gt_viz.shape, 4), dtype=np.float32)
    gt_colored[gt_valid] = cm_jet(1.0 - gt_viz[gt_valid])  # Invert mapping for depth (low depth=near=red, high depth=far=blue)
    gt_colored[~gt_valid, 3] = 0  # Set alpha=0 for invalid regions
    gt_rgb = gt_colored[:, :, :3]  # Keep as float32 in range [0, 1] for imshow
    
    # Prediction visualization depends on whether prediction was originally disparity
    # After processing, all predictions are in depth space, so use consistent mapping
    orig_pred_colored = np.zeros((*orig_pred_viz.shape, 4), dtype=np.float32)
    orig_pred_colored[orig_pred_valid] = cm_jet(1.0 - orig_pred_viz[orig_pred_valid])  # Invert mapping for depth
    orig_pred_colored[~orig_pred_valid, 3] = 0
    orig_pred_rgb = orig_pred_colored[:, :, :3]
    
    proc_pred_colored = np.zeros((*proc_pred_viz.shape, 4), dtype=np.float32)
    proc_pred_colored[proc_pred_valid] = cm_jet(1.0 - proc_pred_viz[proc_pred_valid])  # Invert mapping for depth
    proc_pred_colored[~proc_pred_valid, 3] = 0
    proc_pred_rgb = proc_pred_colored[:, :, :3]
    
    pred_title_suffix = "Depth Prediction (Near=Red, Far=Blue)"
    
    # Create comparison visualization with 2 rows, 3 columns
    plt.figure(figsize=(18, 12))
    
    # First row: Input images and ground truth
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(proc_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title('Processed Image (SeaErra)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(gt_rgb)
    plt.title(f'Ground Truth Depth (Near=Red, Far=Blue)\n{gt_range_text}')
    
    # Add depth scale markers for GT (now confirmed as depth values)
    if gt_valid_depths:
        h, w = gt_depth.shape
        x_center = w // 2
        y_positions = [h // 6, h // 2, 5 * h // 6]  # top, middle, bottom
        
        for y_pos in y_positions:
            if gt_valid[y_pos, x_center]:
                depth_val = gt_depth[y_pos, x_center]
                plt.annotate(f'{depth_val:.4f}m', 
                           xy=(x_center, y_pos), 
                           xytext=(x_center + w//8, y_pos),
                           fontsize=8, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='white', lw=1))
    
    plt.axis('off')
    
    # Second row: Depth predictions and error maps
    plt.subplot(2, 3, 4)
    plt.imshow(orig_pred_rgb)
    plt.title(f'Original Image {pred_title_suffix}\n{pred_range_text}')
    
    # Add depth scale markers for predictions
    if pred_valid_depths:
        h, w = orig_pred_depth.shape
        x_center = w // 2
        y_positions = [h // 6, h // 2, 5 * h // 6]
        
        for y_pos in y_positions:
            if orig_pred_valid[y_pos, x_center]:
                pred_val = orig_pred_depth[y_pos, x_center]
                
                # Calculate display labels based on conversion info
                if conversion_info and conversion_info.get('applied', False):
                    # Unit conversion was applied - show original and converted values
                    original_val = model_characteristics.get_original_value_from_converted(pred_val)
                    raw_label = f'{original_val:.4f}({model_characteristics.output_unit[:4]})'
                    meter_label = f'\n→{pred_val:.4f}m'
                elif model_characteristics.is_metric:
                    # Already in target units (typically meters)
                    raw_label = f'{pred_val:.4f}{model_characteristics.output_unit[:1]}'
                    meter_label = ''  # No conversion needed
                else:
                    # Non-metric models - show relative values with approximate conversion
                    raw_label = f'{pred_val:.4f}(rel)'
                    # For non-metric depth, try to estimate actual depth using GT scale
                    if gt_valid_depths:
                        gt_sample = gt_depth[gt_valid]
                        pred_sample = orig_pred_depth[orig_pred_valid & gt_valid]
                        if len(pred_sample) > 0 and len(gt_sample) > 0:
                            # Rough scale estimation using median ratio
                            scale = np.median(gt_sample) / np.median(pred_sample)
                            meter_val = pred_val * scale
                            meter_label = f'\n≈{meter_val:.4f}m'
                        else:
                            meter_label = '\n≈?m'
                    else:
                        meter_label = '\n≈?m'
                
                # Combine raw and meter labels
                full_label = raw_label + meter_label
                
                plt.annotate(full_label, 
                           xy=(x_center, y_pos), 
                           xytext=(x_center + w//10, y_pos),
                           fontsize=8, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='white', lw=1))
    
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(proc_pred_rgb)
    plt.title(f'Processed Image {pred_title_suffix}\n{pred_range_text}')
    
    # Add depth scale markers for processed predictions
    if pred_valid_depths:
        h, w = proc_pred_depth.shape
        x_center = w // 2
        y_positions = [h // 6, h // 2, 5 * h // 6]
        
        for y_pos in y_positions:
            if proc_pred_valid[y_pos, x_center]:
                pred_val = proc_pred_depth[y_pos, x_center]
                
                # Calculate display labels based on conversion info  
                if conversion_info and conversion_info.get('applied', False):
                    # Unit conversion was applied - show original and converted values
                    original_val = model_characteristics.get_original_value_from_converted(pred_val)
                    raw_label = f'{original_val:.4f}({model_characteristics.output_unit[:4]})'
                    meter_label = f'\n→{pred_val:.4f}m'
                elif model_characteristics.is_metric:
                    # Already in target units (typically meters)
                    raw_label = f'{pred_val:.4f}{model_characteristics.output_unit[:1]}'
                    meter_label = ''  # No conversion needed
                else:
                    # Non-metric models - show relative values with approximate conversion
                    raw_label = f'{pred_val:.4f}(rel)'
                    # For non-metric depth, try to estimate actual depth using GT scale
                    if gt_valid_depths:
                        gt_sample = gt_depth[gt_valid]
                        pred_sample = proc_pred_depth[proc_pred_valid & gt_valid]
                        if len(pred_sample) > 0 and len(gt_sample) > 0:
                            # Rough scale estimation using median ratio
                            scale = np.median(gt_sample) / np.median(pred_sample)
                            meter_val = pred_val * scale
                            meter_label = f'\n≈{meter_val:.4f}m'
                        else:
                            meter_label = '\n≈?m'
                    else:
                        meter_label = '\n≈?m'
                
                # Combine raw and meter labels
                full_label = raw_label + meter_label
                
                plt.annotate(full_label, 
                           xy=(x_center, y_pos), 
                           xytext=(x_center + w//10, y_pos),
                           fontsize=8, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='white', lw=1))
    
    plt.axis('off')
    
    # Compute error maps - use same normalization as metrics calculation
    plt.subplot(2, 3, 6)
    
    # Compute error in original depth space (same as metrics calculation)
    orig_error = np.zeros_like(gt_depth)
    proc_error = np.zeros_like(gt_depth)
    valid_orig = gt_valid & orig_pred_valid
    valid_proc = gt_valid & proc_pred_valid
    
    # Convert predictions to depth domain if needed (same as in compute_depth_metrics)
    orig_pred_depth = orig_pred_depth.copy()
    proc_pred_depth = proc_pred_depth.copy()
    gt_for_error = gt_depth.copy()
    
    eps = 1e-6
    
    # Convert to depth domain (same logic as compute_depth_metrics)
    # After postprocessing, all predictions are in depth domain (meters)
    # No additional conversion needed since model characteristics handle this
    
    # Apply scale alignment (same as in compute_depth_metrics)
    if np.any(valid_orig):
        scale_orig, orig_pred_aligned = align_depth_scale(
            orig_pred_depth[valid_orig], gt_for_error[valid_orig], method='median'
        )
        # Compute absolute error in meters (same as metrics)
        orig_error[valid_orig] = np.abs(gt_for_error[valid_orig] - orig_pred_aligned)
    
    if np.any(valid_proc):
        scale_proc, proc_pred_aligned = align_depth_scale(
            proc_pred_depth[valid_proc], gt_for_error[valid_proc], method='median'
        )
        # Compute absolute error in meters (same as metrics)
        proc_error[valid_proc] = np.abs(gt_for_error[valid_proc] - proc_pred_aligned)
    
    # Create a side-by-side error comparison
    h, w = gt_depth.shape
    combined_error = np.zeros((h, w*2))  # Two error maps side by side
    combined_error[:, :w] = orig_error
    combined_error[:, w:] = proc_error
    
    # Normalize for visualization (now represents actual meter-scale errors)
    valid_error = (combined_error > 0)
    if np.any(valid_error):
        # Use percentile-based normalization instead of max to avoid outliers
        error_95th = np.percentile(combined_error[valid_error], 95)
        if error_95th > 0:
            combined_error_viz = np.clip(combined_error / error_95th, 0, 1)
        else:
            combined_error_viz = combined_error
        
        # Calculate error statistics for display
        orig_errors = orig_error[orig_error > 0]
        proc_errors = proc_error[proc_error > 0]
        
        if len(orig_errors) > 0 and len(proc_errors) > 0:
            avg_orig_error = np.mean(orig_errors)
            avg_proc_error = np.mean(proc_errors)
            improvement = (avg_orig_error - avg_proc_error) / avg_orig_error * 100
            
            error_title = f'Absolute Error Maps (meters)\nOriginal: {avg_orig_error:.4f}m, Processed: {avg_proc_error:.4f}m\nImprovement: {improvement:.1f}%'
        else:
            error_title = 'Absolute Error Maps (meters)\nOriginal (left) vs Processed (right)'
    else:
        combined_error_viz = combined_error
        error_title = 'Absolute Error Maps (meters)\nOriginal (left) vs Processed (right)'
    
    # Apply colormap to combined error
    cmap = plt.colormaps['hot']  # Use 'hot' colormap for errors (black=no error, red/yellow=high error)
    error_rgb = cmap(combined_error_viz)
    error_rgb = error_rgb[:, :, :3]  # Keep as float32 in range [0, 1] for imshow
    
    plt.imshow(error_rgb)
    plt.title(error_title)
    plt.axis('off')
    
    # Add a vertical line to separate the two error maps
    plt.axvline(x=w-0.5, color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()

def validate_flsea(args):
    """
    Validate depth estimation models on FLSea dataset
    
    Args:
        args: Command line arguments
    """
    print("Initializing model...")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model wrapper using factory function
    if args.model_type == 'depthanything':
        model = create_model(
            model_type='depthanything',
            device=device,
            encoder=args.encoder,
            checkpoint_dir=args.checkpoint_dir,
            metric=args.metric
        )
        model_name = get_model_name('depthanything', encoder=args.encoder, metric=args.metric)
    elif args.model_type == 'zoedepth':
        model = create_model(
            model_type='zoedepth',
            device=device,
            zoedepth_type=args.zoedepth_type,
            checkpoint_dir=args.checkpoint_dir
        )
        model_name = get_model_name('zoedepth', zoedepth_type=args.zoedepth_type)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    print(f"Model: {model_name}")
    
    # Create FLSea dataset
    print(f"Loading FLSea dataset from {args.data_root}")
    dataset = FLSeaDataset(args.data_root)
    
    if len(dataset) == 0:
        print("No valid samples found in the dataset")
        return
    
    print(f"Found {len(dataset)} valid samples in the dataset")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize metrics dictionaries for both original and processed images
    metrics_orig = {
        'scale_factor': 0, 'rmse': 0, 'rmse_log': 0, 'abs_rel': 0, 'sq_rel': 0,
        'delta1': 0, 'delta2': 0, 'delta3': 0, 'log10': 0,
        'silog': 0, 'pearson_corr': 0, 'spearman_corr': 0
    }
    
    metrics_proc = {
        'scale_factor': 0, 'rmse': 0, 'rmse_log': 0, 'abs_rel': 0, 'sq_rel': 0,
        'delta1': 0, 'delta2': 0, 'delta3': 0, 'log10': 0,
        'silog': 0, 'pearson_corr': 0, 'spearman_corr': 0
    }
    
    # Get indices of samples to process
    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    
    if args.random_samples:
        indices = random.sample(range(len(dataset)), num_samples)
    else:
        indices = list(range(num_samples))
    
    # Process images
    valid_samples = 0
    
    print(f"Starting validation on {len(indices)} samples...")
    for idx in tqdm(indices):
        try:
            # Load sample from dataset
            sample = dataset[idx]
            
            # Extract data
            original_image = sample['original_image']
            processed_image = sample['processed_image']
            gt_depth = sample['depth']
            processed_img_path = sample['processed_image_path']
            
            # Extract basename for output file
            basename = os.path.splitext(os.path.basename(processed_img_path))[0]
            
            if original_image is None or processed_image is None or gt_depth is None:
                print(f"Invalid sample at index {idx}")
                continue
            
            # Run inference on both original and processed images
            orig_pred_depth = model.predict(original_image, args.input_size)
            proc_pred_depth = model.predict(processed_image, args.input_size)
            
            # Post-process predictions based on model output characteristics
            orig_pred_processed, orig_pred_display, orig_conversion_info = model.postprocess_prediction(orig_pred_depth, gt_depth.shape)
            proc_pred_processed, proc_pred_display, proc_conversion_info = model.postprocess_prediction(proc_pred_depth, gt_depth.shape)
            
            # Ensure model characteristics are available
            assert model.output_characteristics is not None, "Model output characteristics not initialized"
            
            # Create combined visualization (using display depth for better visualization)
            visualize_comparison_combined(
                original_image, processed_image, gt_depth, 
                orig_pred_display, proc_pred_display,
                basename, args.output_dir, 
                model_characteristics=model.output_characteristics,
                conversion_info=orig_conversion_info
            )
            
            # Compute metrics: GT is depth, pred is depth (already processed by postprocess_prediction)
            # Note: After postprocess_prediction, predictions are always in depth domain (meters)
            metrics_orig_sample = compute_depth_metrics(
                orig_pred_processed, gt_depth, 
                is_gt_disparity=False, is_pred_disparity=False
            )
            metrics_proc_sample = compute_depth_metrics(
                proc_pred_processed, gt_depth, 
                is_gt_disparity=False, is_pred_disparity=False
            )
            
            # Add to totals for metrics calculation
            for k, v in metrics_orig_sample.items():
                metrics_orig[k] += v
            for k, v in metrics_proc_sample.items():
                metrics_proc[k] += v
            
            valid_samples += 1
        
        except Exception as e:
            print(f"Error processing sample at index {idx}: {e}")
    
    # Calculate average metrics
    print(f"\nProcessed {valid_samples} valid samples")
    
    if valid_samples > 0:
        # Calculate average metrics for original images
        avg_metrics_orig = {k: v / valid_samples for k, v in metrics_orig.items()}
        
        # Calculate average metrics for processed images
        avg_metrics_proc = {k: v / valid_samples for k, v in metrics_proc.items()}
        
        # Print model information
        assert model.output_characteristics is not None, "Model output characteristics not initialized"
        print(f"\nModel: {model_name}")
        print(f"Model output: {model.output_characteristics.display_name}")
        print(f"Is metric: {model.output_characteristics.is_metric}")
        print(f"Is disparity: {model.output_characteristics.is_disparity}")
        
        # Print metrics in table format
        print("\n" + "="*90)
        print(f"{'Metric':<15} {'Original':<15} {'SeaErra':<15} {'Improvement':<20} {'Status':<10}")
        print("="*90)
        
        # Define metric names and whether higher is better
        metric_info = [
            ('Scale Factor', 'scale_factor', 'neutral'),
            ('RMSE', 'rmse', 'lower_better'),
            ('RMSE-log', 'rmse_log', 'lower_better'),
            ('Abs Rel', 'abs_rel', 'lower_better'),
            ('Sq Rel', 'sq_rel', 'lower_better'),
            ('Delta < 1.25', 'delta1', 'higher_better'),
            ('Delta < 1.25^2', 'delta2', 'higher_better'),
            ('Delta < 1.25^3', 'delta3', 'higher_better'),
            ('Log10', 'log10', 'lower_better'),
            ('SILog', 'silog', 'lower_better'),
            ('Pearson Corr', 'pearson_corr', 'higher_better'),
            ('Spearman Corr', 'spearman_corr', 'higher_better'),
        ]
        
        for display_name, key, criterion in metric_info:
            orig_val = avg_metrics_orig[key]
            proc_val = avg_metrics_proc[key]
            improvement = proc_val - orig_val
            
            # Determine status
            if criterion == 'neutral':
                status = ""
            elif criterion == 'higher_better':
                status = "Better" if improvement > 0 else ("Worse" if improvement < 0 else "Same")
            elif criterion == 'lower_better':
                status = "Better" if improvement < 0 else ("Worse" if improvement > 0 else "Same")
            else:
                status = ""
            
            # Format improvement with appropriate sign
            improvement_str = f"{improvement:+.4f}"
            
            print(f"{display_name:<15} {orig_val:<15.4f} {proc_val:<15.4f} {improvement_str:<20} {status:<10}")
        
        print("="*90)
        
        # Save overall metrics to a JSON file
        assert model.output_characteristics is not None, "Model output characteristics not initialized"
        serializable_metrics = {
            'model': model_name,
            'model_characteristics': {
                'display_name': model.output_characteristics.display_name,
                'is_metric': model.output_characteristics.is_metric,
                'is_disparity': model.output_characteristics.is_disparity,
                'output_unit': model.output_characteristics.output_unit,
                'target_unit': model.output_characteristics.target_unit
            },
            'original': {k: float(v) for k, v in avg_metrics_orig.items()},
            'processed': {k: float(v) for k, v in avg_metrics_proc.items()}
        }
        with open(os.path.join(args.output_dir, 'overall_metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
    else:
        print("No valid samples processed.")

def main():
    parser = argparse.ArgumentParser(description='Unified Depth Model Evaluation on FLSea Dataset')
    
    # Common arguments
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['depthanything', 'zoedepth'],
                        help='Type of depth estimation model to use')
    parser.add_argument('--data-root', type=str, default='assets/FLSea/red_sea/pier_path',
                        help='Path to FLSea dataset')
    parser.add_argument('--output-dir', type=str, default='visualizations/flsea_test',
                        help='Output directory for visualizations and metrics')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input size for model inference')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to evaluate. Use -1 to evaluate all samples')
    parser.add_argument('--random-samples', action='store_true',
                        help='Select random samples instead of the first N')
    
    # DepthAnything specific arguments
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='DepthAnything v2 encoder type')
    parser.add_argument('--metric', action='store_true',
                        help='Use metric DepthAnything v2 model')
    
    # ZoeDepth specific arguments
    parser.add_argument('--zoedepth-type', type=str, default='N',
                        choices=['N', 'K', 'NK'],
                        help='ZoeDepth model type: N (Indoor/outdoor), K (Outdoor), NK (Outdoor with NYU encoder)')
    
    args = parser.parse_args()
    
    # Update output directory to include model information
    if args.model_type == 'depthanything':
        model_suffix = f"depthanything_{args.encoder}" + ("_metric" if args.metric else "")
    elif args.model_type == 'zoedepth':
        model_suffix = f"zoedepth_{args.zoedepth_type}"
    else:
        model_suffix = args.model_type
    
    # Create a descriptive output directory name
    base_output_dir = args.output_dir
    args.output_dir = os.path.join(base_output_dir, model_suffix)
    
    print(f"Output directory: {args.output_dir}")
    
    validate_flsea(args)

if __name__ == '__main__':
    main()
