#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Dataset Validation Script
Evaluates model performance on various depth estimation datasets with optional image enhancement
Supports multiple depth estimation models: DepthAnything v2, ZoeDepth, and VGGT
Supports multiple dataset types: FLSea (with SeaErra enhancement), Standard depth datasets

Usage:
    # For DepthAnything v2
    python test.py --model-type depthanything --encoder vitl --dataset-type standard --data-root assets/FLSea/red_sea/pier_path --num-samples 10
    
    # For ZoeDepth
    python test.py --model-type zoedepth --zoedepth-type N --dataset-type standard --data-root assets/FLSea/red_sea/pier_path --num-samples 10
    
    # For VGGT
    python test.py --model-type vggt --dataset-type standard --data-root assets/FLSea/red_sea/pier_path --num-samples 10
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

# Import dataset base classes
from dataset_base import create_dataset, get_available_datasets

# Import depth processing utilities
from depth_utils import (
    align_depth_scale, 
    compute_depth_metrics, 
    format_depth_label, 
    prepare_depth_visualization,
    create_depth_range_text,
    extract_depth_maps,
    create_robust_valid_mask
)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def visualize_comparison_combined(orig_img, enhanced_img, gt_depth, orig_pred_depth, enhanced_pred_depth, 
                                file_prefix, output_dir, model_characteristics: ModelOutputCharacteristics, 
                                conversion_info=None):
    """
    Create a visualization for depth prediction results
    
    Args:
        orig_img: Original RGB image
        enhanced_img: Enhanced RGB image (may be same as original if no enhancement)
        gt_depth: Ground truth depth map
        orig_pred_depth: Predicted depth map from original image
        enhanced_pred_depth: Predicted depth map from enhanced image (may be same as original)
        file_prefix: Prefix for saved files
        output_dir: Directory to save visualizations
        model_characteristics: ModelOutputCharacteristics describing the model output
        conversion_info: Information about unit conversion applied
    """
    # Check if we have actual enhancement (different images)
    has_enhancement = not np.array_equal(orig_img, enhanced_img)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare for visualization - handle invalid values
    gt_viz = np.zeros_like(gt_depth, dtype=np.float32)
    orig_pred_viz = np.zeros_like(orig_pred_depth, dtype=np.float32)
    enhanced_pred_viz = np.zeros_like(enhanced_pred_depth, dtype=np.float32)
    
    # Create robust masks for valid pixels that filter out extreme values
    gt_valid = create_robust_valid_mask(gt_depth)
    orig_pred_valid = create_robust_valid_mask(orig_pred_depth)
    enhanced_pred_valid = create_robust_valid_mask(enhanced_pred_depth)
    
    # Create a combined mask of valid pixels across all depth maps for consistent normalization
    gt_valid_depths = []
    pred_valid_depths = []
    
    if np.any(gt_valid):
        gt_valid_depths.append(gt_depth[gt_valid])
    if np.any(orig_pred_valid):
        pred_valid_depths.append(orig_pred_depth[orig_pred_valid])
    if np.any(enhanced_pred_valid):
        pred_valid_depths.append(enhanced_pred_depth[enhanced_pred_valid])
    
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
        if np.any(enhanced_pred_valid):
            enhanced_pred_viz[enhanced_pred_valid] = np.clip((enhanced_pred_depth[enhanced_pred_valid] - pred_min_depth) / (pred_max_depth - pred_min_depth + 1e-8), 0, 1)
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
    
    enhanced_pred_colored = np.zeros((*enhanced_pred_viz.shape, 4), dtype=np.float32)
    enhanced_pred_colored[enhanced_pred_valid] = cm_jet(1.0 - enhanced_pred_viz[enhanced_pred_valid])  # Invert mapping for depth
    enhanced_pred_colored[~enhanced_pred_valid, 3] = 0
    enhanced_pred_rgb = enhanced_pred_colored[:, :, :3]
    
    pred_title_suffix = "Depth Prediction (Near=Red, Far=Blue)"
    
    # Create visualization with different layouts based on enhancement availability
    if has_enhancement:
        # 2x3 layout for comparison
        plt.figure(figsize=(18, 12))
        
        # First row: Input images and ground truth
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(enhanced_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title('Enhanced Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(gt_rgb)
        plt.title(f'Ground Truth Depth (Near=Red, Far=Blue)\n{gt_range_text}')
    else:
        # 4x1 layout for single image analysis (4 rows, 1 column)
        plt.figure(figsize=(10, 20))
        
        # First row: Input image
        plt.subplot(4, 1, 1)
        plt.imshow(cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')
        
        # Second row: Ground truth depth
        plt.subplot(4, 1, 2)
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
    if has_enhancement:
        # Show both predictions for comparison
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
                    full_label = format_depth_label(pred_val, model_characteristics, conversion_info, gt_valid_depths, gt_depth, orig_pred_depth, orig_pred_valid, gt_valid)
                    
                    plt.annotate(full_label, 
                               xy=(x_center, y_pos), 
                               xytext=(x_center + w//10, y_pos),
                               fontsize=8, color='white', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', color='white', lw=1))
        
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(enhanced_pred_rgb)
        plt.title(f'Enhanced Image {pred_title_suffix}\n{pred_range_text}')
        
        # Add depth scale markers for enhanced predictions
        if pred_valid_depths:
            h, w = enhanced_pred_depth.shape
            x_center = w // 2
            y_positions = [h // 6, h // 2, 5 * h // 6]
            
            for y_pos in y_positions:
                if enhanced_pred_valid[y_pos, x_center]:
                    pred_val = enhanced_pred_depth[y_pos, x_center]
                    full_label = format_depth_label(pred_val, model_characteristics, conversion_info, gt_valid_depths, gt_depth, enhanced_pred_depth, enhanced_pred_valid, gt_valid)
                    
                    plt.annotate(full_label, 
                               xy=(x_center, y_pos), 
                               xytext=(x_center + w//10, y_pos),
                               fontsize=8, color='white', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', color='white', lw=1))
        
        plt.axis('off')
        
        # Error comparison for enhancement
        plt.subplot(2, 3, 6)
        _plot_error_comparison(gt_depth, orig_pred_depth, enhanced_pred_depth, gt_valid, orig_pred_valid, enhanced_pred_valid, "Enhanced")
    else:
        # 4x1 layout: Third row - Predicted depth
        plt.subplot(4, 1, 3)
        plt.imshow(orig_pred_rgb)
        plt.title(f'Predicted {pred_title_suffix}\n{pred_range_text}')
        
        # Add depth scale markers for prediction
        if pred_valid_depths:
            h, w = orig_pred_depth.shape
            x_center = w // 2
            y_positions = [h // 6, h // 2, 5 * h // 6]
            
            for y_pos in y_positions:
                if orig_pred_valid[y_pos, x_center]:
                    pred_val = orig_pred_depth[y_pos, x_center]
                    full_label = format_depth_label(pred_val, model_characteristics, conversion_info, gt_valid_depths, gt_depth, orig_pred_depth, orig_pred_valid, gt_valid)
                    
                    plt.annotate(full_label, 
                               xy=(x_center, y_pos), 
                               xytext=(x_center + w//10, y_pos),
                               fontsize=8, color='white', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', color='white', lw=1))
        
        plt.axis('off')
        
        # 4x1 layout: Fourth row - Error analysis
        plt.subplot(4, 1, 4)
        _plot_single_error_analysis(gt_depth, orig_pred_depth, gt_valid, orig_pred_valid)
    
    plt.tight_layout()
    
    # Save with appropriate suffix based on enhancement
    if has_enhancement:
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_comparison.png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_analysis.png"), dpi=300, bbox_inches='tight')
    
    plt.close()

def validate_dataset(args):
    """
    Validate depth estimation models on various datasets
    
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
    elif args.model_type == 'vggt':
        model = create_model(
            model_type='vggt',
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            multi_image_mode=args.vggt_multi_image
        )
        model_name = get_model_name('vggt', multi_image_mode=args.vggt_multi_image)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    print(f"Model: {model_name}")
    
    # Create dataset using factory function
    print(f"Loading {args.dataset_type} dataset from {args.data_root}")
    dataset = create_dataset(args.dataset_type, args.data_root)
    
    if len(dataset) == 0:
        print("No valid samples found in the dataset")
        return
    
    print(f"Found {len(dataset)} valid samples in the {dataset.dataset_type} dataset")
    print(f"Dataset has enhancement: {dataset.has_enhancement}")
    if dataset.has_enhancement:
        print(f"Enhancement method: {dataset.enhancement_name}")
    
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
    
    # Special handling for VGGT multi-image mode
    if args.model_type == 'vggt' and args.vggt_multi_image:
        # VGGT multi-image mode: process multiple images together
        sequence_length = min(args.vggt_sequence_length, len(indices))
        
        # Load all samples for multi-image processing
        all_samples = []
        original_images = []
        enhanced_images = []
        gt_depths = []
        basenames = []
        
        print(f"VGGT multi-image mode: loading {sequence_length} images for batch processing...")
        
        for i, idx in enumerate(indices[:sequence_length]):
            try:
                sample = dataset[idx]
                original_image = sample['original_image']
                enhanced_image = sample['enhanced_image']
                gt_depth = sample['depth']
                basename = sample['basename']
                
                # For datasets without enhancement, use original image as "processed" image
                if enhanced_image is None:
                    enhanced_image = original_image.copy()
                
                if original_image is None or enhanced_image is None or gt_depth is None:
                    print(f"Invalid sample at index {idx}")
                    continue
                
                all_samples.append(sample)
                original_images.append(original_image)
                enhanced_images.append(enhanced_image)
                gt_depths.append(gt_depth)
                basenames.append(basename)
                
            except Exception as e:
                print(f"Error loading sample at index {idx}: {e}")
        
        if len(original_images) == 0:
            print("No valid samples loaded for VGGT multi-image processing")
            return
        
        print(f"Loaded {len(original_images)} valid samples for VGGT multi-image processing")
        
        # Run VGGT multi-image inference
        try:
            orig_results = model.predict(original_images, args.input_size)
            proc_results = model.predict(enhanced_images, args.input_size)
            
            # Extract and normalize depth maps from results
            orig_depth_maps = extract_depth_maps(orig_results)
            proc_depth_maps = extract_depth_maps(proc_results)
            
            # Process each image result
            for i, (original_image, enhanced_image, gt_depth, basename) in enumerate(zip(original_images, enhanced_images, gt_depths, basenames)):
                if i >= len(orig_depth_maps) or i >= len(proc_depth_maps):
                    print(f"Missing depth prediction for sample {i}")
                    continue
                
                orig_pred_depth = orig_depth_maps[i]
                proc_pred_depth = proc_depth_maps[i]
                
                # Post-process predictions based on model output characteristics
                orig_pred_processed, orig_pred_display, orig_conversion_info = model.postprocess_prediction(orig_pred_depth, gt_depth.shape)
                proc_pred_processed, proc_pred_display, proc_conversion_info = model.postprocess_prediction(proc_pred_depth, gt_depth.shape)
                
                # Ensure model characteristics are available
                assert model.output_characteristics is not None, "Model output characteristics not initialized"
                
                # Create combined visualization (using display depth for better visualization)
                visualize_comparison_combined(
                    original_image, enhanced_image, gt_depth, 
                    orig_pred_display, proc_pred_display,
                    f"multi_{i}_{basename}", args.output_dir, 
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
            print(f"Error in VGGT multi-image inference: {e}")
            return
    
    else:
        # Standard single-image processing for other models or VGGT single-image mode
        for idx in tqdm(indices):
            try:
                # Load sample from dataset
                sample = dataset[idx]
                
                # Extract data
                original_image = sample['original_image']
                enhanced_image = sample['enhanced_image']  # May be None for datasets without enhancement
                gt_depth = sample['depth']
                enhanced_image_path = sample['enhanced_image_path']  # May be None
                basename = sample['basename']
                
                # For datasets without enhancement, use original image as "processed" image
                if enhanced_image is None:
                    enhanced_image = original_image.copy()
                    enhanced_image_path = f"original_{basename}"
                
                # Extract basename for output file
                if enhanced_image_path:
                    file_basename = os.path.splitext(os.path.basename(enhanced_image_path))[0]
                else:
                    file_basename = basename
                
                if original_image is None or enhanced_image is None or gt_depth is None:
                    print(f"Invalid sample at index {idx}")
                    continue
                
                # Run inference on both original and enhanced images
                orig_pred_depth = model.predict(original_image, args.input_size)
                proc_pred_depth = model.predict(enhanced_image, args.input_size)
                
                # Post-process predictions based on model output characteristics
                orig_pred_processed, orig_pred_display, orig_conversion_info = model.postprocess_prediction(orig_pred_depth, gt_depth.shape)
                proc_pred_processed, proc_pred_display, proc_conversion_info = model.postprocess_prediction(proc_pred_depth, gt_depth.shape)
                
                # Ensure model characteristics are available
                assert model.output_characteristics is not None, "Model output characteristics not initialized"
                
                # Create combined visualization (using display depth for better visualization)
                visualize_comparison_combined(
                    original_image, enhanced_image, gt_depth, 
                    orig_pred_display, proc_pred_display,
                    file_basename, args.output_dir, 
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
        
        # Print metrics in table format with dataset-aware naming
        enhancement_name = dataset.enhancement_name if dataset.has_enhancement else "Original"
        print("\n" + "="*90)
        print(f"{'Metric':<15} {'Original':<15} {enhancement_name:<15} {'Improvement':<20} {'Status':<10}")
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

def _plot_error_comparison(gt_depth, orig_pred_depth, enhanced_pred_depth, gt_valid, orig_pred_valid, enhanced_pred_valid, enhancement_name):
    """Plot side-by-side error comparison for enhancement analysis."""
    # Compute error in original depth space (same as metrics calculation)
    orig_error = np.zeros_like(gt_depth)
    enhanced_error = np.zeros_like(gt_depth)
    valid_orig = gt_valid & orig_pred_valid
    valid_enhanced = gt_valid & enhanced_pred_valid
    
    eps = 1e-6
    
    # Apply scale alignment (same as compute_depth_metrics)
    if np.any(valid_orig):
        scale_orig, orig_pred_aligned = align_depth_scale(
            orig_pred_depth[valid_orig], gt_depth[valid_orig], method='median'
        )
        orig_error[valid_orig] = np.abs(gt_depth[valid_orig] - orig_pred_aligned)
    
    if np.any(valid_enhanced):
        scale_enhanced, enhanced_pred_aligned = align_depth_scale(
            enhanced_pred_depth[valid_enhanced], gt_depth[valid_enhanced], method='median'
        )
        enhanced_error[valid_enhanced] = np.abs(gt_depth[valid_enhanced] - enhanced_pred_aligned)
    
    # Create a side-by-side error comparison
    h, w = gt_depth.shape
    combined_error = np.zeros((h, w*2))
    combined_error[:, :w] = orig_error
    combined_error[:, w:] = enhanced_error
    
    # Normalize for visualization
    valid_error = (combined_error > 0)
    if np.any(valid_error):
        error_95th = np.percentile(combined_error[valid_error], 95)
        if error_95th > 0:
            combined_error_viz = np.clip(combined_error / error_95th, 0, 1)
        else:
            combined_error_viz = combined_error
        
        # Calculate error statistics
        orig_errors = orig_error[orig_error > 0]
        enhanced_errors = enhanced_error[enhanced_error > 0]
        
        if len(orig_errors) > 0 and len(enhanced_errors) > 0:
            avg_orig_error = np.mean(orig_errors)
            avg_enhanced_error = np.mean(enhanced_errors)
            improvement = (avg_orig_error - avg_enhanced_error) / avg_orig_error * 100
            
            error_title = f'Absolute Error Maps (meters)\nOriginal: {avg_orig_error:.4f}m, {enhancement_name}: {avg_enhanced_error:.4f}m\nImprovement: {improvement:.1f}%'
        else:
            error_title = f'Absolute Error Maps (meters)\nOriginal (left) vs {enhancement_name} (right)'
    else:
        combined_error_viz = combined_error
        error_title = f'Absolute Error Maps (meters)\nOriginal (left) vs {enhancement_name} (right)'
    
    # Apply colormap
    cmap = plt.colormaps['hot']
    error_rgb = cmap(combined_error_viz)[:, :, :3]
    
    plt.imshow(error_rgb)
    plt.title(error_title)
    plt.axis('off')
    
    # Add a vertical line to separate the two error maps
    plt.axvline(x=w-0.5, color='white', linestyle='-', linewidth=2)

def _plot_single_error_analysis(gt_depth, pred_depth, gt_valid, pred_valid):
    """Plot single error analysis for datasets without enhancement."""
    # Compute error
    error = np.zeros_like(gt_depth)
    valid = gt_valid & pred_valid
    
    if np.any(valid):
        scale, pred_aligned = align_depth_scale(
            pred_depth[valid], gt_depth[valid], method='median'
        )
        error[valid] = np.abs(gt_depth[valid] - pred_aligned)
    
    # Normalize for visualization
    valid_error = (error > 0)
    if np.any(valid_error):
        error_95th = np.percentile(error[valid_error], 95)
        if error_95th > 0:
            error_viz = np.clip(error / error_95th, 0, 1)
        else:
            error_viz = error
        
        # Calculate error statistics
        errors = error[error > 0]
        if len(errors) > 0:
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            error_title = f'Absolute Error Map (meters)\nMean: {avg_error:.4f}m, Max: {max_error:.4f}m'
        else:
            error_title = 'Absolute Error Map (meters)'
    else:
        error_viz = error
        error_title = 'Absolute Error Map (meters)'
    
    # Apply colormap
    cmap = plt.colormaps['hot']
    error_rgb = cmap(error_viz)[:, :, :3]
    
    plt.imshow(error_rgb)
    plt.title(error_title)
    plt.axis('off')

def main():
    parser = argparse.ArgumentParser(description='Unified Depth Model Evaluation on Various Datasets')
    
    # Common arguments
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['depthanything', 'zoedepth', 'vggt'],
                        help='Type of depth estimation model to use')
    parser.add_argument('--dataset-type', type=str, default='flsea',
                        choices=['flsea', 'standard'],
                        help='Type of dataset to use')
    parser.add_argument('--data-root', type=str, default='assets/FLSea/red_sea/pier_path',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='visualizations/validation',
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
    
    # VGGT specific arguments
    parser.add_argument('--vggt-multi-image', action='store_true',
                        help='Enable VGGT multi-image mode for enhanced geometric consistency')
    parser.add_argument('--vggt-sequence-length', type=int, default=10,
                        help='Number of images to use in VGGT multi-image mode (if dataset supports it)')
    
    args = parser.parse_args()
    
    # Update output directory to include model information
    if args.model_type == 'depthanything':
        model_suffix = f"depthanything_{args.encoder}" + ("_metric" if args.metric else "")
    elif args.model_type == 'zoedepth':
        model_suffix = f"zoedepth_{args.zoedepth_type}"
    elif args.model_type == 'vggt':
        if args.vggt_multi_image:
            model_suffix = "vggt_multi"
        else:
            model_suffix = "vggt"
    else:
        model_suffix = args.model_type
    
    # Create a descriptive output directory name
    base_output_dir = args.output_dir
    args.output_dir = os.path.join(base_output_dir, f"{args.dataset_type}_{model_suffix}")
    
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output directory: {args.output_dir}")
    
    validate_dataset(args)

if __name__ == '__main__':
    main()
