#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Depth Processing Utilities
Common functions for depth map processing, evaluation, and visualization.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Union
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def align_depth_scale(pred: np.ndarray, gt: np.ndarray, method: str = 'median') -> Tuple[float, np.ndarray]:
    """
    Align the scale of predicted depth to ground truth depth
    
    Args:
        pred: predicted depth map
        gt: ground truth depth map
        method: alignment method ('median', 'mean', 'least_squares')
    
    Returns:
        Tuple of (scale factor, aligned prediction)
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


def compute_depth_metrics(pred: Union[np.ndarray, torch.Tensor], 
                         gt: Union[np.ndarray, torch.Tensor], 
                         mask: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                         align_method: str = 'median', 
                         is_gt_disparity: bool = True, 
                         is_pred_disparity: bool = False) -> dict:
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
    
    # Create basic valid mask using simple validity check (no range filtering yet)
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    
    # Apply basic validity masks (no range filtering at this stage)
    # Use simple mask for both pred and gt - range filtering will be done after scale alignment
    pred_valid = create_simple_valid_mask(pred) & mask
    gt_valid = create_simple_valid_mask(gt) & mask
    
    # Use intersection of both valid masks
    valid_mask = pred_valid & gt_valid
    
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
    
    # Now apply robust filtering to the final metric depths (after scale alignment)
    # This filters out unreasonable depth values in the final metric domain
    pred_robust_mask = create_metric_valid_mask(pred_aligned)
    gt_robust_mask = create_metric_valid_mask(gt)
    
    # Apply robust filtering to final aligned data
    final_valid_mask = pred_robust_mask & gt_robust_mask
    
    if final_valid_mask.sum() == 0:
        # Return zeros if no valid pixels after robust filtering
        return {
            'scale_factor': scale_factor,
            'rmse': 0, 'rmse_log': 0, 'abs_rel': 0, 'sq_rel': 0,
            'delta1': 0, 'delta2': 0, 'delta3': 0, 'log10': 0,
            'silog': 0, 'pearson_corr': 0, 'spearman_corr': 0
        }
    
    # Apply robust mask to both aligned prediction and GT
    pred_aligned = pred_aligned[final_valid_mask]
    gt = gt[final_valid_mask]
    
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

def extract_depth_maps(results):
    """
    Extract depth maps from VGGT multi-image inference results.
    Handles different output formats and converts to list of 2D arrays.
    """
    if isinstance(results, dict) and 'depth_maps' in results:
        depth_maps = results['depth_maps']
    elif isinstance(results, np.ndarray):
        depth_maps = results
    elif isinstance(results, list):
        return results  # Already in list format
    else:
        return [results]  # Single result
    
    # Handle different array dimensions
    if isinstance(depth_maps, np.ndarray):
        if depth_maps.ndim == 5:  # (1, N, H, W, 1) - VGGT batch format
            depth_maps = depth_maps.squeeze(0)  # Remove batch dimension -> (N, H, W, 1)
            if depth_maps.shape[-1] == 1:
                depth_maps = depth_maps.squeeze(-1)  # Remove channel dimension -> (N, H, W)
        elif depth_maps.ndim == 4:  # (N, H, W, 1) or (N, H, W, C)
            if depth_maps.shape[-1] == 1:
                depth_maps = depth_maps.squeeze(-1)  # (N, H, W)
        # Convert to list of individual depth maps
        if depth_maps.ndim == 3:  # (N, H, W)
            return [depth_maps[i] for i in range(depth_maps.shape[0])]
    
    return [depth_maps] if not isinstance(depth_maps, list) else depth_maps

def create_metric_valid_mask(depth_map: np.ndarray, max_reasonable_depth: float = 30, min_reasonable_depth: float = 0.2) -> np.ndarray:
    """
    Create a robust valid mask that filters out invalid and extreme depth values
    
    Args:
        depth_map: Input depth map
        max_reasonable_depth: Maximum reasonable depth value (meters or relative units)
        min_reasonable_depth: Minimum reasonable depth value (meters or relative units)
        
    Returns:
        Boolean mask for valid pixels
    """
    # Basic validity checks
    basic_valid = ~np.isnan(depth_map) & ~np.isinf(depth_map) & (depth_map > 0)
    
    if not np.any(basic_valid):
        return basic_valid
    
    # Filter out both too close and too far depths
    reasonable_mask = (depth_map >= min_reasonable_depth) & (depth_map <= max_reasonable_depth)
    
    # Combine basic validity with reasonable depth range
    final_mask = basic_valid & reasonable_mask
    
    return final_mask

def create_simple_valid_mask(depth_map: np.ndarray) -> np.ndarray:
    """
    Create a simple valid mask for GT depth that only filters basic invalid values
    (NaN, infinity, negative values). Does not filter extreme depth values.
    
    Args:
        depth_map: Input GT depth map
        
    Returns:
        Boolean mask for valid GT pixels
    """
    if depth_map is None or depth_map.size == 0:
        return np.array([], dtype=bool)
    
    # Simple mask: only filter NaN, infinity, and negative values
    valid_mask = (~np.isnan(depth_map)) & (~np.isinf(depth_map)) & (depth_map > 0)
    
    return valid_mask

def add_depth_markers(ax, valid_mask, display_depth, aligned_depth, is_matric: bool = False):
    """Add depth value markers at specific points"""
    if not np.any(valid_mask):
        return
        
    h, w = display_depth.shape
    x_center = w // 2
    y_positions = [h // 6, h // 2, 5 * h // 6]  # top, middle, bottom
    
    for y_pos in y_positions:
        if valid_mask[y_pos, x_center]:
            display_depth[y_pos, x_center]
            label = f'{display_depth[y_pos, x_center]:.3f}'
            if not is_matric:
                label += f'\n≈{aligned_depth[y_pos, x_center]:.3f}'
            label += 'm'

            ax.annotate(label, 
                        xy=(x_center, y_pos), 
                        xytext=(x_center + w//8, y_pos),
                        fontsize=6, color='white', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1))


def prepare_depths_for_visualization(gt_depth, predictions, model_names, current_row):
    """Prepare depth visualizations with individual normalization for each depth map"""
    # Prepare GT visualization (individual normalization) - use simple mask for GT
    gt_valid_mask, gt_min, gt_max, gt_viz = prepare_individual_depth_visualization(gt_depth, use_robust_mask=False)
    range_texts: dict[str, str] = {'gt': f"{gt_min:.3f}-{gt_max:.3f}m"}

    # Prepare prediction visualizations (individual normalization for each)
    display_vizs = {}
    display_depths = {}
    aligned_depths = {}  # Store aligned depths for marker annotation
    aligned_valid_masks = {}
    for model_name in model_names:
        if (current_row < len(predictions[model_name]) and 
            predictions[model_name][current_row] is not None):
            pred_data = predictions[model_name][current_row]
            display_depth = pred_data['display_depth']
            model_chars = pred_data['model_characteristics']
            conversion_info = pred_data['conversion_info']

            # Align depth
            display_depths[model_name] = display_depth
            aligned_depths[model_name] = display_depth.copy()
            # For affine-invariant depth, convert to metric using GT scale alignment
            if (model_chars.output_unit == 'affine-invariant' and not conversion_info.get('applied', False)):
                # Scale align with GT to get metric depth for visualization
                display_valid_mask = create_simple_valid_mask(display_depth)
                display_vals = display_depth[display_valid_mask]
                gt_vals = gt_depth[gt_valid_mask]
                if len(gt_vals) > 0 and len(display_vals) > 0:
                    scale, _ = align_depth_scale(display_vals, gt_vals, method='median')
                    aligned_depths[model_name] = display_depth * scale

            # Individual normalization for this prediction - use robust mask for predictions
            aligned_valid_mask, aligned_min, aligned_max, pred_viz = prepare_individual_depth_visualization(aligned_depths[model_name], use_robust_mask=True)
            display_vizs[model_name] = pred_viz
            aligned_valid_masks[model_name] = aligned_valid_mask
            
            # Create range text with proper unit handling
            if np.any(aligned_valid_mask):
                pred_min = np.percentile(display_depth[aligned_valid_mask], 2)
                pred_max = np.percentile(display_depth[aligned_valid_mask], 98)
                
                if conversion_info and conversion_info.get('applied', False):
                    # Show original and converted range
                    min_orig = model_chars.get_original_value_from_converted(pred_min)
                    max_orig = model_chars.get_original_value_from_converted(pred_max)
                    range_texts[model_name] = f'{min_orig:.3f}-{max_orig:.3f}({model_chars.output_unit[:4]})\n→{pred_min:.3f}-{pred_max:.3f}m'
                elif model_chars.is_metric:
                    range_texts[model_name] = f'{pred_min:.3f}-{pred_max:.3f}m'
                else:
                    # Non-metric: show relative values
                    if model_chars.output_unit == 'affine-invariant':
                        range_texts[model_name] = f'{pred_min:.3f}-{pred_max:.3f}(aff-inv)'
                    else:
                        range_texts[model_name] = f'{pred_min:.3f}-{pred_max:.3f}(rel)'
                    range_texts[model_name] += f'\n≈{aligned_min:.3f}-{aligned_max:.3f}m'
            else:
                range_texts[model_name] = "N/A"
        else:
            display_vizs[model_name] = np.zeros((*gt_depth.shape, 3), dtype=np.float32)  # Black image for missing predictions
            range_texts[model_name] = "N/A"
            aligned_depths[model_name] = None
    
    return gt_viz, gt_valid_mask, display_depths, display_vizs, aligned_valid_masks, aligned_depths, range_texts

def prepare_individual_depth_visualization(depth_map: np.ndarray, use_robust_mask: bool = True) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Prepare individual depth map for visualization with independent normalization"""
    # Create mask for valid pixels - use robust for predictions, simple for GT
    if use_robust_mask:
        valid_mask = create_metric_valid_mask(depth_map)
    else:
        valid_mask = create_simple_valid_mask(depth_map)
    
    if not np.any(valid_mask):
        # Return black RGB image for no valid data
        return np.zeros((*depth_map.shape, 3), dtype=np.float32), 0, 0, np.zeros((*depth_map.shape, 3), dtype=np.float32)
    
    # Normalize depth for visualization using its own range
    valid_depths = depth_map[valid_mask]
    min_depth = np.percentile(valid_depths, 2)
    max_depth = np.percentile(valid_depths, 98)
    
    # Normalize to [0, 1] range
    normalized = np.zeros_like(depth_map)
    normalized[valid_mask] = np.clip(
        (depth_map[valid_mask] - min_depth) / (max_depth - min_depth + 1e-8), 0, 1
    )
    
    # Invert for depth visualization (near=red, far=blue in jet colormap)
    normalized_inverted = 1.0 - normalized
    
    # Apply jet colormap and convert to RGB
    cm_jet = plt.colormaps['jet']
    colored = np.zeros((*depth_map.shape, 4), dtype=np.float32)
    colored[valid_mask] = cm_jet(normalized_inverted[valid_mask])
    # Invalid pixels remain black (RGB = [0,0,0])
    rgb_image = colored[:, :, :3]  # Extract RGB channels
    
    return valid_mask, float(min_depth), float(max_depth), rgb_image