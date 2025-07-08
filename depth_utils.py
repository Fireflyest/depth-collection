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
    pred_robust_mask = create_robust_valid_mask(pred_aligned)
    gt_robust_mask = create_robust_valid_mask(gt)
    
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


def format_depth_label(pred_val: float, 
                      model_characteristics, 
                      conversion_info: Optional[dict], 
                      gt_valid_depths: List[np.ndarray], 
                      gt_depth: np.ndarray, 
                      pred_depth: np.ndarray, 
                      pred_valid: np.ndarray, 
                      gt_valid: np.ndarray) -> str:
    """
    Format depth label for visualization showing both original and converted values.
    
    Args:
        pred_val: Predicted depth value at a specific pixel
        model_characteristics: Model output characteristics
        conversion_info: Information about unit conversion applied
        gt_valid_depths: List of valid ground truth depth arrays
        gt_depth: Ground truth depth map
        pred_depth: Predicted depth map
        pred_valid: Valid prediction mask
        gt_valid: Valid ground truth mask
        
    Returns:
        Formatted label string
    """
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
        if model_characteristics.output_unit == 'affine-invariant':
            raw_label = f'{pred_val:.4f}(aff-inv)'
        else:
            raw_label = f'{pred_val:.4f}(rel)'
        
        # For non-metric depth, try to estimate actual depth using GT scale
        if gt_valid_depths:
            gt_sample = gt_depth[gt_valid]
            pred_sample = pred_depth[pred_valid & gt_valid]
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
    return raw_label + meter_label


def prepare_depth_visualization(depth_map: np.ndarray, 
                               percentile_range: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Prepare depth map for visualization with proper normalization
    
    Args:
        depth_map: Input depth map
        percentile_range: Percentile range for normalization (min, max)
        
    Returns:
        Normalized and inverted depth map for visualization
    """
    # Create robust mask for valid pixels that filters out extreme values
    valid_mask = create_robust_valid_mask(depth_map)
    
    if not np.any(valid_mask):
        return np.zeros_like(depth_map)
    
    # Normalize depth for visualization
    valid_depths = depth_map[valid_mask]
    min_depth = np.percentile(valid_depths, percentile_range[0])
    max_depth = np.percentile(valid_depths, percentile_range[1])
    
    # Normalize to [0, 1] range
    normalized = np.zeros_like(depth_map)
    normalized[valid_mask] = np.clip(
        (depth_map[valid_mask] - min_depth) / (max_depth - min_depth + 1e-8), 0, 1
    )
    
    # Invert for depth visualization (near=red, far=blue in jet colormap)
    return 1.0 - normalized


def create_depth_range_text(depth_map: np.ndarray, 
                           model_characteristics, 
                           conversion_info: Optional[dict] = None,
                           gt_depth: Optional[np.ndarray] = None,
                           percentile_range: Tuple[float, float] = (5, 95)) -> str:
    """
    Create range text for depth visualization
    
    Args:
        depth_map: Depth map
        model_characteristics: Model output characteristics
        conversion_info: Unit conversion information
        gt_depth: Ground truth depth for scale estimation (for non-metric models)
        percentile_range: Percentile range for min/max calculation
        
    Returns:
        Formatted range text
    """
    valid_mask = create_robust_valid_mask(depth_map)
    
    if not np.any(valid_mask):
        return "N/A"
    
    min_depth = np.percentile(depth_map[valid_mask], percentile_range[0])
    max_depth = np.percentile(depth_map[valid_mask], percentile_range[1])
    
    if conversion_info and conversion_info.get('applied', False):
        # Show original and converted range
        min_orig = model_characteristics.get_original_value_from_converted(min_depth)
        max_orig = model_characteristics.get_original_value_from_converted(max_depth)
        return f'{min_orig:.3f}-{max_orig:.3f}({model_characteristics.output_unit[:4]})\n→{min_depth:.3f}-{max_depth:.3f}m'
    elif model_characteristics.is_metric:
        return f'{min_depth:.3f}-{max_depth:.3f}m'
    else:
        # Non-metric: show relative values with approximate metric conversion
        if model_characteristics.output_unit == 'affine-invariant':
            range_text = f'{min_depth:.3f}-{max_depth:.3f}(aff-inv)'
        else:
            range_text = f'{min_depth:.3f}-{max_depth:.3f}(rel)'
        
        # Try to estimate metric range using GT scale
        if gt_depth is not None:
            gt_valid = ~np.isnan(gt_depth) & ~np.isinf(gt_depth) & (gt_depth > 0)
            if np.any(gt_valid):
                gt_sample = gt_depth[gt_valid]
                pred_sample = depth_map[valid_mask & gt_valid]
                if len(pred_sample) > 0 and len(gt_sample) > 0:
                    scale = np.median(gt_sample) / np.median(pred_sample)
                    min_metric = min_depth * scale
                    max_metric = max_depth * scale
                    range_text += f'\n≈{min_metric:.3f}-{max_metric:.3f}m'
                    return range_text
        
        range_text += '\n≈?-?m'
        return range_text


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

def create_robust_valid_mask(depth_map: np.ndarray, max_reasonable_depth: float = 30, min_reasonable_depth: float = 0.2) -> np.ndarray:
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
