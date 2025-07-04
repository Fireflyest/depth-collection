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
from skimage.transform import resize

# Import models
from depth_anything_v2.dpt import DepthAnythingV2
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

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

def visualize_comparison_combined(orig_img, proc_img, gt_depth, orig_pred_depth, proc_pred_depth, file_prefix, output_dir, pred_is_disparity=False):
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
        gt_range_text = f"Range: {gt_min_depth:.2f} - {gt_max_depth:.2f}"
        
        # 为gt单独归一化
        if np.any(gt_valid):
            gt_viz[gt_valid] = np.clip((gt_depth[gt_valid] - gt_min_depth) / (gt_max_depth - gt_min_depth + 1e-8), 0, 1)
    else:
        gt_range_text = "Range: N/A"
    
    # 为预测深度确定范围
    if pred_valid_depths:
        pred_valid_values = np.concatenate(pred_valid_depths)
        pred_min_depth, pred_max_depth = np.percentile(pred_valid_values, [5, 95])
        pred_range_text = f"Range: {pred_min_depth:.2f} - {pred_max_depth:.2f}"
        
        # 归一化预测深度
        if np.any(orig_pred_valid):
            orig_pred_viz[orig_pred_valid] = np.clip((orig_pred_depth[orig_pred_valid] - pred_min_depth) / (pred_max_depth - pred_min_depth + 1e-8), 0, 1)
        if np.any(proc_pred_valid):
            proc_pred_viz[proc_pred_valid] = np.clip((proc_pred_depth[proc_pred_valid] - pred_min_depth) / (pred_max_depth - pred_min_depth + 1e-8), 0, 1)
    else:
        pred_range_text = "Range: N/A"
    
    # Apply colormap with custom handling for invalid values
    cm_jet = plt.colormaps['jet']  # Modern way to get colormap
    
    # Ground truth (always disparity map, but FLSea uses reverse color convention)
    # Original FLSea: near=blue, far=red. We want: near=red, far=blue
    # So we need to invert the GT visualization
    gt_colored = np.zeros((*gt_viz.shape, 4), dtype=np.float32)
    gt_colored[gt_valid] = cm_jet(1.0 - gt_viz[gt_valid])  # Invert to match our color convention (high=near=red)
    gt_colored[~gt_valid, 3] = 0  # Set alpha=0 for invalid regions
    gt_rgb = gt_colored[:, :, :3]  # Keep as float32 in range [0, 1] for imshow
    
    # Prediction visualization depends on whether prediction is disparity or relative depth
    if pred_is_disparity:
        # ZoeDepth case: prediction is disparity, use same INVERTED mapping as GT
        orig_pred_colored = np.zeros((*orig_pred_viz.shape, 4), dtype=np.float32)
        orig_pred_colored[orig_pred_valid] = cm_jet(1.0 - orig_pred_viz[orig_pred_valid])  # Invert to match GT convention
        orig_pred_colored[~orig_pred_valid, 3] = 0
        orig_pred_rgb = orig_pred_colored[:, :, :3]
        
        proc_pred_colored = np.zeros((*proc_pred_viz.shape, 4), dtype=np.float32)
        proc_pred_colored[proc_pred_valid] = cm_jet(1.0 - proc_pred_viz[proc_pred_valid])  # Invert to match GT convention
        proc_pred_colored[~proc_pred_valid, 3] = 0
        proc_pred_rgb = proc_pred_colored[:, :, :3]
        
        pred_title_suffix = "Disparity Prediction (Near=Red, Far=Blue)"
    else:
        # DepthAnything case: prediction is relative depth, use direct mapping (low=near=red, high=far=blue)
        orig_pred_colored = np.zeros((*orig_pred_viz.shape, 4), dtype=np.float32)
        orig_pred_colored[orig_pred_valid] = cm_jet(orig_pred_viz[orig_pred_valid])  # Direct mapping for depth
        orig_pred_colored[~orig_pred_valid, 3] = 0
        orig_pred_rgb = orig_pred_colored[:, :, :3]
        
        proc_pred_colored = np.zeros((*proc_pred_viz.shape, 4), dtype=np.float32)
        proc_pred_colored[proc_pred_valid] = cm_jet(proc_pred_viz[proc_pred_valid])  # Direct mapping for depth
        proc_pred_colored[~proc_pred_valid, 3] = 0
        proc_pred_rgb = proc_pred_colored[:, :, :3]
        
        pred_title_suffix = "Depth Prediction (Near=Red, Far=Blue)"
    proc_pred_colored[~proc_pred_valid, 3] = 0
    proc_pred_rgb = proc_pred_colored[:, :, :3]  # Keep as float32 in range [0, 1] for imshow
    
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
    plt.title('Ground Truth Disparity (Near=Red, Far=Blue)')
    plt.axis('off')
    
    # Second row: Depth predictions and error maps
    plt.subplot(2, 3, 4)
    plt.imshow(orig_pred_rgb)
    plt.title(f'Original Image {pred_title_suffix}')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(proc_pred_rgb)
    plt.title(f'Processed Image {pred_title_suffix}')
    plt.axis('off')
    
    # Compute error maps
    plt.subplot(2, 3, 6)
    
    # Compute error for both original and processed predictions
    orig_error = np.zeros_like(gt_depth)
    proc_error = np.zeros_like(gt_depth)
    valid_orig = gt_valid & orig_pred_valid
    valid_proc = gt_valid & proc_pred_valid
    
    if pred_is_disparity:
        # Both GT and predictions use inverted mapping, so compute error directly
        if np.any(valid_orig):
            orig_error[valid_orig] = np.abs((1.0 - gt_viz[valid_orig]) - (1.0 - orig_pred_viz[valid_orig]))
        if np.any(valid_proc):
            proc_error[valid_proc] = np.abs((1.0 - gt_viz[valid_proc]) - (1.0 - proc_pred_viz[valid_proc]))
    else:
        # GT uses inverted mapping, predictions use direct mapping
        if np.any(valid_orig):
            orig_error[valid_orig] = np.abs((1.0 - gt_viz[valid_orig]) - orig_pred_viz[valid_orig])
        if np.any(valid_proc):
            proc_error[valid_proc] = np.abs((1.0 - gt_viz[valid_proc]) - proc_pred_viz[valid_proc])
    
    # Create a side-by-side error comparison
    h, w = gt_depth.shape
    combined_error = np.zeros((h, w*2))  # Two error maps side by side
    combined_error[:, :w] = orig_error
    combined_error[:, w:] = proc_error
    
    # Normalize the combined error for visualization
    valid_error = (combined_error > 0)
    if np.any(valid_error):
        max_error = np.max(combined_error[valid_error])
        if max_error > 0:
            combined_error[valid_error] /= max_error
    
    # Apply colormap to combined error
    cmap = plt.colormaps['jet']
    error_rgb = cmap(combined_error)
    error_rgb = error_rgb[:, :, :3]  # Keep as float32 in range [0, 1] for imshow
    
    plt.imshow(error_rgb)
    plt.title('Error Maps: Original (left) vs Processed (right)')
    plt.axis('off')
    
    # Add a vertical line to separate the two error maps
    plt.axvline(x=w-0.5, color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()

class ModelWrapper:
    """Wrapper class to unify different depth estimation models"""
    
    def __init__(self, model_type, device, **kwargs):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.outputs_metric_depth = False  # True for metric depth, False for relative depth/disparity
        
        if model_type == 'depthanything':
            self._init_depthanything(**kwargs)
        elif model_type == 'zoedepth':
            self._init_zoedepth(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_depthanything(self, encoder='vitl', checkpoint_dir='checkpoints', metric=False):
        """Initialize DepthAnything v2 model"""
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.model = DepthAnythingV2(**model_configs[encoder])
        model_path = os.path.join(checkpoint_dir, f"depth_anything_v2_{encoder}.pth")
        
        if metric:
            model_path = os.path.join(checkpoint_dir, f"depth_anything_v2_metric_vkitti_{encoder}.pth")
            self.outputs_metric_depth = True
        
        print(f"Loading DepthAnything v2 model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        
    def _init_zoedepth(self, zoedepth_type='N', checkpoint_dir='checkpoints'):
        """Initialize ZoeDepth model"""
        if zoedepth_type == "N":
            conf = get_config("zoedepth", "infer")
        elif zoedepth_type == "K":
            conf = get_config("zoedepth", "infer", config_version="kitti")
        elif zoedepth_type == "NK":
            conf = get_config("zoedepth_nk", "infer")
        else:
            raise ValueError(f"Unknown ZoeDepth type: {zoedepth_type}")
        
        # Remove pretrained resource to build model without loading weights
        conf.pop("pretrained_resource", None)
        conf["use_pretrained_midas"] = False
        conf["train_midas"] = False
        
        self.model = build_model(conf)
        
        # Load weights
        weights_path = os.path.join(checkpoint_dir, f"ZoeD_M12_{zoedepth_type}.pt")
        print(f"Loading ZoeDepth model from {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5 only
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5 only
        
        # Apply compatibility patch after model is created
        self._patch_zoedepth_model()
        
        self.model = self.model.to(self.device).eval()
        # Based on visualization evidence, ZoeDepth appears to output disparity (near=blue, far=red)
        # rather than metric depth as originally assumed
        self.outputs_metric_depth = False  # ZoeDepth outputs disparity, not metric depth
        
    def _patch_zoedepth_model(self):
        """Apply compatibility patches for ZoeDepth"""
        # Simply patch individual instances to avoid class-level assignment issues
        def patch_blocks(module):
            for child in module.children():
                if hasattr(child, '__class__') and 'Block' in child.__class__.__name__:
                    if not hasattr(child, 'drop_path'):
                        child.drop_path = lambda x: x
                patch_blocks(child)
        
        # Apply instance-level patches
        patch_blocks(self.model)
        print("Applied drop_path compatibility patches to model instances")
    
    def predict(self, image, input_size=518):
        """
        Run inference on an image
        
        Args:
            image: Input image (BGR format for DepthAnything, RGB for ZoeDepth)
            input_size: Input size for inference
            
        Returns:
            Predicted depth map
        """
        with torch.no_grad():
            if self.model_type == 'depthanything':
                return self.model.infer_image(image, input_size)
            elif self.model_type == 'zoedepth':
                # Convert BGR to RGB for ZoeDepth
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                # Create tensor
                tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                tensor = tensor.to(self.device)
                
                return self.model.infer(tensor).cpu().numpy().squeeze()
    
    def postprocess_prediction(self, pred_depth, gt_depth_shape):
        """
        Post-process prediction based on model type
        
        Args:
            pred_depth: Raw prediction from model
            gt_depth_shape: Shape to resize to
            
        Returns:
            processed_depth: Processed depth for metrics calculation
            display_depth: Depth for visualization (original values)
        """
        # Resize to match ground truth
        if pred_depth.shape != gt_depth_shape:
            pred_depth = resize(pred_depth, gt_depth_shape, order=1, preserve_range=True)
        
        if self.outputs_metric_depth:
            # This model outputs metric depth, keep as depth for metrics calculation
            processed_depth = pred_depth  # Keep as metric depth
            display_depth = pred_depth.copy()  # Keep as metric depth for consistent visualization
            
            # For metrics calculation: pred is depth, gt is disparity
            is_pred_disparity = False  # Pred is metric depth, not disparity
        else:
            # Non-metric models: distinguish between DepthAnything (relative depth) and ZoeDepth (disparity)
            if self.model_type == 'zoedepth':
                # ZoeDepth: based on visualization evidence, outputs disparity
                processed_depth = pred_depth  # Keep as disparity
                display_depth = pred_depth.copy()  # Keep as disparity for visualization
                is_pred_disparity = True  # Pred is disparity (same as GT)
            else:
                # DepthAnything: outputs relative depth, not disparity
                processed_depth = pred_depth  # Keep as relative depth
                display_depth = pred_depth.copy()  # Keep as relative depth for visualization
                is_pred_disparity = False  # Pred is relative depth, will be converted to depth in metrics
        
        return processed_depth, display_depth, is_pred_disparity

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
    
    # Initialize model wrapper
    if args.model_type == 'depthanything':
        model = ModelWrapper(
            model_type='depthanything',
            device=device,
            encoder=args.encoder,
            checkpoint_dir=args.checkpoint_dir,
            metric=args.metric
        )
        model_name = f"DepthAnything-v2-{args.encoder}" + ("-metric" if args.metric else "")
    elif args.model_type == 'zoedepth':
        model = ModelWrapper(
            model_type='zoedepth',
            device=device,
            zoedepth_type=args.zoedepth_type,
            checkpoint_dir=args.checkpoint_dir
        )
        model_name = f"ZoeDepth-{args.zoedepth_type}"
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
            
            # Post-process predictions based on model type
            orig_pred_processed, orig_pred_display, is_pred_disparity = model.postprocess_prediction(orig_pred_depth, gt_depth.shape)
            proc_pred_processed, proc_pred_display, _ = model.postprocess_prediction(proc_pred_depth, gt_depth.shape)
            
            # Create combined visualization (using display depth for better visualization)
            visualize_comparison_combined(
                original_image, processed_image, gt_depth, 
                orig_pred_display, proc_pred_display,
                basename, args.output_dir, pred_is_disparity=is_pred_disparity
            )
            
            # Compute metrics: Follow original script logic
            # ZoeDepth: treat output as metric depth, GT as disparity  
            # DepthAnything: treat output as relative depth, GT as disparity
            metrics_orig_sample = compute_depth_metrics(
                orig_pred_processed, gt_depth, 
                is_gt_disparity=True, is_pred_disparity=is_pred_disparity
            )
            metrics_proc_sample = compute_depth_metrics(
                proc_pred_processed, gt_depth, 
                is_gt_disparity=True, is_pred_disparity=is_pred_disparity
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
        print(f"\nModel: {model_name}")
        print(f"Outputs metric depth: {model.outputs_metric_depth}")
        
        # Print metrics for original images
        print("\nOriginal Images - Average Metrics:")
        print(f"Scale Factor: {avg_metrics_orig['scale_factor']:.4f}")
        print(f"RMSE: {avg_metrics_orig['rmse']:.4f}")
        print(f"RMSE-log: {avg_metrics_orig['rmse_log']:.4f}")
        print(f"Abs Rel: {avg_metrics_orig['abs_rel']:.4f}")
        print(f"Sq Rel: {avg_metrics_orig['sq_rel']:.4f}")
        print(f"Delta < 1.25: {avg_metrics_orig['delta1']:.4f}")
        print(f"Delta < 1.25^2: {avg_metrics_orig['delta2']:.4f}")
        print(f"Delta < 1.25^3: {avg_metrics_orig['delta3']:.4f}")
        print(f"Log10: {avg_metrics_orig['log10']:.4f}")
        print(f"SILog: {avg_metrics_orig['silog']:.4f}")
        print(f"Pearson Corr: {avg_metrics_orig['pearson_corr']:.4f}")
        print(f"Spearman Corr: {avg_metrics_orig['spearman_corr']:.4f}")
        
        # Print metrics for processed images
        print("\nProcessed Images (SeaErra) - Average Metrics:")
        print(f"Scale Factor: {avg_metrics_proc['scale_factor']:.4f}")
        print(f"RMSE: {avg_metrics_proc['rmse']:.4f}")
        print(f"RMSE-log: {avg_metrics_proc['rmse_log']:.4f}")
        print(f"Abs Rel: {avg_metrics_proc['abs_rel']:.4f}")
        print(f"Sq Rel: {avg_metrics_proc['sq_rel']:.4f}")
        print(f"Delta < 1.25: {avg_metrics_proc['delta1']:.4f}")
        print(f"Delta < 1.25^2: {avg_metrics_proc['delta2']:.4f}")
        print(f"Delta < 1.25^3: {avg_metrics_proc['delta3']:.4f}")
        print(f"Log10: {avg_metrics_proc['log10']:.4f}")
        print(f"SILog: {avg_metrics_proc['silog']:.4f}")
        print(f"Pearson Corr: {avg_metrics_proc['pearson_corr']:.4f}")
        print(f"Spearman Corr: {avg_metrics_proc['spearman_corr']:.4f}")
        
        # Print comparison
        print("\nMetric Improvement (SeaErra vs Original):")
        for k in metrics_orig.keys():
            improvement = avg_metrics_proc[k] - avg_metrics_orig[k]
            if k in ['delta1', 'delta2', 'delta3', 'pearson_corr', 'spearman_corr']:
                # For these metrics, higher is better
                better = improvement > 0
                print(f"{k}: {improvement:+.4f} ({'better' if better else 'worse'})")
            elif k == 'scale_factor':
                # Scale factor is just a scaling coefficient, not a quality metric
                print(f"{k}: {improvement:+.4f}")
            else:
                # For error metrics, lower is better
                better = improvement < 0
                print(f"{k}: {improvement:+.4f} ({'better' if better else 'worse'})")
        
        # Save overall metrics to a JSON file
        serializable_metrics = {
            'model': model_name,
            'outputs_metric_depth': model.outputs_metric_depth,
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
