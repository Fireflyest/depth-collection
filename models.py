#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Model Interface for Depth Estimation
Provides a consistent interface for different depth estimation models with automatic
output characteristic handling and extensible model registration.

Supported models:
- DepthAnything v2 (metric and non-metric variants)
- ZoeDepth (N, K, NK variants)

Usage:
    # Create model wrapper
    model = ModelWrapper(
        model_type='depthanything',
        device=device,
        encoder='vitl',
        metric=False
    )
    
    # Run inference
    pred_depth = model.predict(image)
    
    # Post-process with automatic unit conversion
    processed_depth, display_depth, conversion_info = model.postprocess_prediction(pred_depth, gt_shape)
"""

import os
import torch
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from skimage.transform import resize

# Import models
from depth_anything_v2.dpt import DepthAnythingV2
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

@dataclass
class ModelOutputCharacteristics:
    """
    Describes the characteristics of a model's output
    """
    # Output type
    is_disparity: bool  # True if output is disparity, False if depth
    is_metric: bool     # True if output has metric scale, False if relative
    
    # Unit information
    output_unit: str    # Original output unit (e.g., 'disparity', 'meters', 'millimeters')
    target_unit: str    # Target unit for processing (typically 'meters')
    
    # Conversion parameters
    needs_inversion: bool = False  # True if disparity->depth conversion needed
    scale_factor: float = 1.0      # Multiplication factor for unit conversion
    
    # Display information
    display_name: str = ""         # Human readable description
    
    def __post_init__(self):
        if not self.display_name:
            if self.is_disparity:
                if self.is_metric:
                    self.display_name = f"Metric Disparity ({self.output_unit})"
                else:
                    self.display_name = f"Relative Disparity ({self.output_unit})"
            else:
                if self.is_metric:
                    self.display_name = f"Metric Depth ({self.output_unit})"
                else:
                    self.display_name = f"Relative Depth ({self.output_unit})"
    
    def convert_to_target_unit(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Convert model output to target unit
        
        Returns:
            converted_data: Data in target unit
            conversion_info: Information about the conversion applied
        """
        eps = 1e-6
        conversion_info = {
            'applied': False,
            'factor': self.scale_factor,
            'original_unit': self.output_unit,
            'target_unit': self.target_unit
        }
        
        if self.needs_inversion and self.scale_factor != 1.0:
            # Case: disparity -> depth with unit conversion
            # Example: DepthAnything disparity -> km -> meters
            depth_intermediate = 1.0 / (data + eps)  # disparity -> intermediate depth
            converted_data = depth_intermediate * self.scale_factor  # apply scale
            conversion_info['applied'] = True
        elif self.needs_inversion:
            # Case: disparity -> depth without unit conversion
            # Example: ZoeDepth relative disparity -> relative depth
            converted_data = 1.0 / (data + eps)
        elif self.scale_factor != 1.0:
            # Case: unit conversion without inversion
            # Example: millimeters -> meters
            converted_data = data * self.scale_factor
            conversion_info['applied'] = True
        else:
            # Case: no conversion needed
            # Example: ZoeDepth metric depth in meters
            converted_data = data.copy()
        
        return converted_data, conversion_info
    
    def get_original_value_from_converted(self, converted_value: float) -> float:
        """
        Get original model output value from converted value (for visualization)
        """
        eps = 1e-6
        
        if self.needs_inversion and self.scale_factor != 1.0:
            # converted_value = (1/original) * scale_factor
            # original = 1 / (converted_value / scale_factor)
            intermediate = converted_value / self.scale_factor
            return 1.0 / (intermediate + eps)
        elif self.needs_inversion:
            # converted_value = 1/original
            return 1.0 / (converted_value + eps)
        elif self.scale_factor != 1.0:
            # converted_value = original * scale_factor
            return converted_value / self.scale_factor
        else:
            return converted_value


class ModelWrapper:
    """Wrapper class to unify different depth estimation models"""
    
    def __init__(self, model_type, device, **kwargs):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.output_characteristics: Optional[ModelOutputCharacteristics] = None
        
        # Register and initialize the model
        self._initialize_model(**kwargs)
        
        # Ensure output_characteristics is set
        if self.output_characteristics is None:
            raise ValueError(f"Model initialization failed: output_characteristics not set for {model_type}")
    
    def _initialize_model(self, **kwargs):
        """Initialize model based on type using registry pattern"""
        if self.model_type == 'depthanything':
            self._init_depthanything(**kwargs)
        elif self.model_type == 'zoedepth':
            self._init_zoedepth(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
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
            # Metric DepthAnything outputs metric depth
            self.output_characteristics = ModelOutputCharacteristics(
                is_disparity=False,
                is_metric=True,
                output_unit='meters',
                target_unit='meters',
                needs_inversion=False,
                scale_factor=1.0,
                display_name="Metric Depth (meters)"
            )
        else:
            # Non-metric DepthAnything outputs disparity that converts to depth in kilometers
            self.output_characteristics = ModelOutputCharacteristics(
                is_disparity=True,
                is_metric=False,
                output_unit='disparity->km',
                target_unit='meters',
                needs_inversion=True,
                scale_factor=1000.0,  # km -> meters
                display_name="Relative Disparity (converts to km, then to m)"
            )
        
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
        
        # ZoeDepth outputs metric depth in meters
        self.output_characteristics = ModelOutputCharacteristics(
            is_disparity=False,
            is_metric=True,
            output_unit='meters',
            target_unit='meters',
            needs_inversion=False,
            scale_factor=1.0,
            display_name="Metric Depth (meters)"
        )
        
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
        Post-process prediction based on model output characteristics
        
        Args:
            pred_depth: Raw prediction from model
            gt_depth_shape: Shape to resize to
            
        Returns:
            processed_depth: Processed depth for metrics calculation
            display_depth: Depth for visualization
            conversion_info: Information about unit conversion applied
        """
        # Resize to match ground truth
        if pred_depth.shape != gt_depth_shape:
            pred_depth = resize(pred_depth, gt_depth_shape, order=1, preserve_range=True)
        
        # Ensure output_characteristics is available
        assert self.output_characteristics is not None, "Model output characteristics not initialized"
        
        # Convert using model characteristics
        processed_depth, conversion_info = self.output_characteristics.convert_to_target_unit(pred_depth)
        display_depth = processed_depth.copy()  # Use converted depth for consistent visualization
        
        return processed_depth, display_depth, conversion_info


def create_model(model_type: str, device: torch.device, **kwargs) -> ModelWrapper:
    """
    Factory function to create model instances
    
    Args:
        model_type: Type of model to create ('depthanything', 'zoedepth')
        device: PyTorch device to use
        **kwargs: Model-specific parameters
        
    Returns:
        ModelWrapper instance
    """
    return ModelWrapper(model_type, device, **kwargs)


def get_available_models() -> dict:
    """
    Get information about available models
    
    Returns:
        Dictionary containing model information
    """
    return {
        'depthanything': {
            'description': 'DepthAnything v2 - Relative/Metric depth estimation',
            'variants': ['vits', 'vitb', 'vitl', 'vitg'],
            'supports_metric': True,
            'required_params': ['encoder'],
            'optional_params': ['metric', 'checkpoint_dir']
        },
        'zoedepth': {
            'description': 'ZoeDepth - Metric depth estimation',
            'variants': ['N', 'K', 'NK'],
            'supports_metric': True,
            'required_params': ['zoedepth_type'],
            'optional_params': ['checkpoint_dir']
        }
    }


def get_model_name(model_type: str, **kwargs) -> str:
    """
    Generate a human-readable model name based on type and parameters
    
    Args:
        model_type: Type of model
        **kwargs: Model parameters
        
    Returns:
        Human-readable model name
    """
    if model_type == 'depthanything':
        encoder = kwargs.get('encoder', 'vitl')
        metric = kwargs.get('metric', False)
        return f"DepthAnything-v2-{encoder}" + ("-metric" if metric else "")
    elif model_type == 'zoedepth':
        zoedepth_type = kwargs.get('zoedepth_type', 'N')
        return f"ZoeDepth-{zoedepth_type}"
    else:
        return model_type
