#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Model Interface for Depth Estimation
Provides a consistent interface for different depth estimation models with automatic
output characteristic handling and extensible model registration.

Supported models:
- DepthAnything v2 (metric and non-metric variants)
- ZoeDepth (N, K, NK variants)
- VGGT
- Metric3D

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
import sys

# Import models
from depth_anything_v2.dpt import DepthAnythingV2
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# Add VGGT imports
sys.path.append("vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Add ViT Large model imports
try:
    from mmcv.utils import Config
except:
    from mmengine import Config
from mono.model.monodepth_model import get_configured_monodepth_model


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
        elif self.output_unit == 'affine-invariant':
            # Case: affine-invariant depth (relative depth requiring scale alignment)
            # No conversion applied here - scale alignment will be done during metrics computation
            converted_data = data.copy()
            conversion_info['applied'] = False
            conversion_info['note'] = 'Affine-invariant depth - scale alignment required during evaluation'
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
        elif self.model_type == 'vggt':
            self._init_vggt(**kwargs)
        elif self.model_type == 'metric3d':
            self._init_metric3d(**kwargs)
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
            # Non-metric DepthAnything outputs disparity that converts to affine-invariant depth
            self.output_characteristics = ModelOutputCharacteristics(
                is_disparity=True,
                is_metric=False,
                output_unit='affine-invariant',
                target_unit='meters',
                needs_inversion=True,
                scale_factor=1.0,  # No scale conversion needed
                display_name="Relative Disparity (affine-invariant depth)"
            )
        
        print(f"Loading DepthAnything v2 model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
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
    
    def _init_vggt(self, checkpoint_dir='checkpoints', multi_image_mode=False):
        """Initialize VGGT model"""
        self.model = VGGT()
        self.multi_image_mode = multi_image_mode  # Store multi-image mode setting
        
        # Load weights
        weights_path = os.path.join(checkpoint_dir, "vggt.pt")
        print(f"Loading VGGT model from {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        
        # VGGT outputs affine-invariant depth (relative depth requiring scale alignment)
        self.output_characteristics = ModelOutputCharacteristics(
            is_disparity=False,
            is_metric=False,  # Affine-invariant depth is not metric (needs scale alignment)
            output_unit='affine-invariant',
            target_unit='meters',
            needs_inversion=False,
            scale_factor=1.0,  # Scale will be determined by alignment with GT
            display_name="Affine-Invariant Depth"
        )
        
        if multi_image_mode:
            print("VGGT initialized in multi-image mode for enhanced geometric consistency")
    
    def _init_metric3d(self, checkpoint_dir='checkpoints'):
        """Initialize Metric3D model"""
        # Set configuration file path
        cfg_file = 'mono/configs/HourglassDecoder/vit.raft5.large.py'
        
        # Load configuration
        cfg = Config.fromfile(cfg_file)
        # Convert Config object to dictionary format
        if hasattr(cfg, '_cfg_dict'):
            cfg_dict = cfg._cfg_dict
        else:
            cfg_dict = dict(cfg)
        
        # Create model
        self.model = get_configured_monodepth_model(cfg_dict)
        
        # Load weights
        weights_path = os.path.join(checkpoint_dir, "metric_depth_vit_large_800k.pth")
        print(f"Loading Metric3D model from {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Metric3D outputs metric depth in meters
        self.output_characteristics = ModelOutputCharacteristics(
            is_disparity=False,
            is_metric=True,
            output_unit='meters',
            target_unit='meters',
            needs_inversion=False,
            scale_factor=1.0,
            display_name="Metric Depth (meters)"
        )
    
    def predict(self, image, input_size=518):
        """
        Run inference on an image or multiple images
        
        Args:
            image: Input image/images. For VGGT multi-image mode, can be:
                   - Single image (BGR format for DepthAnything, RGB for ZoeDepth, path/array for VGGT)
                   - List of images for VGGT multi-image mode
                   - Directory path containing images for VGGT multi-image mode
            input_size: Input size for inference
            
        Returns:
            Predicted depth map(s)
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
            elif self.model_type == 'vggt':
                return self._predict_vggt(image, input_size)
            elif self.model_type == 'metric3d':
                return self._predict_metric3d(image)
    
    def _predict_vggt(self, image, input_size):
        """VGGT-specific prediction with single and multi-image support"""
        import tempfile
        import glob
        
        temp_files = []  # Initialize temp_files list
        
        # Handle different input types for VGGT
        if hasattr(self, 'multi_image_mode') and self.multi_image_mode:
            # Multi-image mode
            if isinstance(image, str):
                if os.path.isdir(image):
                    # Directory path provided
                    image_paths = sorted(glob.glob(os.path.join(image, "*")))
                    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                    if len(image_paths) == 0:
                        raise ValueError(f"No valid images found in directory: {image}")
                else:
                    # Single image path
                    image_paths = [image]
            elif isinstance(image, (list, tuple)):
                # List of images (paths or arrays)
                image_paths = []
                
                for idx, img in enumerate(image):
                    if isinstance(img, str):
                        image_paths.append(img)
                    else:
                        # Image array - save temporarily
                        tmp_file = tempfile.NamedTemporaryFile(suffix=f'_{idx}.jpg', delete=False)
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        else:
                            img_rgb = img
                        cv2.imwrite(tmp_file.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                        image_paths.append(tmp_file.name)
                        temp_files.append(tmp_file.name)
                        tmp_file.close()
            else:
                # Single image array - convert to single image mode
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                cv2.imwrite(temp_file.name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                image_paths = [temp_file.name]
                temp_files = [temp_file.name]
                temp_file.close()
            
            print(f"VGGT multi-image mode: processing {len(image_paths)} images")
            
        else:
            # Single image mode (backward compatibility)
            if isinstance(image, str):
                image_paths = [image]
            else:
                # Image array provided - save temporarily
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                cv2.imwrite(temp_file.name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                image_paths = [temp_file.name]
                temp_files = [temp_file.name]
                temp_file.close()
        
        try:
            # Preprocess images
            images_tensor = load_and_preprocess_images(image_paths)
            images_tensor = images_tensor.to(self.device)
            
            # Run inference
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images_tensor)
            
            # Extract depth - VGGT returns a dict with 'depth' key
            depth = predictions["depth"].cpu().numpy()
            
            # Handle output format based on number of images
            if hasattr(self, 'multi_image_mode') and self.multi_image_mode and len(image_paths) > 1:
                # Multi-image mode: return all depth maps and additional info
                result = {
                    'depth_maps': depth,  # Shape: (N, H, W) or (N, H, W, 1)
                    'num_images': len(image_paths),
                    'extrinsic': None,
                    'intrinsic': None,
                    'image_paths': image_paths
                }
                
                # Add camera parameters if available
                if "pose_enc" in predictions:
                    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
                    if extrinsic is not None:
                        result['extrinsic'] = extrinsic.cpu().numpy()
                    if intrinsic is not None:
                        result['intrinsic'] = intrinsic.cpu().numpy()
                
                return result
            else:
                # Single image mode: return single depth map
                return depth.squeeze()
                
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def _predict_metric3d(self, image):
        """Metric3D-specific prediction with proper preprocessing"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        
        # ViT model input size
        input_size = (616, 1064)
        h, w = rgb.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        
        # Padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, 
                               pad_w_half, pad_w - pad_w_half, 
                               cv2.BORDER_CONSTANT, value=padding)
        
        # Normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :]  # add batch dimension
        
        # Move to device
        rgb = rgb.to(self.device)
        mean = mean.to(self.device)
        std = std.to(self.device)
        
        # Model inference
        pred_depth, confidence, output_dict = self.model.inference({'input': rgb})
        
        # Return depth on CPU as numpy array
        return pred_depth.squeeze().cpu().numpy()
    
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
        model_type: Type of model to create ('depthanything', 'zoedepth', 'vggt', 'metric3d')
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
        },
        'vggt': {
            'description': 'VGGT - Affine-invariant depth estimation',
            'variants': ['single-image', 'multi-image'],
            'supports_metric': False,  # Affine-invariant depth requires scale alignment
            'required_params': [],
            'optional_params': ['checkpoint_dir', 'multi_image_mode']
        },
        'metric3d': {
            'description': 'Metric3D - Metric depth estimation',
            'variants': ['large'],
            'supports_metric': True,
            'required_params': [],
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
    elif model_type == 'vggt':
        multi_mode = " (Multi-Image)" if kwargs.get('multi_image_mode', False) else ""
        return f"VGGT{multi_mode}"
    elif model_type == 'metric3d':
        return "Metric3D"
    else:
        return model_type

def print_models_structures():
    """
    Print the structures of all available models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_infos = get_available_models()
    out_dir = './out'
    os.makedirs(out_dir, exist_ok=True)
    for model_type, info in model_infos.items():
        print(f'\n=== {model_type} ===')
        # 构造参数
        kwargs = {}
        for param in info.get('required_params', []):
            # 默认参数
            if param == 'encoder':
                kwargs['encoder'] = info['variants'][0]
            elif param == 'zoedepth_type':
                kwargs['zoedepth_type'] = info['variants'][0]
        # 可选参数
        for param in info.get('optional_params', []):
            if param == 'metric':
                kwargs['metric'] = False
            if param == 'multi_image_mode':
                kwargs['multi_image_mode'] = False
            if param == 'checkpoint_dir':
                kwargs['checkpoint_dir'] = 'checkpoints'
        try:
            model = create_model(model_type, device, **kwargs)
            structure_str = str(model.model)
            out_path = os.path.join(out_dir, f'{model_type}.txt')
            with open(out_path, 'w') as f:
                f.write(structure_str)
            print(f'Saved structure to {out_path}')
        except Exception as e:
            print(f'Error creating {model_type}: {e}')

if __name__ == '__main__':
    print_models_structures()