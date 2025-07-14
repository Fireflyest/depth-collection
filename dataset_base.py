#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Dataset Classes for Depth Estimation Validation

Provides extensible dataset interfaces that can handle various dataset types:
- Datasets with original image + enhanced image + depth (e.g., FLSea with SeaErra)
- Datasets with only original image + depth (e.g., standard depth datasets)
- Datasets with different enhancement methods

Usage:
    from dataset_base import create_dataset
    
    # For FLSea dataset with SeaErra enhancement
    dataset = create_dataset('flsea', data_root='/path/to/flsea')
    
    # For standard datasets without enhancement
    dataset = create_dataset('standard', data_root='/path/to/dataset')
"""

import os
import numpy as np
import cv2
import glob
import rasterio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import warnings
from torch.utils.data import Dataset

# Suppress rasterio warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
except ImportError:
    # Older versions of rasterio
    warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

def load_image(path: str) -> np.ndarray:
    """Load an image and convert to BGR format."""
    try:
        with rasterio.open(path) as src:
            img = src.read()
            if img.shape[0] >= 3:
                # RGB -> BGR for OpenCV compatibility
                img = img[:3].transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[0] == 1:
                # Grayscale -> BGR
                img = np.repeat(img, 3, axis=0).transpose(1, 2, 0)
            return img.astype(np.uint8)
    except Exception as e:
        # Fallback to OpenCV for common formats
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image from {path}: {e}")
        return img

def load_depth(path: str) -> np.ndarray:
    """Load a depth map from file."""
    try:
        with rasterio.open(path) as src:
            depth = src.read(1)  # Read first band
            return depth.astype(np.float32)
    except Exception as e:
        # Try loading as numpy array
        try:
            return np.load(path).astype(np.float32)
        except:
            raise ValueError(f"Cannot load depth from {path}: {e}")

class BaseDepthDataset(Dataset, ABC):
    """
    Abstract base class for depth estimation datasets.
    
    Defines the interface that all depth datasets should implement.
    """
    
    def __init__(self, data_root: str, transform=None):
        """
        Args:
            data_root: Path to dataset root directory
            transform: Optional transform to be applied on samples
        """
        self.data_root = data_root
        self.transform = transform
        self.valid_samples = []
        self._initialize_dataset()
    
    @abstractmethod
    def _initialize_dataset(self):
        """Initialize dataset by finding valid samples. Should populate self.valid_samples."""
        pass
    
    @abstractmethod
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Load a single sample by index. Should return a dictionary with required keys."""
        pass
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._load_sample(idx)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Return a string identifier for the dataset type."""
        pass
    
    @property
    @abstractmethod
    def has_enhancement(self) -> bool:
        """Return True if dataset provides enhanced images."""
        pass
    
    @property
    @abstractmethod
    def enhancement_name(self) -> Optional[str]:
        """Return the name of the enhancement method, or None if no enhancement."""
        pass

class FLSeaDataset(BaseDepthDataset):
    """
    FLSea underwater dataset with SeaErra enhancement.
    
    Expected structure:
    data_root/
        imgs/           # Original images (*.tiff)
        seaErra/        # SeaErra enhanced images (*_SeaErra.tiff)  
        depth/          # Depth maps (*_SeaErra_abs_depth.tif)
    """
    
    def _initialize_dataset(self):
        """Initialize FLSea dataset by finding matching image pairs and depth maps."""
        # Define dataset paths
        self.processed_img_dir = os.path.join(self.data_root, 'seaErra')
        self.orig_img_dir = os.path.join(self.data_root, 'imgs')
        self.depth_dir = os.path.join(self.data_root, 'depth')
        
        # Check if directories exist
        if not os.path.exists(self.orig_img_dir):
            raise ValueError(f"Original images directory not found: {self.orig_img_dir}")
        if not os.path.exists(self.depth_dir):
            raise ValueError(f"Depth directory not found: {self.depth_dir}")
        
        # Use original images as the primary index
        orig_img_files = glob.glob(os.path.join(self.orig_img_dir, '*.tiff'))
        orig_img_files = sorted(orig_img_files)
        
        if not orig_img_files:
            raise ValueError(f"No TIFF images found in {self.orig_img_dir}")
        
        # Find matching processed images and depth maps
        for orig_img_path in orig_img_files:
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
            self.valid_samples.append({
                'original_image_path': orig_img_path,
                'enhanced_image_path': processed_img_path,
                'depth_path': depth_path,
                'basename': basename
            })
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Load a single FLSea sample."""
        sample_info = self.valid_samples[idx]
        
        # Load images and depth
        original_image = load_image(sample_info['original_image_path'])
        enhanced_image = load_image(sample_info['enhanced_image_path'])
        depth = load_depth(sample_info['depth_path'])
        
        return {
            'original_image': original_image,
            'enhanced_image': enhanced_image,
            'depth': depth,
            'enhanced_image_path': sample_info['enhanced_image_path'],
            'basename': sample_info['basename'],
            'dataset_type': self.dataset_type,
            'enhancement_name': self.enhancement_name
        }
    
    @property
    def dataset_type(self) -> str:
        return "flsea"
    
    @property
    def has_enhancement(self) -> bool:
        return True
    
    @property
    def enhancement_name(self) -> str:
        return "SeaErra"

class StandardDepthDataset(BaseDepthDataset):
    """
    Standard depth dataset with only original images and depth maps.
    Alternative hierarchical structure:
        data_root/
            imgs/           # Original images (any format)
            depth/          # Depth maps (various naming patterns)
    """
    
    def _initialize_dataset(self, img_dirname: str = 'imgs', depth_dirname: str = 'depth'):
        """Initialize standard dataset by finding matching images and depth maps."""
        img_dir = os.path.join(self.data_root, img_dirname)
        depth_dir = os.path.join(self.data_root, depth_dirname)
        if os.path.exists(img_dir) and os.path.exists(depth_dir):
            self._init_hierarchical_structure(img_dir, depth_dir)
        if len(self.valid_samples) == 0:
            print(f"[StandardDepthDataset] No valid samples found in hierarchical structure ({img_dir}, {depth_dir}).")


    def _init_hierarchical_structure(self, img_dir: str, depth_dir: str):
        """Initialize dataset with hierarchical directory structure."""
        # Look for common image formats
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        img_files = []
        for pattern in image_patterns:
            img_files.extend(glob.glob(os.path.join(img_dir, pattern)))
        img_files = sorted(img_files)
        print(f"[StandardDepthDataset] Hierarchical structure image files found: {len(img_files)}")
        if not img_files:
            raise ValueError(f"No images found in {img_dir}")
        # Find matching depth maps
        for img_path in img_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # Try different depth file patterns (more comprehensive)
            depth_patterns = [
                f"{basename}.tif",
                f"{basename}_SeaErra_abs_depth.tif",
                f"{basename}.tiff",
            ]
            depth_path = None
            for pattern in depth_patterns:
                candidate_path = os.path.join(depth_dir, pattern)
                if os.path.exists(candidate_path):
                    depth_path = candidate_path
                    break
            if depth_path is None:
                continue  # Skip samples without depth maps
            self.valid_samples.append({
                'original_image_path': img_path,
                'depth_path': depth_path,
                'basename': basename
            })
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Load a single standard dataset sample."""
        sample_info = self.valid_samples[idx]
        
        # Load image and depth
        original_image = load_image(sample_info['original_image_path'])
        depth = load_depth(sample_info['depth_path'])
        
        return {
            'original_image': original_image,
            'enhanced_image': None,  # No enhancement available
            'depth': depth,
            'enhanced_image_path': None,
            'basename': sample_info['basename'],
            'dataset_type': self.dataset_type,
            'enhancement_name': self.enhancement_name
        }
    
    @property
    def dataset_type(self) -> str:
        return "standard"
    
    @property
    def has_enhancement(self) -> bool:
        return False
    
    @property
    def enhancement_name(self) -> Optional[str]:
        return None

# Dataset factory function
def create_dataset(dataset_type: str, data_root: str, **kwargs) -> BaseDepthDataset:
    """
    Factory function to create dataset instances.
    
    Args:
        dataset_type: Type of dataset ('flsea', 'standard')
        data_root: Path to dataset root directory
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Dataset instance
    """
    dataset_registry = {
        'flsea': FLSeaDataset,
        'standard': StandardDepthDataset,
    }
    
    if dataset_type not in dataset_registry:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(dataset_registry.keys())}")
    
    dataset_class = dataset_registry[dataset_type]
    return dataset_class(data_root, **kwargs)

def get_available_datasets() -> Dict[str, str]:
    """
    Get information about available dataset types.
    
    Returns:
        Dictionary mapping dataset type to description
    """
    return {
        'flsea': 'FLSea underwater dataset with SeaErra enhancement',
        'standard': 'Standard depth dataset with original images and depth maps'
    }
