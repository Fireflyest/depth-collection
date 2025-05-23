import os
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class USOD10kDataset(Dataset):
    """
    Dataset loader for USOD10k dataset
    Directory structure:
    assets/USOD10k/
    ├── TE/
    │   ├── Boundary/
    │   ├── depth/
    │   ├── GT/
    │   └── RGB/
    ├── TR/
    │   ├── Boundary/
    │   ├── depth/
    │   ├── GT/
    │   └── RGB/
    └── VAL/
        ├── Boundary/
        ├── depth/
        ├── GT/
        └── RGB/
    """
    def __init__(self, root_dir='assets/USOD10k', split='TR', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'TR' for training, 'VAL' for validation, 'TE' for testing
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        # Get all image filenames
        self.rgb_dir = self.root_dir / self.split / 'RGB'
        self.gt_dir = self.root_dir / self.split / 'GT'
        self.depth_dir = self.root_dir / self.split / 'depth'
        self.boundary_dir = self.root_dir / self.split / 'Boundary'
        
        # Get image file names
        self.image_names = [f for f in os.listdir(self.rgb_dir) if not f.startswith('.')]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.image_names[idx]
        
        # Load images
        rgb_path = os.path.join(self.rgb_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        
        # OpenCV loads images in BGR format, convert to RGB
        image = cv2.imread(str(rgb_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Handle different file extensions if necessary for GT, depth and boundary
        gt_extensions = ['.png', '.jpg', '.jpeg']
        depth_extensions = ['.png', '.jpg', '.jpeg']
        boundary_extensions = ['.png', '.jpg', '.jpeg']
        
        gt_image = None
        for ext in gt_extensions:
            potential_path = os.path.join(self.gt_dir, os.path.splitext(img_name)[0] + ext)
            if os.path.exists(potential_path):
                gt_image = cv2.imread(str(potential_path), cv2.IMREAD_GRAYSCALE)
                break
        
        depth_image = None
        for ext in depth_extensions:
            potential_path = os.path.join(self.depth_dir, os.path.splitext(img_name)[0] + ext)
            if os.path.exists(potential_path):
                depth_image = cv2.imread(str(potential_path), cv2.IMREAD_GRAYSCALE)
                break
        
        boundary_image = None
        for ext in boundary_extensions:
            potential_path = os.path.join(self.boundary_dir, os.path.splitext(img_name)[0] + "_edge" + ext)
            if os.path.exists(potential_path):
                boundary_image = cv2.imread(str(potential_path), cv2.IMREAD_GRAYSCALE)
                break
        
        if gt_image is None or depth_image is None or boundary_image is None:
            raise FileNotFoundError(f"Could not find corresponding files for {img_name}")
        
        # Apply transforms
        if self.transform:
            # Convert numpy arrays to tensors
            image = self.transform(image)
            
            # For GT, depth, and boundary, we need to handle numpy arrays
            gt_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            # Convert numpy arrays to tensors
            gt_image = gt_transform(gt_image)
            depth_image = gt_transform(depth_image)
            boundary_image = gt_transform(boundary_image)
        
        sample = {
            'image': image,
            'gt': gt_image,
            'depth': depth_image,
            'boundary': boundary_image,
            'name': os.path.splitext(img_name)[0]
        }
        
        return sample


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset instances
    val_dataset = USOD10kDataset(root_dir='assets/USOD10k', split='VAL', transform=transform)
    
    # Create data loaders
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Print dataset sizes
    print(f"Validation set size: {len(val_dataset)}")
    
    # Display a sample
    sample = val_dataset[2]
    print(f"Sample name: {sample['name']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"GT shape: {sample['gt'].shape}")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Boundary shape: {sample['boundary'].shape}")

    # Save the figure instead of showing it
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    # Visualize a validation sample
    if len(val_dataset) > 0:
        print("\nVisualizing validation sample:")
        # Denormalize the image
        img = sample['image'].clone()
        img = img * torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.clamp(0, 1)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Display RGB image
        axes[0].imshow(img.permute(1, 2, 0).numpy())
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
        
        # Display GT segmentation
        axes[1].imshow(sample['gt'][0].numpy(), cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Display Depth map
        axes[2].imshow(sample['depth'][0].numpy(), cmap='gray')
        axes[2].set_title('Depth Map')
        axes[2].axis('off')
        
        # Display Boundary map
        axes[3].imshow(sample['boundary'][0].numpy(), cmap='gray')
        axes[3].set_title('Boundary Map')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{sample["name"]}.png')
        plt.close()

