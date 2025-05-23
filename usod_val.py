import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import time
import torchvision.transforms as transforms

from usod_dataset import USOD10kDataset
from depth_anything_v2.dpt import DepthAnythingV2


def compute_depth_metrics(pred, gt, mask=None):
    """
    Compute depth evaluation metrics
    
    Args:
        pred: predicted depth
        gt: ground truth depth
        mask: optional mask for valid pixels
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    
    if mask is not None:
        mask = mask.squeeze().cpu().numpy().astype(bool)
        pred = pred[mask]
        gt = gt[mask]
    
    # Create valid value masks - exclude zeros and negative values
    valid_mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    # Apply valid mask
    if valid_mask.sum() == 0:
        # Return zeros if no valid pixels
        return {
            'rmse': 0, 'rmse_log': 0, 'abs_rel': 0, 'sq_rel': 0,
            'delta1': 0, 'delta2': 0, 'delta3': 0, 'log10': 0
        }
        
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    # Add a small epsilon to prevent division by zero
    eps = 1e-6
    
    # Align scale - we're interested in relative depth evaluation
    scale = np.median(gt) / np.median(pred)
    pred_aligned = pred * scale
    
    # Threshold accuracy metrics: δ < 1.25^n
    thresh = np.maximum((gt / (pred_aligned + eps)), ((pred_aligned + eps) / (gt + eps)))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()
    
    # Error metrics
    rmse = np.sqrt(((gt - pred_aligned) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt + eps) - np.log(pred_aligned + eps)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred_aligned) / (gt + eps))
    sq_rel = np.mean(((gt - pred_aligned) ** 2) / (gt + eps))
    
    # Log accuracy
    log10 = np.mean(np.abs(np.log10(gt + eps) - np.log10(pred_aligned + eps)))
    
    return {
        'rmse': rmse,
        'rmse_log': rmse_log,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
        'log10': log10
    }


def validate(model, dataloader, device, args):
    """
    Validate model on dataset
    """
    model.eval()
    
    metrics_sum = {
        'rmse': 0, 'rmse_log': 0, 'abs_rel': 0, 'sq_rel': 0,
        'delta1': 0, 'delta2': 0, 'delta3': 0, 'log10': 0
    }
    
    total_samples = 0
    inference_time = 0
    
    # Create output directory for visualizations if needed
    if args.save_viz:
        os.makedirs(args.outdir, exist_ok=True)
    
    cmap = plt.get_cmap('plasma')
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            # Get input image and ground truth depth
            image = sample['image'].to(device)
            gt_depth = sample['depth']
            img_name = sample['name']
            
            # Measure inference time
            start_time = time.time()
            
            # For batch size of 1, use infer_image, otherwise use model directly
            if image.shape[0] == 1:
                # Convert tensor to numpy for infer_image
                input_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # Denormalize
                input_image = input_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                input_image = (input_image * 255).astype(np.uint8)
                # BGR conversion for OpenCV
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
                
                # Inference
                pred_depth = model.infer_image(input_image, args.input_size)
                pred_depth = torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0).float()
            else:
                # Forward pass (for batches)
                pred_depth = model(image)
                
            inference_time += time.time() - start_time
            
            # Compute metrics
            for i in range(image.shape[0]):
                metrics = compute_depth_metrics(
                    pred_depth[i], 
                    gt_depth[i],
                    mask=None  # You can define a mask if needed
                )
                
                # Save visualization if requested
                if args.save_viz:
                    # Normalize predicted depth for visualization
                    pred_vis = pred_depth[i].squeeze().cpu().numpy()
                    pred_vis = (pred_vis - pred_vis.min()) / (pred_vis.max() - pred_vis.min())
                    
                    # Convert to colormap
                    pred_vis_colored = (cmap(pred_vis)[:, :, :3] * 255).astype(np.uint8)
                    
                    # Ground truth depth visualization
                    gt_vis = gt_depth[i].squeeze().cpu().numpy()
                    gt_vis = (gt_vis - gt_vis.min()) / (gt_vis.max() - gt_vis.min())
                    gt_vis_colored = (cmap(gt_vis)[:, :, :3] * 255).astype(np.uint8)
                    
                    # Original image (denormalize)
                    orig_img = image[i].cpu().numpy().transpose(1, 2, 0)
                    orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    orig_img = (orig_img * 255).astype(np.uint8)
                    
                    # Create comparison visualization
                    # Convert RGB to BGR for cv2
                    viz = np.hstack([
                        cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR), 
                        cv2.cvtColor(gt_vis_colored, cv2.COLOR_RGB2BGR), 
                        cv2.cvtColor(pred_vis_colored, cv2.COLOR_RGB2BGR)
                    ])
                    
                    # Save visualization
                    cv2.imwrite(
                        os.path.join(args.outdir, f"{img_name[i]}_comparison.png"), 
                        viz
                    )
                
                # Update metrics sum
                for k in metrics:
                    metrics_sum[k] += metrics[k]
                total_samples += 1
    
    # Calculate average metrics
    metrics_avg = {k: metrics_sum[k] / total_samples for k in metrics_sum}
    avg_inference_time = inference_time / total_samples
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Validation Results on {total_samples} samples")
    print("-" * 50)
    print(f"RMSE: {metrics_avg['rmse']:.4f}")
    print(f"RMSE log: {metrics_avg['rmse_log']:.4f}")
    print(f"Abs Rel: {metrics_avg['abs_rel']:.4f}")
    print(f"Sq Rel: {metrics_avg['sq_rel']:.4f}")
    print(f"Delta < 1.25: {metrics_avg['delta1']:.4f}")
    print(f"Delta < 1.25^2: {metrics_avg['delta2']:.4f}")
    print(f"Delta < 1.25^3: {metrics_avg['delta3']:.4f}")
    print(f"Log10: {metrics_avg['log10']:.4f}")
    print(f"Average inference time: {avg_inference_time * 1000:.2f}ms")
    print("=" * 50)
    
    return metrics_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Validation on USOD dataset')
    
    parser.add_argument('--data-root', type=str, default='assets/USOD10k', help='Path to USOD10k dataset')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--input-size', type=int, default=518, help='Input resolution for depth prediction')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--outdir', type=str, default='./visualizations/val')
    parser.add_argument('--save-viz', action='store_true', default=True, help='Save visualization of predictions')
    
    args = parser.parse_args()
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Load model
    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    
    # Define transforms for validation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset
    val_dataset = USOD10kDataset(root_dir=args.data_root, split='VAL', transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # Run validation
    metrics = validate(model, val_loader, device, args)
