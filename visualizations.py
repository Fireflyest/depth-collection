#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Model Depth Estimation Comparison Visualization
Runs multiple depth estimation models and creates a comprehensive comparison visualization.

Usage:
    python visualizations.py --data-root assets/FLSea/red_sea/pier_path --num-samples 5
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from typing import List, Dict, Tuple, Optional
import random
from io import BytesIO
import PIL.Image

# Import model wrapper and characteristics
from models import create_model
from dataset_base import create_dataset
from depth_utils import compute_depth_metrics, add_depth_markers, prepare_depths_for_visualization

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class MultiModelComparison:
    """Class to handle multi-model depth estimation comparison"""
    
    def __init__(self, data_root: str, num_samples: int = 5, random_seed: int = 42, device: str = "cuda"):
        self.data_root = data_root
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        # Set random seed for reproducible results
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        
        # Model configurations
        self.model_configs = {
            'DepthAnything-v2': {
                'type': 'depthanything',
                'encoder': 'vitl',
                'metric': False
            },
            'ZoeDepth-N': {
                'type': 'zoedepth',
                'zoedepth_type': 'N'
            },
            'VGGT': {
                'type': 'vggt',
                'multi_image_mode': False
            },
            'Metric3D': {
                'type': 'metric3d'
            }
        }
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Load dataset
        self.dataset = create_dataset('standard', data_root)
        if len(self.dataset) == 0:
            raise ValueError("No valid samples found in the dataset")
        
        print(f"Found {len(self.dataset)} valid samples in the dataset")
        
        # Select samples
        self.sample_indices = self._select_samples()
        
    def _initialize_models(self):
        """Initialize all depth estimation models"""
        print("Initializing models...")
        
        for model_name, config in self.model_configs.items():
            print(f"Loading {model_name}...")
            model = create_model(
                model_type=config['type'],
                device=self.device,
                **{k: v for k, v in config.items() if k != 'type'}
            )
            self.models[model_name] = model
            print(f"{model_name} loaded successfully")
    
    def _select_samples(self) -> List[int]:
        """Select sample indices for comparison"""
        available_samples = min(self.num_samples, len(self.dataset))
        
        # Use consistent random selection
        np.random.seed(self.random_seed)
        indices = np.random.choice(len(self.dataset), size=available_samples, replace=False)
        return sorted(indices.tolist())
    
    def run_inference(self) -> Dict:
        """Run inference on all models for selected samples"""
        results = {
            'samples': [],
            'predictions': {model_name: [] for model_name in self.models.keys()}
        }
        
        print(f"Running inference on {len(self.sample_indices)} samples...")
        
        for idx in tqdm(self.sample_indices, desc="Processing samples"):
            # Load sample
            sample = self.dataset[idx]
            original_image = sample['original_image']
            gt_depth = sample['depth']
            basename = sample['basename']
            
            if original_image is None or gt_depth is None:
                print(f"Invalid sample at index {idx}")
                continue
            
            # Store sample data
            sample_data = {
                'index': idx,
                'basename': basename,
                'original_image': original_image,
                'gt_depth': gt_depth
            }
            results['samples'].append(sample_data)
            
            # Run inference for each model
            for model_name, model in self.models.items():
                # Run inference
                pred_depth = model.predict(original_image, input_size=518)
                
                # Post-process prediction
                processed_depth, display_depth, conversion_info = model.postprocess_prediction(
                    pred_depth, gt_depth.shape
                )
                
                # Store prediction
                pred_data = {
                    'raw_prediction': pred_depth,
                    'processed_depth': processed_depth,
                    'display_depth': display_depth,
                    'conversion_info': conversion_info,
                    'model_characteristics': model.output_characteristics
                }
                results['predictions'][model_name].append(pred_data)
        
        return results
    
    def create_comparison_visualization(self, results: Dict, output_path: str = "model_comparison.png", depth_markers: bool = True,
                                      max_rows_in_final: Optional[int] = None):
        """Create comprehensive comparison visualization with 5 columns layout"""
        samples = results['samples']
        predictions = results['predictions']
        
        if len(samples) == 0:
            print("No valid samples to visualize")
            return
        
        n_samples = len(samples)
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        # Create figure: columns = [Original, GT, Model1, Model2, Model3]
        n_cols = 2 + n_models  # Original + GT + models
        col_headers = ['Original Image', 'Ground Truth'] + model_names
        
        print(f"Creating visualization with {n_samples} samples and {n_models} models...")
        
        # Create output directory for individual rows
        output_dir = os.path.dirname(output_path)
        rows_dir = os.path.join(output_dir, "individual_rows")
        os.makedirs(rows_dir, exist_ok=True)
        
        # Store row images for final concatenation
        row_images = []
        
        for row in range(n_samples):
            sample = samples[row]
            original_image = sample['original_image']
            gt_depth = sample['gt_depth']
            basename = sample['basename']
            
            # Prepare depth visualizations
            gt_viz, gt_valid_mask, display_depths, display_vizs, aligned_valid_masks, aligned_depths, range_texts = prepare_depths_for_visualization(
                gt_depth, predictions, model_names, row
            )
            
            # Create single row figure
            fig_width = n_cols * 4
            fig_height = 4.5 if row == 0 else 4  # First row slightly taller for titles
            
            fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))
            if n_cols == 1:
                axes = [axes]
            
            # Set tight layout with no spacing
            plt.subplots_adjust(left=0, right=1, top=0.85 if row == 0 else 1, bottom=0, wspace=0)
            
            for col in range(n_cols):
                ax = axes[col]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.axis('off')
                
                if col == 0:  # Original image
                    img_rgb = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    if row == 0:
                        ax.set_title(col_headers[col], fontsize=14, fontweight='bold')
                    ax.set_ylabel(f"Sample {row+1}\n{basename}", fontsize=10, rotation=0, ha='right', va='center')
                    
                elif col == 1:  # Ground truth
                    ax.imshow(gt_viz)
                    if row == 0:
                        ax.set_title(col_headers[col], fontsize=14, fontweight='bold')
                
                    ax.text(0.02, 0.98, range_texts['gt'], 
                            transform=ax.transAxes, fontsize=8, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            verticalalignment='top')
                    
                    # Add depth scale markers for GT
                    if depth_markers:
                        add_depth_markers(ax, gt_valid_mask, gt_depth, None, True)
                
                else:  # Model predictions
                    model_idx = col - 2
                    model_name = model_names[model_idx]
                    
                    if (row < len(predictions[model_name]) and predictions[model_name][row] is not None and model_name in display_vizs):
                        pred_data = predictions[model_name][row]
                        model_chars = pred_data['model_characteristics']
                        ax.imshow(display_vizs[model_name])
                        if row == 0:
                            title = f"{model_name}\n({model_chars.display_name})"
                            ax.set_title(title, fontsize=12, fontweight='bold')

                        ax.text(0.02, 0.98, range_texts[model_name], 
                            transform=ax.transAxes, fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                            verticalalignment='top')

                        # Add depth scale markers for model predictions
                        if depth_markers:
                            add_depth_markers(ax, aligned_valid_masks[model_name],
                                            display_depths[model_name], aligned_depths[model_name], model_chars.is_metric)

                    else:
                        # No prediction available
                        ax.text(0.5, 0.5, 'No Prediction', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=12, color='red')
                        if row == 0:
                            ax.set_title(model_names[model_idx], fontsize=12, fontweight='bold')
            
            # Save individual row
            row_filename = f"row_{row+1:02d}_{basename}.png"
            row_path = os.path.join(rows_dir, row_filename)
            plt.savefig(row_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"  Saved row {row+1}: {row_filename}")
            
            # Save row to memory as image array for concatenation
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            
            # Read back as image array
            row_img = PIL.Image.open(buf)
            row_images.append(np.array(row_img))
            
            plt.close()
            buf.close()
        
        # Determine how many rows to include in final image
        if max_rows_in_final is None:
            max_rows_in_final = min(5, n_samples)  # Default to 5 or less
        
        rows_to_include = min(max_rows_in_final, len(row_images))
        
        # Concatenate selected rows vertically
        print(f"Concatenating first {rows_to_include} rows into final image...")
        if rows_to_include > 0:
            final_image = np.vstack(row_images[:rows_to_include])
            
            # Save final concatenated image
            final_pil = PIL.Image.fromarray(final_image)
            final_pil.save(output_path, dpi=(300, 300))
            
            print(f"✅ Final comparison visualization saved to: {output_path}")
            print(f"📁 Individual rows saved to: {rows_dir}")
            print(f"📊 Final image contains {rows_to_include} rows out of {n_samples} total samples")
        else:
            print("❌ No rows to concatenate")

    def compute_and_print_metrics(self, results: Dict):
        """Compute and print comparison metrics between models"""
        samples = results['samples']
        predictions = results['predictions']
        
        metrics_summary = {}
        
        print("\n" + "="*80)
        print("COMPUTING MODEL METRICS...")
        print("="*80)
        
        for model_name in self.models.keys():
            model_metrics = []
            
            for i, sample in enumerate(samples):
                if i >= len(predictions[model_name]) or predictions[model_name][i] is None:
                    continue
                
                gt_depth = sample['gt_depth']
                pred_data = predictions[model_name][i]
                processed_depth = pred_data['processed_depth']
                
                metrics = compute_depth_metrics(
                    processed_depth, gt_depth,
                    is_gt_disparity=False, is_pred_disparity=False
                )
                model_metrics.append(metrics)
            
            # Calculate average metrics
            if model_metrics:
                avg_metrics = {}
                for key in ['rmse', 'abs_rel', 'delta1', 'delta2', 'delta3', 'silog']:
                    values = [m[key] for m in model_metrics if key in m]
                    avg_metrics[key] = np.mean(values) if values else 0.0
                metrics_summary[model_name] = avg_metrics
            else:
                print(f"No valid metrics for {model_name}")
        
        # Print formatted metrics table
        if metrics_summary:
            print("\n" + "="*90)
            print("MODEL COMPARISON METRICS")
            print("="*90)
            
            # Header
            models = list(metrics_summary.keys())
            header = f"{'Metric':<15}"
            for model in models:
                header += f"{model:<20}"
            print(header)
            print("-"*90)
            
            # Metrics rows
            metric_info = [
                ('RMSE', 'rmse'),
                ('Abs Rel', 'abs_rel'), 
                ('Delta < 1.25', 'delta1'),
                ('Delta < 1.25^2', 'delta2'),
                ('Delta < 1.25^3', 'delta3'),
                ('SILog', 'silog')
            ]
            
            for name, key in metric_info:
                row = f"{name:<15}"
                for model in models:
                    value = metrics_summary[model].get(key, 0.0)
                    row += f"{value:<20.4f}"
                print(row)
            
            print("="*90)
        
        return metrics_summary

def main():
    parser = argparse.ArgumentParser(description='Multi-Model Depth Estimation Comparison')
    parser.add_argument('--data-root', type=str, default='assets/FLSea/red_sea/pier_path',
                       help='Path to dataset')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to compare')
    parser.add_argument('--max-rows-final', type=int, default=4,
                       help='Maximum rows to include in final concatenated image')
    parser.add_argument('--output-dir', type=str, default='visualizations/comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible sample selection')
    parser.add_argument('--depth_markers', action='store_true',
                       help='Whether to include depth markers in the visualization')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize comparison
    print("🚀 Starting Multi-Model Depth Estimation Comparison")
    print("="*60)
    
    comparison = MultiModelComparison(
        data_root=args.data_root,
        num_samples=args.num_samples,
        random_seed=args.random_seed
    )
    
    # Run inference
    print("\n📊 Running inference on all models...")
    results = comparison.run_inference()
    
    # Create visualization
    print("\n🎨 Creating comparison visualization...")
    output_path = os.path.join(args.output_dir, "model_comparison.png")
    comparison.create_comparison_visualization(results, output_path, args.depth_markers, args.max_rows_final)

    # Compute and print metrics
    print("\n📈 Computing evaluation metrics...")
    metrics = comparison.compute_and_print_metrics(results)
    
    print(f"\n✅ Comparison completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")
    


if __name__ == '__main__':
    main()
