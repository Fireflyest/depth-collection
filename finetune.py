import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import create_model
from dataset_base import create_dataset
import numpy as np
from tqdm import tqdm
import json

def get_decoder_encoder(model, model_type):
    """
    根据模型类型和结构自动获取 decoder 和 encoder
    """
    if model_type == 'depthanything':
        decoder = model.model.depth_head
        encoder = model.model.pretrained
    elif model_type == 'zoedepth':
        # 兼容不同 ZoeDepth 结构
        try:
            decoder = model.model.core.core.scratch
            encoder = model.model.core.core.pretrained
        except AttributeError:
            decoder = model.model.core.scratch
            encoder = model.model.core.pretrained
    elif model_type == 'vggt':
        # 以 aggregator 为主
        decoder = getattr(model.model.aggregator, 'decoder', None)
        encoder = getattr(model.model.aggregator, 'patch_embed', None)
        if decoder is None or encoder is None:
            raise AttributeError('VGGT aggregator结构未找到decoder/patch_embed')
    elif model_type == 'metric3d':
        decoder = model.model.depth_model.decoder
        encoder = model.model.depth_model.encoder
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return decoder, encoder

def get_model_config(args):
    """
    参考 visualizations.py，自动生成模型参数配置
    """
    if args.model_type == 'depthanything':
        config = {
            'model_type': 'depthanything',
            'encoder': args.model_kwargs.get('encoder', 'vitl'),
            'metric': args.model_kwargs.get('metric', False),
            'checkpoint_dir': args.model_kwargs.get('checkpoint_dir', 'checkpoints')
        }
    elif args.model_type == 'zoedepth':
        config = {
            'model_type': 'zoedepth',
            'zoedepth_type': args.model_kwargs.get('zoedepth_type', 'N'),
            'checkpoint_dir': args.model_kwargs.get('checkpoint_dir', 'checkpoints')
        }
    elif args.model_type == 'vggt':
        config = {
            'model_type': 'vggt',
            'multi_image_mode': args.model_kwargs.get('multi_image_mode', False),
            'checkpoint_dir': args.model_kwargs.get('checkpoint_dir', 'checkpoints')
        }
    elif args.model_type == 'metric3d':
        config = {
            'model_type': 'metric3d',
            'checkpoint_dir': args.model_kwargs.get('checkpoint_dir', 'checkpoints')
        }
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    return config

def convert_output_to_depth(pred, model_type):
    """
    根据模型类型将输出统一转换为深度（米）
    """
    eps = 1e-6
    if model_type == 'depthanything':
        # 视差转深度
        depth = 1.0 / (pred + eps)
    elif model_type == 'zoedepth' or model_type == 'metric3d':
        # 已是米制深度
        depth = pred
    elif model_type == 'vggt':
        # 仿射不变深度，训练时可直接用，评估需做 scale alignment
        depth = pred
    else:
        depth = pred
    return depth

def resize_to_patch_multiple(img, patch_size=14):
    """
    自动将图片 resize 到 patch_size 的整数倍
    """
    import cv2
    h, w = img.shape[:2]
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    if (new_h, new_w) != (h, w):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img

# ----------- 微调 Decoder 主流程 -----------
def finetune_decoder(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 统一模型参数配置
    model_config = get_model_config(args)
    model_type = model_config.pop('model_type')
    model = create_model(model_type, device, **model_config)
    # 获取 decoder/encoder
    decoder, encoder = get_decoder_encoder(model, model_type)
    decoder.train()
    # 冻结其它参数，只训练 decoder
    for param in model.model.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = True
    # 加载数据集
    dataset = create_dataset(args.dataset_type, args.data_root)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]
    train_loader = [dataset[i] for i in train_idx]
    val_loader = [dataset[i] for i in val_idx]
    # 优化器和损失
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    # resume
    if args.resume and os.path.exists(args.resume):
        decoder.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed decoder weights from {args.resume}")
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        decoder.train()
        train_loss = 0
        for sample in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            img = sample['original_image']
            if model_type == 'depthanything':
                img = resize_to_patch_multiple(img, patch_size=14)
            img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            gt = torch.from_numpy(sample['depth']).float().unsqueeze(0).to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = encoder(img)
            pred = decoder(features)
            pred_depth = convert_output_to_depth(pred, model_type)
            loss = criterion(pred_depth.squeeze(), gt.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")
        # 验证
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                img = sample['original_image']
                if model_type == 'depthanything':
                    img = resize_to_patch_multiple(img, patch_size=14)
                img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
                gt = torch.from_numpy(sample['depth']).float().unsqueeze(0).to(device)
                features = encoder(img)
                pred = decoder(features)
                pred_depth = convert_output_to_depth(pred, model_type)
                loss = criterion(pred_depth.squeeze(), gt.squeeze())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")
        # 保存 checkpoint
        ckpt_path = os.path.join(args.output_dir, f"decoder_epoch{epoch+1}.pth")
        torch.save(decoder.state_dict(), ckpt_path)
        print(f"Saved decoder checkpoint: {ckpt_path}")
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(args.output_dir, "decoder_best.pth")
            torch.save(decoder.state_dict(), best_ckpt_path)
            print(f"Best model updated: {best_ckpt_path}")
    print(f"Training finished. Best val loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    '''
    Finetune a model's decoder on a specified dataset.
    Usage:
    python finetune.py --model-type  --data-root assets/pinkfish
    '''
    parser = argparse.ArgumentParser(description='Finetune model decoder')
    parser.add_argument('--model-type', type=str, required=True, help='Model type')
    parser.add_argument('--dataset-type', type=str, default='standard', help='Dataset type')
    parser.add_argument('--data-root', type=str, required=True, help='Dataset root')
    parser.add_argument('--output-dir', type=str, default='checkpoints/finetune', help='Checkpoint output dir')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--resume', type=str, default='', help='Resume decoder weights')
    parser.add_argument('--model-kwargs', type=str, default='{}', help='Extra model kwargs as json string')
    args = parser.parse_args()
    args.model_kwargs = json.loads(args.model_kwargs)
    os.makedirs(args.output_dir, exist_ok=True)
    finetune_decoder(args)
