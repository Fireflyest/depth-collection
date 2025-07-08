import torch
import cv2
import numpy as np
import os

# 导入必要的模块
try:
    from mmcv.utils import Config
except:
    from mmengine import Config
from mono.model.monodepth_model import get_configured_monodepth_model

# 设置配置文件和权重文件路径
cfg_file = 'mono/configs/HourglassDecoder/vit.raft5.large.py'
weight_file = 'weight/metric_depth_vit_large_800k.pth'

# 加载配置并创建模型
cfg = Config.fromfile(cfg_file)
# 将Config对象转换为字典格式，因为get_configured_monodepth_model期望dict类型
if hasattr(cfg, '_cfg_dict'):
    cfg_dict = cfg._cfg_dict
else:
    cfg_dict = dict(cfg)
model = get_configured_monodepth_model(cfg_dict)
print("模型创建成功!")

# 加载本地权重
print(f"正在加载权重文件: {weight_file}")
checkpoint = torch.load(weight_file, map_location='cpu', weights_only=True)
print(f"权重文件包含的keys: {list(checkpoint.keys())}")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# 将模型移动到GPU
if torch.cuda.is_available():
    model = model.cuda()
    print("模型已移动到GPU")
else:
    print("CUDA不可用，使用CPU")
print("权重加载完成!")

# Load an example RGB image
rgb = cv2.imread('data/kitti_demo/rgb/0000000005.png')  # Replace with your image path
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# 按照hubconf.py中的预处理方式处理输入图像
input_size = (616, 1064)  # ViT model input size
h, w = rgb.shape[:2]
scale = min(input_size[0] / h, input_size[1] / w)
rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

# padding to input_size
padding = [123.675, 116.28, 103.53]
h, w = rgb.shape[:2]
pad_h = input_size[0] - h
pad_w = input_size[1] - w
pad_h_half = pad_h // 2
pad_w_half = pad_w // 2
rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)

# normalize
mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
rgb = torch.div((rgb - mean), std)
rgb = rgb[None, :, :, :] # add batch dimension

# 将输入数据移动到GPU
if torch.cuda.is_available():
    rgb = rgb.cuda()
    print("输入数据已移动到GPU")

# 模型推理
print(f"输入图像形状: {rgb.shape}")
print("开始模型推理...")
with torch.no_grad():
    pred_depth, confidence, output_dict = model.inference({'input': rgb})

print(f"预测深度形状: {pred_depth.shape}")
print(f"置信度形状: {confidence.shape}")
print("模型推理完成!")

# 可以添加保存深度图的代码
depth_numpy = pred_depth.squeeze().cpu().numpy()
cv2.imwrite('predicted_depth.png', (depth_numpy * 255 / depth_numpy.max()).astype(np.uint8))

