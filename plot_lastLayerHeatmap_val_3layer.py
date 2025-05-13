import argparse 
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import glob
from models.common import DetectMultiBackend

import torch
from models.yolo import Model

"""
    验证模式下，一共22层，第22层是最后的输出。
    ·大目标特征图 (cv2)：特征图分辨率高，适合检测较大目标。cv2.0.2是特征
    ·中目标特征图 (cv3)：特征图分辨率适中，兼顾检测中等大小目标。cv3.0.1是特征，cv3.0.2是分类结果
    ·小目标特征图 (cv4)：特征图分辨率低，适合检测小目标。cv4是小目标特征，cv5是分类结果

    通过卷积层的特征图合并在一起来展示。
    'model.22.cv2.1.conv',  # cv2 的特征输出
    'model.22.cv3.1.conv',  # cv3 的特征输出
    'model.22.cv4.conv',  # cv4 的特征输出s
"""

features = {}

def load_model(weights, device, out_channels=1):
    cfg = "./data/gelan-c.yaml"
    model = Model(cfg, ch=3, nc=41, anchors=3).to(device)
    _ = model.eval()
    ckpt = torch.load(weights, map_location=device)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc
    return ckpt['model']

    return model

def hook_fn(module, input, output):
    global features
    # print(f"Hook triggered for {module.name}")
    if output is not None:
        features[module.name] = output  # 存储特征
        # print(f"Captured output shape: {output.shape}")
    else:
        print(f"No output captured for {module.name}")

def load_image(image_path, device):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device).half()

    return img_tensor

def data_mean_norm(feature_map):
    # 数据按照通道取均值并归一化 
    feature_map_mean = torch.mean(feature_map, dim=1) # [1, C, H, W]到[1, H, W]

    # return feature_map_mean.squeeze()

    feature_map_min = feature_map_mean.min()
    feature_map_max = feature_map_mean.max()
    feature_map_normalized = (feature_map_mean - feature_map_min) / (feature_map_max - feature_map_min)
    return feature_map_normalized.squeeze()
    
def fuse_features(features, target_layers, device):
    # 对齐通道数
    aligned_features = [
        data_mean_norm(features[layer]) for layer in target_layers
    ]

    # 将所有特征图堆叠并求和
    fused_feature = torch.stack(aligned_features, dim=0)  # 形状 [N, H, W]
    fused_feature = torch.sum(fused_feature, dim=0)  # 按元素求和，结果形状 [H, W]
    fused_feature = (fused_feature - fused_feature.min()) / (fused_feature.max() - fused_feature.min())

    return fused_feature.cpu().detach().numpy()

def save_heatmap(image_path, heatmap):
    original_image = cv2.imread(image_path)
    original_size = original_image.shape

    heatmap = heatmap.squeeze()
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_image, 0.5, heatmap_colored, 0.5, 0)

    return heatmap_colored, overlay

def main(weights, image_path, save_path, device):
    device = torch.device(device)
    model = load_model(weights, device)

    global features
    hooks = []

    target_layers = [
        'model.22.cv2.1.conv',  # cv2 的特征输出
        'model.22.cv3.1.conv',  # cv3 的特征输出
        'model.22.cv4.conv',  # cv4 的特征输出
    ]

    for layer_name in target_layers:
        module = model.get_submodule(layer_name)
        # print(f"Registering hook for {layer_name}")
        module.name = layer_name
        hooks.append(module.register_forward_hook(hook_fn))

    if(os.path.isfile(image_path)): # 如果是图片文件
        image_paths = [image_path]
    elif(os.path.isdir(image_path)): # 如果是图片文件夹
        image_paths = glob.glob(f'{image_path}/*.jpg')

    for im_path in image_paths:
        img_tensor = load_image(im_path, device)

        try:
            with torch.no_grad():
                _ = model(img_tensor)
        except:
            print(img_tensor.shape, im_path, "图像在模型推理过程中报错，跳过")
            continue

        if len(features) == 0:
            print("No features captured. Check the layer names.")
            return

        im_file_name = os.path.splitext(os.path.basename(im_path))[0]
        save_heatmap_path = os.path.join(save_path, f"{im_file_name}_heatmap.jpg")
        save_overlay_path = os.path.join(save_path, f"{im_file_name}_overlay.jpg")

        heatmap = fuse_features(features, target_layers, device)
        heatmap_colored, overlay = save_heatmap(im_path, heatmap)

        cv2.imwrite(save_heatmap_path, heatmap_colored)
        cv2.imwrite(save_overlay_path, overlay)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/BMblood/CP-yolov9-c/weights/best.pt', help='model path')
    parser.add_argument('--image_path', type=str, default='../data/BMBlood/IN/train/images/', help='image path or dir')
    parser.add_argument('--save_path', type=str, default='runs/heatmap/featureMap-layer-all/train/', help='save heatmap dir')
    parser.add_argument('--device', default='cuda:1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.save_path, exist_ok=True)
    main(opt.weights, opt.image_path, opt.save_path, opt.device)
