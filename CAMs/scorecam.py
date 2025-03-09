import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class ScoreCAM:
    """Score-CAM 实现"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # 注册前向钩子以捕获特征图
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        target_layer.register_forward_hook(forward_hook)
    
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # 前向传播获取激活和原始输出
        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        # 获取特征图
        activations = self.activations[0]  # (C, H, W)
        num_channels, h, w = activations.shape
        
        # 获取输入图像的空间尺寸
        input_h, input_w = x.shape[2], x.shape[3]
        
        # 初始化权重
        weights = []
        
        # 对每个通道计算得分
        for i in range(num_channels):
            # 提取并归一化当前通道的特征图
            activation_map = activations[i]  # (H, W)
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            
            # 上采样到输入图像大小
            activation_map = activation_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mask = nn.functional.interpolate(
                activation_map, size=(input_h, input_w), mode='bilinear', align_corners=False
            ).squeeze()  # (input_h, input_w)
            
            # 将掩码应用到输入图像
            masked_input = x * mask.unsqueeze(0).unsqueeze(0)  # (1, C, input_h, input_w)
            
            # 计算模型得分
            with torch.no_grad():
                score = self.model(masked_input)[0, class_idx]
            weights.append(score.item())
        
        # 转换为张量并归一化权重
        weights = torch.tensor(weights, device=activations.device)
        weights = weights / (weights.sum() + 1e-8)  # 归一化
        
        # 生成 CAM
        cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)  # (H, W)
        cam = torch.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

def preprocess_image(img_path, device):
    """将图像预处理为模型输入格式"""
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, image

def generate_heatmap(cam, img):
    """将CAM转换为热力图并叠加到原始图像上"""
    # 将CAM大小调整为与原图一致
    cam = cv2.resize(cam, (img.width, img.height))
    
    # 转换为热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 转换原图为numpy数组
    img_np = np.array(img)
    
    # 叠加热力图和原图
    superimposed_img = heatmap * 0.4 + img_np * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img

def main():
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载预训练的 ResNet50 模型
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # 获取目标层（这里选择 layer3 的最后一层，可根据需要调整）
    target_layer = model.layer3[-1]
    
    # 初始化 ScoreCAM
    scorecam = ScoreCAM(model, target_layer)
    
    # 测试图片路径
    img_path = "../input/zebra.jpg"
    
    # 检查图片是否存在
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return
    
    # 预处理图像
    input_tensor, original_img = preprocess_image(img_path, device)
    
    # 使用模型进行前向传播并获取预测类别
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = predicted_idx.item()
    
    # 生成 Score-CAM
    cam = scorecam(input_tensor, class_idx=predicted_class)
    
    # 生成叠加热力图
    superimposed_heatmap = generate_heatmap(cam, original_img)
    
    # 保存结果
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_heatmap)
    plt.title(f'Score-CAM (Class: {predicted_class})')
    plt.axis('off')
    plt.savefig('scorecam_heatmap.png', dpi=300, bbox_inches='tight')
    print("Score-CAM heatmap saved as 'scorecam_heatmap.png'")
    plt.close()

if __name__ == "__main__":
    main()
