import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class GradCAMpp:
    """Grad-CAM++ 实现"""
    def __init__(self, model, target_layer):
        """
        参数:
        model: 预训练的深度学习模型
        target_layer: 目标层，通常是最后一个卷积层
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册前向和反向钩子
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, x, class_idx=None):
        """
        生成 Grad-CAM++ 类激活图
        
        参数:
        x: 输入图像张量，形状为 (batch_size, C, H, W)
        class_idx: 目标类别索引，如果为 None 则使用预测的最高可能类别
        
        返回:
        cam: 类激活映射 (热力图)
        """
        self.model.eval()
        logits = self.model(x)  # (batch_size, num_classes)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot)
        
        # 获取特征图和梯度
        activations = self.activations[0]  # (C, H, W)
        gradients = self.gradients[0]      # (C, H, W)
        
        # 计算 Grad-CAM++ 的权重
        # 1. 计算梯度的 ReLU 值
        grad_relu = torch.relu(gradients)  # (C, H, W)
        
        # 2. 计算每个通道的 alpha（权重因子）
        # alpha = 1 / (2 + sum(A * grad^2))，其中 grad^2 近似二阶导数的影响
        numerator = 1.0 / (2.0 + torch.sum(activations * grad_relu**2, dim=(1, 2), keepdim=True))  # (C, 1, 1)
        alpha = numerator / (torch.sum(grad_relu, dim=(1, 2), keepdim=True) + 1e-8)  # 避免除以零
        
        # 3. 计算最终权重
        weights = torch.sum(alpha * grad_relu, dim=(1, 2))  # (C,)
        weights = weights.reshape(-1, 1, 1)  # (C, 1, 1)
        
        # 生成 CAM
        cam = torch.sum(weights * activations, dim=0)  # (H, W)
        cam = torch.relu(cam)  # 去除负值
        
        # 归一化
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        cam = cam.cpu().numpy()
        return cam

def preprocess_image(img_path, device):
    """将图像预处理为模型输入格式"""
    image = Image.open(img_path)
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
    
    # 获取目标层（这里选择 layer1 的最后一层，可根据需要调整）
    target_layer = model.layer3[-1]
    
    # 初始化 GradCAM++
    gradcampp = GradCAMpp(model, target_layer)
    
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
    
    # 生成 Grad-CAM++
    cam = gradcampp(input_tensor, class_idx=predicted_class)
    
    # 生成叠加热力图
    superimposed_heatmap = generate_heatmap(cam, original_img)
    
    # 保存结果
    plt.figure(figsize=(6, 6))  # 单张图，调整大小
    plt.imshow(superimposed_heatmap)
    plt.title(f'Grad-CAM++ (Class: {predicted_class})')
    plt.axis('off')
    plt.savefig('gradcampp_heatmap.png', dpi=300, bbox_inches='tight')
    print("Grad-CAM++ heatmap saved as 'gradcampp_heatmap.png'")
    plt.close()  # 关闭图形，避免显示

if __name__ == "__main__":
    main()