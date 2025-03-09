import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class AblationCAM:
    """Ablation-CAM 实现"""
    def __init__(self, model, target_layer):
        """
        参数:
        model: 预训练的深度学习模型
        target_layer: 目标层，通常是最后一个卷积层
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # 注册前向钩子以捕获特征图
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        target_layer.register_forward_hook(forward_hook)
    
    def __call__(self, x, class_idx=None):
        """
        生成 Ablation-CAM 类激活图
        
        参数:
        x: 输入图像张量，形状为 (batch_size, C, H, W)
        class_idx: 目标类别索引，如果为 None 则使用预测的最高可能类别
        
        返回:
        cam: 类激活映射 (热力图)
        """
        self.model.eval()
        
        # 获取原始输出和特征图
        with torch.no_grad():
            original_logits = self.model(x)
            if class_idx is None:
                class_idx = torch.argmax(original_logits, dim=1).item()
            original_score = original_logits[0, class_idx].item()
        
        activations = self.activations[0]  # (C, H, W)
        num_channels = activations.shape[0]
        
        # 计算每个通道的消融权重
        weights = []
        for i in range(num_channels):
            # 创建特征图副本并消融当前通道
            ablated_activations = activations.clone()
            ablated_activations[i] = 0  # 将第 i 个通道置零
            
            # 替换目标层的输出并重新计算得分
            def ablate_hook(module, input, output):
                return ablated_activations.unsqueeze(0)  # 恢复 batch 维度
            
            handle = self.target_layer.register_forward_hook(ablate_hook)
            with torch.no_grad():
                ablated_logits = self.model(x)
                ablated_score = ablated_logits[0, class_idx].item()
            handle.remove()  # 移除钩子
            
            # 计算得分下降作为权重
            weight = max(original_score - ablated_score, 0)  # 只保留正贡献
            weights.append(weight)
        
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
    cam = cv2.resize(cam, (img.width, img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(img)
    superimposed_img = heatmap * 0.4 + img_np * 0.6
    return np.uint8(superimposed_img)

def main():
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载预训练的 ResNet50 模型
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # 获取目标层（这里选择 layer3 的最后一层，可根据需要调整）
    target_layer = model.layer3[-1]
    
    # 初始化 AblationCAM
    ablationcam = AblationCAM(model, target_layer)
    
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
    
    # 生成 Ablation-CAM
    cam = ablationcam(input_tensor, class_idx=predicted_class)
    
    # 生成叠加热力图
    superimposed_heatmap = generate_heatmap(cam, original_img)
    
    # 保存结果
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_heatmap)
    plt.title(f'Ablation-CAM (Class: {predicted_class})')
    plt.axis('off')
    plt.savefig('ablationcam_heatmap.png', dpi=300, bbox_inches='tight')
    print("Ablation-CAM heatmap saved as 'ablationcam_heatmap.png'")
    plt.close()

if __name__ == "__main__":
    main()