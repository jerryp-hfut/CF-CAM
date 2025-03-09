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

class GradCAM:
    """原始Grad-CAM实现"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
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
    gradcam = GradCAM(model, target_layer)
    
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
    cam = gradcam(input_tensor, class_idx=predicted_class)
    
    # 生成叠加热力图
    superimposed_heatmap = generate_heatmap(cam, original_img)
    
    # 保存结果
    plt.figure(figsize=(6, 6))  # 单张图，调整大小
    plt.imshow(superimposed_heatmap)
    plt.title(f'Grad-CAM (Class: {predicted_class})')
    plt.axis('off')
    plt.savefig('gradcam_heatmap.png', dpi=300, bbox_inches='tight')
    print("Grad-CAM heatmap saved as 'gradcam_heatmap.png'")
    plt.close()  # 关闭图形，避免显示

if __name__ == "__main__":
    main()