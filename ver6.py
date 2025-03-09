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
from sklearn.cluster import DBSCAN

class ClusteredFilteredGradCAM:
    def __init__(self, model, target_layer, adaptive=True):
        """
        参数:
        model: 预训练的深度学习模型
        target_layer: 目标层，通常是网络的最后一个卷积层
        adaptive: 是否启用自适应参数（保留选项，但实际由滤波和聚类自适应）
        """
        self.model = model
        self.target_layer = target_layer
        self.adaptive = adaptive
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        for name, module in self.model.named_modules():
            if module == self.target_layer:
                self.target_module = module
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def estimate_dbscan_params(self, feature_maps_flat):
        distances = np.linalg.norm(feature_maps_flat[:, None] - feature_maps_flat[None, :], axis=2)
        eps = np.percentile(distances.flatten(), 25)
        min_samples = max(2, int(feature_maps_flat.shape[0] * 0.05))
        return eps, min_samples

    def cluster_with_dbscan(self, feature_maps_flat):
        eps, min_samples = self.estimate_dbscan_params(feature_maps_flat)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_maps_flat)
        labels = db.labels_
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        return labels, n_clusters

    def estimate_guided_filter_params(self, gradient):
        height, width = gradient.shape[-2], gradient.shape[-1]
        r = int(np.sqrt(height * width) / 4)
        eps = np.var(gradient.cpu().numpy().flatten()) * 0.1
        return r, eps

    def apply_guided_filter(self, gradient, guidance):
        r, eps = self.estimate_guided_filter_params(gradient)
        gradient_np = gradient.cpu().numpy()
        guidance_np = guidance.cpu().numpy()
        filtered = cv2.ximgproc.guidedFilter(guidance_np, gradient_np, r, eps)
        return filtered

    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot)

        feature_maps = self.feature_maps.detach()[0]  # (C, H, W)
        gradients = self.gradients.detach()[0]  # (C, H, W)

        feature_maps_flat = feature_maps.view(feature_maps.size(0), -1).cpu().numpy()
        cluster_labels, n_clusters = self.cluster_with_dbscan(feature_maps_flat)

        filtered_gradients = torch.zeros_like(gradients)
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_gradients = gradients[cluster_indices]
                mean_cluster_gradient = torch.mean(cluster_gradients, dim=0)
                filtered_mean_gradient = self.apply_guided_filter(mean_cluster_gradient, mean_cluster_gradient)
                for idx in cluster_indices:
                    filtered_gradients[idx] = torch.from_numpy(filtered_mean_gradient).to(gradients.device)

        weights = torch.mean(filtered_gradients, dim=(1, 2)).reshape(-1, 1, 1)
        cam = torch.sum(weights * feature_maps, dim=0)
        cam = torch.clamp(cam, min=0)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        return cam.cpu().numpy()

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
    cfcam = ClusteredFilteredGradCAM(model, target_layer)
    
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
    cam = cfcam(input_tensor, class_idx=predicted_class)
    
    # 生成叠加热力图
    superimposed_heatmap = generate_heatmap(cam, original_img)
    
    # 保存结果
    plt.figure(figsize=(6, 6))  # 单张图，调整大小
    plt.imshow(superimposed_heatmap)
    plt.title(f'CF-CAM (Class: {predicted_class})')
    plt.axis('off')
    plt.savefig('cfcam_heatmap.png', dpi=300, bbox_inches='tight')
    print("CF-CAM heatmap saved as 'cfcam_heatmap.png'")
    plt.close()  # 关闭图形，避免显示

if __name__ == "__main__":
    main()