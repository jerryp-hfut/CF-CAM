import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from skimage.metrics import structural_similarity as ssim
from cfcam import CFCAM

class GradCAM:
    """Grad-CAM 实现，支持梯度加噪"""
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
    
    def __call__(self, x, class_idx=None, noise_level=None):
        self.model.eval()
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot)
        
        # 对梯度加噪（如果指定了 noise_level）
        gradients = self.gradients
        if noise_level is not None:
            noise = torch.randn_like(gradients) * noise_level
            gradients = gradients + noise
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

class GradCAMpp:
    """Grad-CAM++ 实现，支持梯度加噪"""
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
    
    def __call__(self, x, class_idx=None, noise_level=None):
        self.model.eval()
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot)
        
        activations = self.activations[0]
        gradients = self.gradients[0]
        
        if noise_level is not None:
            noise = torch.randn_like(gradients) * noise_level
            gradients = gradients + noise
        
        grad_relu = torch.relu(gradients)
        numerator = 1.0 / (2.0 + torch.sum(activations * grad_relu**2, dim=(1, 2), keepdim=True))
        alpha = numerator / (torch.sum(grad_relu, dim=(1, 2), keepdim=True) + 1e-8)
        weights = torch.sum(alpha * grad_relu, dim=(1, 2)).reshape(-1, 1, 1)
        
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

class SmoothGradCAMpp:
    """SmoothGradCAM++ 实现，通过输入噪声平滑梯度，支持梯度加噪测试"""
    def __init__(self, model, target_layer, n_samples=10, noise_std=0.1):
        """
        参数:
        model: 预训练的深度学习模型
        target_layer: 目标层
        n_samples: 采样次数，用于平滑
        noise_std: 输入噪声的标准差
        """
        self.model = model
        self.target_layer = target_layer
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, x, class_idx=None, noise_level=None):
        self.model.eval()
        
        if class_idx is None:
            logits = self.model(x)
            class_idx = torch.argmax(logits, dim=1).item()
        
        # 初始化平均梯度和特征图
        avg_gradients = None
        activations = None
        
        # 多次采样
        for _ in range(self.n_samples):
            # 添加输入噪声
            noisy_x = x + torch.randn_like(x) * self.noise_std
            noisy_x = torch.clamp(noisy_x, 0, 1)  # 确保输入在有效范围内
            
            self.model.zero_grad()
            logits = self.model(noisy_x)
            one_hot = torch.zeros_like(logits)
            one_hot[0, class_idx] = 1
            logits.backward(gradient=one_hot)
            
            if activations is None:
                activations = self.activations[0]  # 只取第一次的特征图
            gradients = self.gradients[0]
            
            # 对梯度加噪（用于鲁棒性测试）
            if noise_level is not None:
                noise = torch.randn_like(gradients) * noise_level
                gradients = gradients + noise
            
            # 累加梯度
            if avg_gradients is None:
                avg_gradients = gradients / self.n_samples
            else:
                avg_gradients += gradients / self.n_samples
        
        # 计算 Grad-CAM++ 的权重
        grad_relu = torch.relu(avg_gradients)
        numerator = 1.0 / (2.0 + torch.sum(activations * grad_relu**2, dim=(1, 2), keepdim=True))
        alpha = numerator / (torch.sum(grad_relu, dim=(1, 2), keepdim=True) + 1e-8)
        weights = torch.sum(alpha * grad_relu, dim=(1, 2)).reshape(-1, 1, 1)
        
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

def preprocess_image(img_path, device):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, image

def load_model(model_path, device):
    import torchvision.models as models
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def overlay_heatmap(cam, original_img, alpha=0.4):
    cam_resized = cv2.resize(cam, (original_img.width, original_img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(original_img)
    superimposed_img = heatmap * alpha + img_np * (1 - alpha)
    return np.uint8(superimposed_img)

def evaluate_robustness(original_cam, noisy_cam):
    ssim_score = ssim(original_cam, noisy_cam, data_range=1.0)
    mse = np.mean((original_cam - noisy_cam) ** 2)
    return ssim_score, mse

def test_all_cams_with_noise(test_folder, output_folder, model, device, noise_level=0.1):
    os.makedirs(output_folder, exist_ok=True)
    
    # 初始化 CAM 方法
    cam_methods = {
        "GradCAM": GradCAM(model, model.layer4[-1]),
        "GradCAMpp": GradCAMpp(model, model.layer4[-1]),
        "CFCAM": CFCAM(model, model.layer4[-1]),
        "SmoothGradCAMpp": SmoothGradCAMpp(model, model.layer4[-1], n_samples=10, noise_std=0.1)
    }
    
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files found in {test_folder}")
        return None, None
    
    # 存储所有图像的 SSIM 和 MSE
    results = {name: {"ssim": [], "mse": []} for name in cam_methods.keys()}
    
    print(f"Testing {len(image_files)} images with noise level {noise_level}...")
    for img_file in image_files:
        img_path = os.path.join(test_folder, img_file)
        try:
            input_tensor, original_img = preprocess_image(img_path, device)
            
            for cam_name, cam_method in cam_methods.items():
                # 生成原始和加噪热力图
                original_cam = cam_method(input_tensor)
                noisy_cam = cam_method(input_tensor, noise_level=noise_level)
                
                # 计算 SSIM 和 MSE
                ssim_score, mse = evaluate_robustness(original_cam, noisy_cam)
                results[cam_name]["ssim"].append(ssim_score)
                results[cam_name]["mse"].append(mse)
                
                # 可视化
                orig_superimposed = overlay_heatmap(original_cam, original_img)
                noisy_superimposed = overlay_heatmap(noisy_cam, original_img)
                
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(orig_superimposed)
                plt.title(f"{cam_name} Original")
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(noisy_superimposed)
                plt.title(f"{cam_name} Noisy (σ={noise_level})")
                plt.axis('off')
                
                output_path = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_{cam_name.lower()}_noise.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"{cam_name} - {img_file}: SSIM={ssim_score:.4f}, MSE={mse:.6f}")
        
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # 计算平均值
    avg_results = {
        name: {
            "avg_ssim": np.mean(metrics["ssim"]),
            "avg_mse": np.mean(metrics["mse"])
        }
        for name, metrics in results.items()
    }
    return avg_results, results

def plot_robustness_curves(test_folder, output_folder, model, device, noise_levels=[0.05, 0.1, 0.2, 0.3]):
    os.makedirs(output_folder, exist_ok=True)
    
    cam_methods = ["GradCAM", "GradCAMpp", "CFCAM", "SmoothGradCAMpp"]
    ssim_curves = {name: [] for name in cam_methods}
    mse_curves = {name: [] for name in cam_methods}
    
    for noise_level in noise_levels:
        avg_results, _ = test_all_cams_with_noise(test_folder, output_folder, model, device, noise_level)
        if avg_results is None:
            continue
        for cam_name in cam_methods:
            ssim_curves[cam_name].append(avg_results[cam_name]["avg_ssim"])
            mse_curves[cam_name].append(avg_results[cam_name]["avg_mse"])
    
    # 绘制 SSIM 曲线
    plt.figure(figsize=(10, 6))
    for cam_name in cam_methods:
        plt.plot(noise_levels, ssim_curves[cam_name], marker='o', label=cam_name)
    plt.xlabel("Noise Level (σ)")
    plt.ylabel("Average SSIM")
    plt.title("SSIM vs Noise Level for Different CAM Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "ssim_vs_noise.png"), dpi=300)
    plt.close()
    
    # 绘制 MSE 曲线
    plt.figure(figsize=(10, 6))
    for cam_name in cam_methods:
        plt.plot(noise_levels, mse_curves[cam_name], marker='o', label=cam_name)
    plt.xlabel("Noise Level (σ)")
    plt.ylabel("Average MSE")
    plt.title("MSE vs Noise Level for Different CAM Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "mse_vs_noise.png"), dpi=300)
    plt.close()
    
    print(f"Robustness curves saved to {output_folder}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_folder = "testimgs"
    output_folder = "cam_robustness_results"
    model_path = "best_model.pth"
    
    if not os.path.exists(test_folder) or not os.path.exists(model_path):
        print("Error: Required paths do not exist.")
        exit(1)
    
    model = load_model(model_path, device)
    
    # 测试单一噪声水平并返回平均 SSIM 和 MSE
    avg_results, _ = test_all_cams_with_noise(test_folder, output_folder, model, device, noise_level=0.1)
    if avg_results:
        for cam_name, metrics in avg_results.items():
            print(f"{cam_name} - Avg SSIM: {metrics['avg_ssim']:.4f}, Avg MSE: {metrics['avg_mse']:.6f}")
    
    plot_robustness_curves(test_folder, output_folder, model, device, noise_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])