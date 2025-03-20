import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from cfcam import CFCAM

def preprocess_image(img_path, device):
    """预处理图像并返回张量和原始 PIL 图像"""
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, image

def load_model(model_path, device):
    """加载训练好的模型"""
    import torchvision.models as models
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def overlay_heatmap(cam, original_img, alpha=0.4):
    """将热力图叠加到原始图像上"""
    cam_resized = cv2.resize(cam, (original_img.width, original_img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(original_img)
    superimposed_img = heatmap * alpha + img_np * (1 - alpha)
    return np.uint8(superimposed_img)

def create_masked_image(original_image, cam, threshold=0.5):
    """
    创建一个遮挡不重要区域的图像
    
    参数:
    original_image: PIL图像对象
    cam: Grad-CAM热力图
    threshold: 阈值，用于确定被视为"重要"的区域
    
    返回:
    masked_image: 遮挡不重要区域后的PIL图像
    """
    # 将原始图像转换为numpy数组
    img_array = np.array(original_image)
    
    # 将CAM调整为与原始图像相同的尺寸
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    
    # 创建掩码（将不重要区域设为0，重要区域为1）
    mask = np.zeros_like(cam_resized)
    mask[cam_resized > threshold] = 1
    
    # 扩展掩码维度以匹配图像
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    
    # 应用掩码（将不重要区域遮挡）
    masked_array = img_array * mask
    
    # 转回PIL图像
    masked_image = Image.fromarray(np.uint8(masked_array))
    return masked_image

def calculate_average_drop(model, device, test_file_list, data_folder, output_folder=None, save_images=False):
    """
    计算Average Drop指标
    
    参数:
    model: 训练好的模型
    device: 计算设备
    test_file_list: 包含测试图像文件名的文本文件路径
    data_folder: 包含图像的文件夹路径
    output_folder: 输出文件夹，用于保存可视化结果（可选）
    save_images: 是否保存遮挡后的图像和热力图（可选）
    
    返回:
    average_drop: Average Drop指标值（百分比）
    """
    # 初始化GradCAM
    grad_cam = CFCAM(model, model.layer4[-1])
    
    # 读取测试文件列表
    with open(test_file_list, 'r') as f:
        image_files = [line.strip() for line in f.readlines()]
    
    # 创建输出文件夹（如果需要）
    if save_images and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # 初始化统计数据
    total_images = len(image_files)
    drops = []
    processed_images = []
    
    # 处理每个图像文件
    for i, img_file in enumerate(image_files):
        try:
            img_path = os.path.join(data_folder, img_file)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} does not exist, skipping.")
                continue
                
            # 预处理图像
            input_tensor, original_img = preprocess_image(img_path, device)
            
            # 获取原始预测和Grad-CAM
            with torch.no_grad():
                original_output = model(input_tensor)
                original_prob = torch.softmax(original_output, dim=1)
                predicted_class = torch.argmax(original_output, dim=1).item()
                original_confidence = original_prob[0, predicted_class].item()
            
            # 获取Grad-CAM热力图
            cam, class_idx, _ = grad_cam(input_tensor, predicted_class)
            
            # 创建遮挡不重要区域的图像
            masked_img = create_masked_image(original_img, cam, threshold=0.5)
            
            # 处理遮挡后的图像
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            masked_tensor = transform(masked_img).unsqueeze(0).to(device)
            
            # 获取遮挡后的预测
            with torch.no_grad():
                masked_output = model(masked_tensor)
                masked_prob = torch.softmax(masked_output, dim=1)
                masked_confidence = masked_prob[0, predicted_class].item()
            
            # 计算drop（百分比）
            drop = (original_confidence - masked_confidence) / original_confidence * 100
            drops.append(drop)
            processed_images.append(img_file)
            
            # 可视化和保存结果（如果需要）
            if save_images and output_folder:
                # 创建一个包含原始图像、热力图和遮挡图像的组合图
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(original_img)
                plt.title(f"Original: {original_confidence:.4f}")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                superimposed_img = overlay_heatmap(cam, original_img)
                plt.imshow(superimposed_img)
                plt.title(f"GradCAM")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(masked_img)
                plt.title(f"Masked: {masked_confidence:.4f}")
                plt.axis('off')
                
                plt.suptitle(f"Drop: {drop:.2f}%")
                
                # 保存图像
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                output_path = os.path.join(output_folder, f"{base_name}_drop.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"Processed {i+1}/{total_images} images. Current avg drop: {np.mean(drops):.2f}%")
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    # 计算Average Drop
    average_drop = np.mean(drops) if drops else 0
    
    # 保存Average Drop结果到文件
    if output_folder:
        with open(os.path.join(output_folder, "average_drop_results.txt"), "w") as f:
            f.write(f"Average Drop: {average_drop:.2f}%\n")
            f.write(f"Total images processed: {len(drops)}/{total_images}\n")
            
            # 添加每个图像的详细信息
            f.write("\nDetailed Drop values:\n")
            for img_file, drop_val in zip(processed_images, drops):
                f.write(f"{img_file}: {drop_val:.2f}%\n")
    
    return average_drop

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 配置路径
    model_path = "best_model.pth"
    test_file_list = "test_files.txt"
    data_folder = "CXR_png"
    output_folder = "average_drop_results"

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        exit(1)
    if not os.path.exists(test_file_list):
        print(f"Error: Test file list '{test_file_list}' does not exist.")
        exit(1)
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' does not exist.")
        exit(1)

    # 加载模型
    print("Loading model...")
    model = load_model(model_path, device)

    # 计算Average Drop
    print("Calculating Average Drop...")
    avg_drop = calculate_average_drop(
        model=model,
        device=device,
        test_file_list=test_file_list,
        data_folder=data_folder,
        output_folder=output_folder,
        save_images=False  # 设置为False可以加快处理速度
    )

    print(f"Overall Average Drop: {avg_drop:.2f}%")