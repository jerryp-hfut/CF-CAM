import torch
import torch.nn as nn
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

def calculate_average_increase(model, device, test_file_list, data_folder, output_folder=None, save_images=False):
    # 初始化 GradCAM
    grad_cam = CFCAM(model, model.layer4[-1])
    
    # 读取测试文件列表
    with open(test_file_list, 'r') as f:
        image_files = [line.strip() for line in f.readlines()]
    
    # 创建输出文件夹（如果需要）
    if save_images and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # 初始化统计数据
    total_images = len(image_files)
    increases = []
    processed_images = []
    
    # 定义基准图像
    mean = [0.485, 0.456, 0.406]
    baseline_image = torch.ones(1, 3, 224, 224).to(device) * torch.Tensor(mean).view(1,3,1,1).to(device)
    
    # 处理每个图像文件
    for i, img_file in enumerate(image_files):
        try:
            img_path = os.path.join(data_folder, img_file)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} does not exist, skipping.")
                continue
            
            # 预处理图像
            input_tensor, original_img = preprocess_image(img_path, device)
            
            # 获取原始预测和 Grad-CAM
            with torch.no_grad():
                original_output = model(input_tensor)
                original_prob = torch.softmax(original_output, dim=1)
                predicted_class = torch.argmax(original_output, dim=1).item()
                original_confidence = original_prob[0, predicted_class].item()
            
            # 获取 CAM
            cam, class_idx, _ = grad_cam(input_tensor, predicted_class)
            
            # 调整 CAM 到输入大小 [224,224]
            cam_resized = cv2.resize(cam, (224, 224))
            
            # 创建重要区域掩码
            mask = (cam_resized > 0.3).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
            
            # 创建插入掩盖图像
            insertion_masked_image = baseline_image * (1 - mask) + input_tensor * mask
            
            # 计算基准置信度
            with torch.no_grad():
                baseline_output = model(baseline_image)
                C_baseline = torch.softmax(baseline_output, dim=1)[0, predicted_class].item()
            
            # 计算插入置信度
            with torch.no_grad():
                insertion_output = model(insertion_masked_image)
                C_insertion = torch.softmax(insertion_output, dim=1)[0, predicted_class].item()
            
            # 计算增加量
            increase = C_insertion - C_baseline
            increases.append(increase)
            processed_images.append(img_file)
            
            # 可视化和保存结果（如果需要）
            if save_images and output_folder:
                plt.figure(figsize=(15,5))
                
                plt.subplot(1,3,1)
                plt.imshow(original_img)
                plt.title(f"Original: {original_confidence:.4f}")
                plt.axis('off')
                
                plt.subplot(1,3,2)
                superimposed_img = overlay_heatmap(cam_resized / np.max(cam_resized), original_img.resize((224,224)))
                plt.imshow(superimposed_img)
                plt.title(f"GradCAM")
                plt.axis('off')
                
                plt.subplot(1,3,3)
                insertion_img = transforms.ToPILImage()(insertion_masked_image.cpu().squeeze(0))
                plt.imshow(insertion_img)
                plt.title(f"Insertion: {C_insertion:.4f}, Baseline: {C_baseline:.4f}, Increase: {increase:.4f}")
                plt.axis('off')
                
                plt.suptitle(f"Image: {img_file}")
                
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                output_path = os.path.join(output_folder, f"{base_name}_increase.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"Processed {i+1}/{total_images} images. Current avg increase: {np.mean(increases):.4f}")
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    # 计算 Average Increase
    average_increase = np.mean(increases) if increases else 0
    
    # 保存结果到文件
    if output_folder:
        with open(os.path.join(output_folder, "average_increase_results.txt"), "w") as f:
            f.write(f"Average Increase: {average_increase:.4f}\n")
            f.write(f"Total images processed: {len(increases)}/{total_images}\n")
            f.write("\nDetailed Increase values:\n")
            for img_file, inc_val in zip(processed_images, increases):
                f.write(f"{img_file}: {inc_val:.4f}\n")
    
    return average_increase

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
    print("Calculating Average Increase...")
    avg_drop = calculate_average_increase(
        model=model,
        device=device,
        test_file_list=test_file_list,
        data_folder=data_folder,
        output_folder=output_folder,
        save_images=False  # 设置为False可以加快处理速度
    )

    print(f"Overall Average Increase: {avg_drop:.2f}%")