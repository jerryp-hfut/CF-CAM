import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
import numpy as np
from cfcam import CFCAM

def preprocess_image(img_path, device):
    """预处理图像并返回张量，确保数据类型为float32"""
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 默认生成float32张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).float().to(device)  # 显式转换为float32并移到设备
    return input_tensor

def load_model(model_path, device):
    """加载训练好的模型"""
    import torchvision.models as models
    model = models.resnet50(weights=None)  # 不使用预训练权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 假设输出为2类
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def compute_curves(model, gradcam, inputs, class_idx, device):
    """计算删除和插入曲线的AUC"""
    # 获取Grad-CAM热图
    cam, _, _ = gradcam(inputs)
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float().to(device)  # 确保float32
    cam_upsampled = F.interpolate(cam_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

    # 展平并排序像素重要性
    cam_flat = cam_upsampled.flatten()
    sorted_indices = np.argsort(cam_flat)[::-1]  # 降序
    total_pixels = 224 * 224
    step_size = total_pixels // 100  # 100个步骤

    # 初始化曲线分数
    deletion_scores = []
    insertion_scores = []
    fraction_perturbed = np.linspace(0, 1, 101)  # 0到1的扰动比例

    # 原始图像和空白图像
    original_image = inputs.clone()
    blank_image = torch.zeros_like(inputs)

    # 计算原始预测概率
    with torch.no_grad():
        original_prob = F.softmax(model(original_image), dim=1)[0, class_idx].item()

    # 删除曲线
    for k in range(101):
        num_deleted = total_pixels if k == 100 else k * step_size
        mask_2d = np.ones((224, 224), dtype=np.float32)  # 确保掩码为float32
        deleted_indices = sorted_indices[:num_deleted]
        mask_2d.ravel()[deleted_indices] = 0
        mask_image = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float().to(device)  # 转换为float32
        
        modified_image = original_image * mask_image
        with torch.no_grad():
            prob_modified = F.softmax(model(modified_image), dim=1)[0, class_idx].item()
        deletion_scores.append(prob_modified)

    # 插入曲线
    for k in range(101):
        num_inserted = total_pixels if k == 100 else k * step_size
        mask_2d = np.zeros((224, 224), dtype=np.float32)  # 确保掩码为float32
        inserted_indices = sorted_indices[:num_inserted]
        mask_2d.ravel()[inserted_indices] = 1
        mask_image = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float().to(device)  # 转换为float32
        
        modified_image = original_image * mask_image + blank_image * (1 - mask_image)
        with torch.no_grad():
            prob_modified = F.softmax(model(modified_image), dim=1)[0, class_idx].item()
        insertion_scores.append(prob_modified)

    # 计算AUC
    deletion_auc = np.trapz(deletion_scores, fraction_perturbed)
    insertion_auc = np.trapz(insertion_scores, fraction_perturbed)
    return deletion_auc, insertion_auc

def calculate_average_auc(model, device, test_file_list, data_folder):
    """计算Deletion Curve和Insertion Curve的平均AUC"""
    # 初始化GradCAM
    grad_cam = CFCAM(model, model.layer4[-1])

    # 读取测试文件列表
    with open(test_file_list, 'r') as f:
        image_files = [line.strip() for line in f.readlines()]
    
    deletion_aucs = []
    insertion_aucs = []
    total_images = len(image_files)
    
    # 处理每个图像文件
    for i, img_file in enumerate(image_files):
        try:
            img_path = os.path.join(data_folder, img_file)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} does not exist, skipping.")
                continue
                
            # 预处理图像
            input_tensor = preprocess_image(img_path, device)
            
            # 获取原始预测
            with torch.no_grad():
                original_output = model(input_tensor)
                predicted_class = torch.argmax(original_output, dim=1).item()
            
            # 计算曲线和AUC
            deletion_auc, insertion_auc = compute_curves(model, grad_cam, input_tensor, predicted_class, device)
            deletion_aucs.append(deletion_auc)
            insertion_aucs.append(insertion_auc)
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"Processed {i+1}/{total_images} images. "
                      f"Current avg deletion AUC: {np.mean(deletion_aucs):.4f}, "
                      f"Current avg insertion AUC: {np.mean(insertion_aucs):.4f}")
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    # 计算平均AUC
    mean_deletion_auc = np.mean(deletion_aucs) if deletion_aucs else 0
    mean_insertion_auc = np.mean(insertion_aucs) if insertion_aucs else 0
    
    return mean_deletion_auc, mean_insertion_auc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 配置路径
    model_path = "best_model.pth"
    test_file_list = "test_files.txt"
    data_folder = "CXR_png"

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

    # 计算平均AUC
    print("Calculating Average AUCs...")
    mean_deletion_auc, mean_insertion_auc = calculate_average_auc(
        model=model,
        device=device,
        test_file_list=test_file_list,
        data_folder=data_folder
    )

    print(f"Overall Average Deletion AUC: {mean_deletion_auc:.4f}")
    print(f"Overall Average Insertion AUC: {mean_insertion_auc:.4f}")