'''
ClusterInterpret Ver. 1.0
Author: Xu Pan, Hongjie He
Last edited time: 2025/2/26
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from scipy.spatial.distance import pdist, squareform

class ClusterBasedInterpretability:
    def __init__(self, model, target_layer, num_clusters=5):
        """
        初始化基于聚类的可解释性算法
        
        参数:
        model: 预训练的神经网络模型
        target_layer: 目标层名称
        num_clusters: 聚类数量
        """
        self.model = model
        self.target_layer = target_layer
        self.num_clusters = num_clusters
        self.feature_maps = None
        self.gradients = None
        
        # 注册钩子
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向和反向传播的钩子函数"""
        def forward_hook(module, input, output):
            self.feature_maps = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 目标层
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.target_module = module
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def _compute_feature_similarity(self, features):
        """
        计算特征图之间的相似度
        
        参数:
        features: 特征图 [C, H, W]
        
        返回:
        similarity_matrix: 相似度矩阵 [C, C]
        """
        C, H, W = features.shape
        
        # 特征图展平为向量形式
        features_flat = features.reshape(C, -1)
        
        # 余弦相似度
        features_norm = features_flat / (torch.norm(features_flat, dim=1, keepdim=True) + 1e-8)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        return similarity_matrix.cpu().numpy()
    
    def _cluster_feature_maps(self, features, similarity_matrix=None):
        """
        对特征图进行聚类
        
        参数:
        features: 特征图 [C, H, W]
        similarity_matrix: 可选的预计算相似度矩阵
        
        返回:
        cluster_labels: 聚类标签
        """
        C, H, W = features.shape
        
        if similarity_matrix is None:
            similarity_matrix = self._compute_feature_similarity(features)
        
        # 相似度矩阵转距离矩阵
        distance_matrix = 1 - similarity_matrix
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        return cluster_labels
    
    def _fuse_feature_maps(self, features, cluster_labels):
        """
        融合相同簇内的特征图
        
        参数:
        features: 特征图 [C, H, W]
        cluster_labels: 聚类标签
        
        返回:
        fused_maps: 融合后的特征图 [K, H, W]，其中K是聚类数量
        """
        C, H, W = features.shape
        fused_maps = torch.zeros((self.num_clusters, H, W), device=features.device)
        counts = torch.zeros(self.num_clusters, device=features.device)
        
        # 对每个簇内的特征图求平均
        for i, label in enumerate(cluster_labels):
            fused_maps[label] += features[i]
            counts[label] += 1
        
        # 避免除以零
        counts = torch.clamp(counts, min=1)
        
        # 计算平均值
        for k in range(self.num_clusters):
            fused_maps[k] = fused_maps[k] / counts[k]
        
        return fused_maps
    
    def _compute_cluster_gradients(self, gradients, cluster_labels):
        """
        计算每个簇的平均梯度权重
        
        参数:
        gradients: 梯度 [C, H, W]
        cluster_labels: 聚类标签
        
        返回:
        cluster_weights: 每个簇的梯度权重
        """
        C, H, W = gradients.shape
        cluster_weights = torch.zeros(self.num_clusters, device=gradients.device)
        counts = torch.zeros(self.num_clusters, device=gradients.device)
        
        # 计算每个簇的平均梯度
        for i, label in enumerate(cluster_labels):
            cluster_weights[label] += torch.mean(torch.abs(gradients[i]))
            counts[label] += 1
        
        # 避免除零
        counts = torch.clamp(counts, min=1)
        
        # 计算平均梯度
        for k in range(self.num_clusters):
            cluster_weights[k] = cluster_weights[k] / counts[k]
        
        return cluster_weights
    
    def generate_heatmap(self, img_path, class_idx=None):
        """
        生成热力图
        
        参数:
        img_path: 输入图像路径
        class_idx: 目标类别索引，如果为None则使用预测的最高概率类别
        
        返回:
        heatmap: 热力图
        pred_class: 预测的类别
        """
        # 加载并预处理图像
        img = Image.open(img_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)
        
        # 前向传播
        self.model.eval()
        logits = self.model(input_tensor)
        
        # 如果未指定类别，使用预测的最高概率类别
        if class_idx is None:
            pred = logits.argmax(dim=1)
            class_idx = pred.item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot)
        
        # 获取特征图和梯度
        feature_maps = self.feature_maps.detach()[0]  # [C, H, W]
        gradients = self.gradients.detach()[0]  # [C, H, W]
        
        # 计算特征图相似度
        similarity_matrix = self._compute_feature_similarity(feature_maps)
        
        # 聚类特征图
        cluster_labels = self._cluster_feature_maps(feature_maps, similarity_matrix)
        
        # 融合特征图
        fused_maps = self._fuse_feature_maps(feature_maps, cluster_labels)
        
        # 计算每个簇的梯度权重
        cluster_weights = self._compute_cluster_gradients(gradients, cluster_labels)
        
        # 加权融合得到最终的热力图
        weighted_sum = torch.zeros_like(fused_maps[0])
        for k in range(self.num_clusters):
            weighted_sum += fused_maps[k] * cluster_weights[k]
        
        # 转换为热力图
        heatmap = weighted_sum.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        # 归一化
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap, class_idx
    
    def visualize(self, img_path, class_idx=None, alpha=0.5, cmap='jet'):
        """
        可视化热力图叠加在原始图像上
        
        参数:
        img_path: 输入图像路径
        class_idx: 目标类别索引
        alpha: 热力图透明度
        cmap: 颜色映射
        
        返回:
        superimposed_img: 叠加后的图像
        """
        heatmap, pred_class = self.generate_heatmap(img_path, class_idx)
        
        # 加载原始图像
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # 将热力图调整为与原始图像相同的大小
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), getattr(cv2, f'COLORMAP_{cmap.upper()}'))
        
        # 叠加热力图和原始图像
        superimposed_img = heatmap * alpha + img * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)
        
        return superimposed_img, pred_class

# 测试代码
def test_on_resnet(img_path, target_layer='layer2', num_clusters=5):
    """
    在ResNet上测试算法
    
    参数:
    img_path: 测试图像路径
    target_layer: 目标层
    num_clusters: 聚类数量
    """
    # 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=True)
    model.eval()
    print("Model initialized successfully.")
    # 初始化解释器
    interpreter = ClusterBasedInterpretability(model, target_layer, num_clusters)
    print("Interpreter initialized successfully.")
    # 生成并可视化热力图
    superimposed_img, pred_class = interpreter.visualize(img_path)
    
    # 显示结果
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Heatmap (Predicted Class: {pred_class})')
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'result_1/cluster{num_clusters}.png')
    plt.show()
    print("heatmap saved successfully.")
    return interpreter

# 主函数
if __name__ == "__main__":
    test_on_resnet('input/BWshark.jpg')