'''
GMM+ActivationEnergy Interpret Ver. 2.0
Based on ClusterInterpret by Xu Pan, Hongjie He
Last edited time: 2025/2/28
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import cv2
import os

class GMMActivationEnergyInterpret:
    def __init__(self, model, target_layer, num_clusters='auto', cluster_method='bic'):
        """
        初始化基于GMM聚类和激活能量的频域解释算法
        
        参数:
        model: 预训练的神经网络模型
        target_layer: 目标层名称
        num_clusters: 聚类数量
        """
        self.model = model
        self.target_layer = target_layer
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        self.feature_maps = None
        
        # 注册钩子
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向传播的钩子函数"""
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        # 目标层
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.target_module = module
                module.register_forward_hook(forward_hook)
                break
    
    def _fft_transform(self, features):
        """
        对特征图进行傅里叶变换
        
        参数:
        features: 特征图 [C, H, W]
        
        返回:
        freq_features: 频域特征向量 [C, D]，其中D是降维后的特征维度
        """
        C, H, W = features.shape
        freq_features = []
        
        for i in range(C):
            # 对每个通道进行2D FFT
            feature_fft = torch.fft.fft2(features[i])
            # 将FFT结果移动到中心，低频在中间，高频在四周
            feature_fft_shift = torch.fft.fftshift(feature_fft)
            # 取幅度谱(magnitude spectrum)
            magnitude_spectrum = torch.abs(feature_fft_shift)
            # 对数变换，增强对比度
            log_magnitude = torch.log1p(magnitude_spectrum)  # log(1+x) to avoid log(0)
            # 展平为一维向量
            freq_feature_flat = log_magnitude.reshape(-1)
            freq_features.append(freq_feature_flat)
        
        # 将所有频域特征堆叠为一个张量 [C, H*W]
        freq_features = torch.stack(freq_features)
        
        # 使用PCA降维，减少计算量
        if freq_features.shape[1] > 512:
            freq_features_np = freq_features.cpu().numpy()
            pca = PCA(n_components=min(512, C))
            freq_features_reduced = pca.fit_transform(freq_features_np)
            freq_features = torch.tensor(freq_features_reduced, device=features.device)
        
        return freq_features

    def _find_optimal_clusters(self, freq_features, min_clusters=2, max_clusters=10, method='bic'):
        """
        自适应确定最佳的聚类数量
    
        参数:
        freq_features: 频域特征
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数
        method: 使用的方法 ('bic', 'aic', 'silhouette', 'gap')
    
        返回:
        optimal_k: 最佳聚类数量
        """
        features_np = freq_features.cpu().numpy()
    
        if method in ['bic', 'aic']:
            # 使用BIC或AIC评估
            scores = []
            for k in range(min_clusters, max_clusters + 1):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200,
                    tol=1e-3
                )
                gmm.fit(features_np)
            
                if method == 'bic':
                    scores.append(gmm.bic(features_np))
                else:  # AIC
                    scores.append(gmm.aic(features_np))
        
            # BIC和AIC越小越好
            optimal_k = range(min_clusters, max_clusters + 1)[np.argmin(scores)]
        
        elif method == 'silhouette':
            # 使用轮廓系数
            from sklearn.metrics import silhouette_score
            scores = []
        
            for k in range(min_clusters, max_clusters + 1):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200,
                    tol=1e-3
                )
                cluster_labels = gmm.fit_predict(features_np)
            
                # 当只有一个聚类或者所有样本都在一个聚类时，轮廓系数无法计算
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(features_np, cluster_labels)
                    scores.append(score)
                else:
                    scores.append(-1)  # 用较低的分数代替
        
            # 轮廓系数越大越好
            optimal_k = range(min_clusters, max_clusters + 1)[np.argmax(scores)]
        else:
            raise ValueError(f"Unsupported method: {method}")
    
        return optimal_k
    
    def _cluster_feature_maps_gmm(self, features, freq_features=None):
        """
        使用高斯混合模型对特征图进行聚类
        
        参数:
        features: 特征图 [C, H, W]
        freq_features: 预计算的频域特征
        
        返回:
        cluster_labels: 聚类标签
        """
        C, H, W = features.shape
        
        # 如果未提供频域特征，则计算频域特征
        if freq_features is None:
            freq_features = self._fft_transform(features)
        
        # 高斯混合模型聚类
        gmm = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type='full',
            random_state=42,
            max_iter=200,
            tol=1e-3
        )
        
        # 对频域特征进行聚类
        cluster_labels = gmm.fit_predict(freq_features.cpu().numpy())
        
        return cluster_labels
    
    def _compute_activation_energy(self, features):
        """
        计算每个特征图的激活能量
        
        参数:
        features: 特征图 [C, H, W]
        
        返回:
        activation_energy: 每个特征图的激活能量
        """
        C, H, W = features.shape
        activation_energy = torch.zeros(C, device=features.device)
        
        # 计算每个特征图的能量（平方和）
        for i in range(C):
            feature = features[i]
            activation_energy[i] = torch.sum(feature**2) / (H*W)
        
        # 归一化激活能量
        if torch.max(activation_energy) > 0:
            activation_energy = activation_energy / torch.max(activation_energy)
        
        return activation_energy
    
    def _compute_cluster_weights(self, features, cluster_labels):
        """
        计算每个簇的权重，基于激活能量
        
        参数:
        features: 特征图 [C, H, W]
        cluster_labels: 聚类标签
        
        返回:
        cluster_weights: 每个簇的权重
        """
        # 计算每个特征图的激活能量
        activation_energy = self._compute_activation_energy(features)
        
        # 计算每个簇的平均激活能量
        cluster_weights = torch.zeros(self.num_clusters, device=features.device)
        counts = torch.zeros(self.num_clusters, device=features.device)
        
        for i, label in enumerate(cluster_labels):
            cluster_weights[label] += activation_energy[i]
            counts[label] += 1
        
        # 避免除零
        counts = torch.clamp(counts, min=1)
        
        # 计算平均激活能量
        for k in range(self.num_clusters):
            cluster_weights[k] = cluster_weights[k] / counts[k]
        
        # 对簇权重进行归一化
        if torch.sum(cluster_weights) > 0:
            cluster_weights = cluster_weights / torch.sum(cluster_weights)
        
        return cluster_weights
    
    def _fuse_feature_maps(self, features, cluster_labels):
        """
        融合相同簇内的特征图，使用激活能量加权
        
        参数:
        features: 特征图 [C, H, W]
        cluster_labels: 聚类标签
        
        返回:
        fused_maps: 融合后的特征图 [K, H, W]，其中K是聚类数量
        """
        C, H, W = features.shape
        fused_maps = torch.zeros((self.num_clusters, H, W), device=features.device)
        counts = torch.zeros(self.num_clusters, device=features.device)
        
        # 计算每个特征图的激活能量
        activation_energy = self._compute_activation_energy(features)
        
        # 对每个簇内的特征图进行加权平均
        for i, label in enumerate(cluster_labels):
            fused_maps[label] += features[i] * activation_energy[i]
            counts[label] += activation_energy[i]
        
        # 避免除以零
        counts = torch.clamp(counts, min=1e-8)
        
        # 计算加权平均值
        for k in range(self.num_clusters):
            fused_maps[k] = fused_maps[k] / counts[k]
        
        return fused_maps
    
    def generate_heatmap(self, img_path, class_idx=None):
        """
        生成热力图
        
        参数:
        img_path: 输入图像路径
        class_idx: 目标类别索引，如果为None则使用预测的最高概率类别
        
        返回:
        heatmap: 热力图
        pred_class: 预测的类别
        cluster_labels: 聚类标签
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
        
        # 获取特征图
        feature_maps = self.feature_maps.detach()[0]  # [C, H, W]
        
        # 计算频域特征
        freq_features = self._fft_transform(feature_maps)

        # 如果设置为自动确定聚类数量
        if self.num_clusters == 'auto':
            self.num_clusters = self._find_optimal_clusters(
                freq_features, 
                min_clusters=2, 
                max_clusters=10, 
                method=self.cluster_method
            )
            print(f"Automatically determined optimal number of clusters: {self.num_clusters}")
        
        # 使用GMM对频域特征聚类
        cluster_labels = self._cluster_feature_maps_gmm(feature_maps, freq_features)
        
        # 融合特征图
        fused_maps = self._fuse_feature_maps(feature_maps, cluster_labels)
        
        # 计算每个簇的权重
        cluster_weights = self._compute_cluster_weights(feature_maps, cluster_labels)
        
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
        
        return heatmap, class_idx, cluster_labels
    
    def visualize(self, img_path, save_path=None, class_idx=None, alpha=0.5, cmap='jet'):
        """
        可视化热力图叠加在原始图像上
        
        参数:
        img_path: 输入图像路径
        save_path: 保存路径，如果为None则不保存
        class_idx: 目标类别索引
        alpha: 热力图透明度
        cmap: 颜色映射
        
        返回:
        superimposed_img: 叠加后的图像
        pred_class: 预测的类别
        """
        heatmap, pred_class, cluster_labels = self.generate_heatmap(img_path, class_idx)
        
        # 加载原始图像
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # 将热力图调整为与原始图像相同的大小
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), getattr(cv2, f'COLORMAP_{cmap.upper()}'))
        
        # 叠加热力图和原始图像
        superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)
        
        # 保存结果
        if save_path is not None:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            cv2.imwrite(save_path, superimposed_img)
            print(f"Heatmap saved to {save_path}")
        
        return superimposed_img, pred_class
    
    def visualize_detail(self, img_path, save_dir='./result_2/', class_idx=None):
        """
        详细可视化，包括原始图像、热力图以及各个聚类簇
        
        参数:
        img_path: 输入图像路径
        save_dir: 保存目录
        class_idx: 目标类别索引
        
        返回:
        None
        """
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成热力图
        heatmap, pred_class, cluster_labels = self.generate_heatmap(img_path, class_idx)
        
        # 加载原始图像
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (224, 224))
        
        # 获取特征图和融合后的特征图
        feature_maps = self.feature_maps.detach()[0]  # [C, H, W]
        fused_maps = self._fuse_feature_maps(feature_maps, cluster_labels)
        cluster_weights = self._compute_cluster_weights(feature_maps, cluster_labels)
        
        # 应用颜色映射到主热力图
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # 叠加热力图和原始图像
        alpha = 0.5
        superimposed_main = heatmap_colored * alpha + original_img * (1 - alpha)
        superimposed_main = np.uint8(superimposed_main)
        
        # 保存主热力图
        main_heatmap_path = os.path.join(save_dir, f"gmm_actE_heatmap_{os.path.basename(img_path)}")
        cv2.imwrite(main_heatmap_path, superimposed_main)
        print(f"Main heatmap saved to {main_heatmap_path}")
        
        # 创建详细可视化图
        plt.figure(figsize=(15, 10))
        
        # 原始图像
        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.axis('off')
        
        # 完整热力图
        plt.subplot(2, 3, 2)
        plt.title(f'GMM+ActE Heatmap (Class: {pred_class})')
        superimposed_rgb = cv2.cvtColor(superimposed_main, cv2.COLOR_BGR2RGB)
        plt.imshow(superimposed_rgb)
        plt.axis('off')
        
        # 各聚类簇热力图
        n_clusters_to_show = min(4, self.num_clusters)  # 最多显示4个簇
        
        # 按权重排序
        sorted_indices = torch.argsort(cluster_weights, descending=True).cpu().numpy()
        
        for i in range(n_clusters_to_show):
            k = sorted_indices[i]
            
            # 获取该簇的特征图
            cluster_map = fused_maps[k].cpu().numpy()
            cluster_map = np.maximum(cluster_map, 0)  # ReLU
            
            # 归一化
            if np.max(cluster_map) > 0:
                cluster_map = cluster_map / np.max(cluster_map)
            
            # 调整大小
            cluster_map = cv2.resize(cluster_map, (original_img.shape[1], original_img.shape[0]))
            
            # 应用颜色映射
            cluster_heatmap = cv2.applyColorMap(np.uint8(255 * cluster_map), cv2.COLORMAP_JET)
            
            # 叠加热力图和原始图像
            cluster_superimposed = cluster_heatmap * alpha + original_img * (1 - alpha)
            cluster_superimposed = np.uint8(cluster_superimposed)
            cluster_superimposed_rgb = cv2.cvtColor(cluster_superimposed, cv2.COLOR_BGR2RGB)
            
            plt.subplot(2, 3, i + 3)
            plt.title(f'Cluster {k+1} (Weight: {cluster_weights[k]:.3f})')
            plt.imshow(cluster_superimposed_rgb)
            plt.axis('off')
        
        plt.tight_layout()
        detail_path = os.path.join(save_dir, f"gmm_actE_detail_{os.path.basename(img_path).split('.')[0]}.png")
        plt.savefig(detail_path)
        plt.show()
        print(f"Detailed visualization saved to {detail_path}")

# 测试函数
def test_gmm_activation_energy(img_path, model_name='resnet50', target_layer='layer3', num_clusters='auto', cluster_method='bic', save_dir='./result_2/'):
    """
    测试GMM+激活能量解释算法
    
    参数:
    img_path: 测试图像路径
    model_name: 模型名称
    target_layer: 目标层
    num_clusters: 聚类数量
    save_dir: 保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载预训练模型
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    # elif model_name == 'resnet101':
    #     model = models.resnet101(pretrained=True)
    # elif model_name == 'vgg16':
    #     model = models.vgg16(pretrained=True)
    #     if target_layer == 'layer3':
    #         target_layer = 'features.28'
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.eval()
    print(f"Model {model_name} initialized successfully.")
    
    # 初始化解释器
    interpreter = GMMActivationEnergyInterpret(model, target_layer, num_clusters)
    print(f"GMMActivationEnergyInterpret initialized with {num_clusters} clusters.")
    
    # 可视化热力图
    save_path = os.path.join(save_dir, f"gmm_actE_heatmap_{os.path.basename(img_path)}")
    superimposed_img, pred_class = interpreter.visualize(img_path, save_path=save_path)
    
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
    plt.title(f'GMM+ActE Heatmap (Class: {pred_class})')
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    plt.imshow(superimposed_img_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gmm_actE_result_{os.path.basename(img_path).split('.')[0]}.png"))
    plt.show()
    
    # 显示详细结果
    interpreter.visualize_detail(img_path, save_dir=save_dir)
    
    return interpreter

# 比较不同层的GMM+激活能量结果
def compare_layers(img_path, save_dir='./result_2/layers'):
    """比较不同层的GMM+激活能量效果"""
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    
    # 加载原始图像
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 10))
    
    # 显示原始图像
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_img_rgb)
    plt.axis('off')
    
    # 比较不同层
    for i, layer in enumerate(layers):
        # 加载预训练的ResNet50模型
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # 初始化解释器
        interpreter = GMMActivationEnergyInterpret(model, layer, num_clusters=5)
        
        # 生成热力图
        save_path = os.path.join(save_dir, f"gmm_actE_{layer}_{os.path.basename(img_path)}")
        superimposed_img, pred_class = interpreter.visualize(img_path, save_path=save_path)
        
        # 显示结果
        plt.subplot(2, 3, i + 2)
        plt.title(f'{layer}')
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        plt.imshow(superimposed_img_rgb)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"layer_comparison_{os.path.basename(img_path).split('.')[0]}.png"))
    plt.show()
    print(f"Layer comparison saved successfully.")

# 主函数
if __name__ == "__main__":
    # 测试GMM+激活能量方法
    img_path = '../input/treefrog.jpg'
    
    # 使用GMM聚类和激活能量作为权重
    interpreter = test_gmm_activation_energy(
        img_path, 
        model_name='resnet50',
        target_layer='layer4',
        num_clusters='auto', 
        cluster_method='bic',
        save_dir='./result_2/'
    )
    
    # 比较不同层的结果
    compare_layers(img_path)