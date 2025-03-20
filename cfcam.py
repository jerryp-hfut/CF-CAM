import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class CFCAM:
    """Cluster Filter Class Activation Mapping (CF-CAM) implementation."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Initialize CFCAM with a model and target layer.

        Args:
            model: Pre-trained deep learning model.
            target_layer: Target convolutional layer for feature extraction.
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks to capture feature maps and gradients."""
        def forward_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.feature_maps = output

        def backward_hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if module == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                print(f"  - Hooks registered for layer: {name}")
                break

    def _compute_dynamic_eps(self, feature_matrix: np.ndarray) -> float:
        """Dynamically compute DBSCAN's eps parameter based on feature statistics."""
        distance_matrix = np.linalg.norm(feature_matrix[:, None] - feature_matrix[None, :], axis=2)
        np.fill_diagonal(distance_matrix, np.inf)
        distances = distance_matrix.flatten()
        eps = np.percentile(distances[distances != np.inf], 25)
        return max(eps, 1e-5)

    def _hierarchical_clustering(self, feature_maps: torch.Tensor) -> tuple:
        """Perform hierarchical clustering: filter important channels, then cluster the rest."""
        num_channels, height, width = feature_maps.shape
        feature_matrix = feature_maps.view(num_channels, -1).cpu().numpy()

        # Step 1: Filter important channels based on L2 norm
        channel_norms = np.linalg.norm(feature_matrix, ord=2, axis=1)
        norm_threshold = np.percentile(channel_norms, 75)
        important_mask = channel_norms >= norm_threshold
        important_indices = np.where(important_mask)[0]
        remaining_indices = np.where(~important_mask)[0]

        # Step 2: DBSCAN clustering on remaining channels
        if len(remaining_indices) > 0:
            remaining_features = feature_matrix[remaining_indices]
            eps = self._compute_dynamic_eps(remaining_features)
            min_samples = max(2, int(len(remaining_indices) * 0.05))
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(remaining_features)
            labels_remaining = db.labels_
            labels = np.full(num_channels, -2, dtype=int)  # -2 for important channels
            labels[remaining_indices] = labels_remaining
            num_clusters = len(np.unique(labels_remaining)) - (1 if -1 in labels_remaining else 0)
        else:
            labels = np.full(num_channels, -2, dtype=int)
            num_clusters = 0

        return labels, num_clusters, important_indices

    def _bilateral_filter(self, gradient: torch.Tensor, sigma_spatial: float = 5.0,
                         sigma_range: float = 0.1) -> torch.Tensor:
        """Apply a differentiable bilateral filter."""
        gradient = gradient.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        height, width = gradient.shape[-2:]

        x, y = torch.meshgrid(torch.arange(height, device=gradient.device),
                              torch.arange(width, device=gradient.device), indexing='ij')
        x = x.float() - height / 2
        y = y.float() - width / 2
        spatial_dist = x**2 + y**2
        spatial_weight = torch.exp(-spatial_dist / (2 * sigma_spatial**2)).unsqueeze(0).unsqueeze(0)

        diff = gradient - gradient.mean()
        range_weight = torch.exp(-(diff**2) / (2 * sigma_range**2))
        weights = spatial_weight * range_weight
        weights = weights / (weights.sum(dim=(-1, -2), keepdim=True) + 1e-6)

        return F.conv2d(gradient, weights, padding='same').squeeze()

    def __call__(self, x: torch.Tensor, class_idx: int = None, noise_level: float = None) -> tuple:
        """Generate Class Activation Map (CAM)."""
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot)

        if self.feature_maps is None or self.gradients is None:
            raise ValueError("Feature maps or gradients not captured.")

        feature_maps = self.feature_maps.detach()[0]
        gradients = self.gradients.detach()[0]
        num_channels, height_prime, width_prime = feature_maps.shape

        cluster_labels, num_clusters, important_indices = self._hierarchical_clustering(feature_maps)
        if len(important_indices) == 0 and num_clusters == 0:
            raise ValueError("No valid clusters or important channels found.")

        valid_indices = np.concatenate([important_indices, np.where(cluster_labels != -1)[0]])
        filtered_gradients = torch.zeros_like(gradients)
        weights = torch.zeros(num_channels, device=x.device)
        valid_mask = torch.zeros(num_channels, dtype=torch.bool, device=x.device)
        valid_mask[valid_indices] = True

        for idx in important_indices:
            filtered_gradients[idx] = gradients[idx]
            weights[idx] = torch.mean(gradients[idx])

        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_gradients = gradients[cluster_indices]
            mean_gradient = torch.mean(cluster_gradients, dim=0)
            filtered_gradient = self._bilateral_filter(mean_gradient)
            for idx in cluster_indices:
                filtered_gradients[idx] = filtered_gradient
                weights[idx] = torch.mean(filtered_gradient)

        if noise_level is not None:
            noise = torch.randn_like(filtered_gradients) * noise_level
            filtered_gradients = torch.where(valid_mask.unsqueeze(1).unsqueeze(2),
                                            filtered_gradients + noise, filtered_gradients)

        valid_weights = weights[valid_mask]
        valid_feature_maps = feature_maps[valid_mask]
        weights = torch.softmax(valid_weights, dim=0).reshape(-1, 1, 1)
        cam = torch.sum(weights * valid_feature_maps, dim=0)
        cam = torch.relu(cam)

        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(x.shape[2], x.shape[3]),
                            mode='bilinear', align_corners=False).squeeze()

        cam_max = torch.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        cam = cam.cpu().numpy()
        return cam, class_idx, logits


def preprocess_image(img_path: str, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    """Preprocess an image and return tensor and original PIL image."""
    try:
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        return input_tensor, image
    except Exception as e:
        raise ValueError(f"Failed to preprocess image {img_path}: {str(e)}")


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a trained ResNet50 model with a custom classifier."""
    try:
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")


def overlay_heatmap(cam: np.ndarray, original_img: Image.Image, alpha: float = 0.4) -> np.ndarray:
    """Overlay heatmap on the original image."""
    cam_resized = cv2.resize(cam, (original_img.width, original_img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(original_img)
    superimposed_img = heatmap * alpha + img_np * (1 - alpha)
    return np.uint8(superimposed_img)


def process_test_folder(test_folder: str, output_folder: str, model: nn.Module,
                       device: torch.device) -> None:
    """Process all images in test folder and save CF-CAM heatmaps."""
    os.makedirs(output_folder, exist_ok=True)
    cfcam = CFCAM(model, model.layer4[-1])

    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"WARNING: No image files found in '{test_folder}'")
        return

    print(f"Starting processing of {len(image_files)} images from '{test_folder}'...")
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_folder, img_file)
        print(f"[{idx}/{len(image_files)}] Processing '{img_file}'")
        try:
            input_tensor, original_img = preprocess_image(img_path, device)
            cam, class_idx, logits = cfcam(input_tensor)

            probs = F.softmax(logits, dim=1)[0]
            confidence = probs[class_idx].item()
            print(f"  - Predicted class: {class_idx}, Confidence: {confidence:.4f}")
            print(f"  - CAM stats: min={cam.min():.4f}, max={cam.max():.4f}, mean={cam.mean():.4f}")

            superimposed_img = overlay_heatmap(cam, original_img)
            output_path = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_cfcam.png")

            plt.figure(figsize=(6, 6))
            plt.imshow(superimposed_img)
            plt.axis('off')
            plt.title(f"CF-CAM: {img_file} (Class: {class_idx}, Conf: {confidence:.2f})")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  - Saved heatmap to '{output_path}'")
        except Exception as e:
            print(f"  - ERROR: Failed to process '{img_file}': {str(e)}")

    print(f"Completed processing. All heatmaps saved to '{output_folder}'")


def main() -> None:
    """Main execution function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_folder = "testimgs"
    output_folder = "cfcam_heatmaps"
    model_path = "best_model.pth"

    if not os.path.exists(test_folder):
        print(f"ERROR: Test folder '{test_folder}' does not exist")
        return
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' does not exist")
        return

    try:
        model = load_model(model_path, device)
        print(f"Model loaded successfully from '{model_path}'")
    except ValueError as e:
        print(f"ERROR: {str(e)}")
        return

    process_test_folder(test_folder, output_folder, model, device)


if __name__ == "__main__":
    main()
