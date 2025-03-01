import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import sys
from statistics import mode
from torch.autograd import Variable

# Import the GMMActivationEnergyInterpret class
from ver2 import GMMActivationEnergyInterpret

# class ScoreCAMGenerator:
#     """Score-weighted Class Activation Map generator"""
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.feature_maps = None
        
#         # Register hook
#         self._register_hooks()
        
#     def _register_hooks(self):
#         """Register forward hook to capture feature maps"""
#         def forward_hook(module, input, output):
#             self.feature_maps = output
            
#         # Get target layer
#         for name, module in self.model.named_modules():
#             if name == self.target_layer:
#                 self.target_module = module
#                 module.register_forward_hook(forward_hook)
#                 break
    
#     def generate_scorecam(self, img_path, class_idx=None):
#         """Generate Score-CAM for the given image and class"""
#         # Load and preprocess image
#         img = Image.open(img_path).convert('RGB')
#         preprocess = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         input_tensor = preprocess(img).unsqueeze(0)
        
#         # Forward pass to get feature maps
#         self.model.eval()
#         with torch.no_grad():
#             logits = self.model(input_tensor)
        
#         # If class index not specified, use predicted class
#         if class_idx is None:
#             pred = logits.argmax(dim=1)
#             class_idx = pred.item()
        
#         # Get raw image (without normalization) for masking
#         raw_preprocess = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#         ])
#         raw_input = raw_preprocess(img).unsqueeze(0)
        
#         # Get base score for the target class
#         base_score = torch.nn.functional.softmax(logits, dim=1)[0, class_idx].item()
        
#         # Get activation maps from the target layer
#         activation_maps = self.feature_maps.detach()
        
#         # Get the number of channels in the feature maps
#         num_channels = activation_maps.shape[1]
        
#         # Prepare normalized activation maps
#         normalized_maps = []
#         for i in range(num_channels):
#             # Extract single channel activation map
#             channel_map = activation_maps[0, i, :, :].unsqueeze(0).unsqueeze(0)
            
#             # Upsample to input size
#             upsampled_map = torch.nn.functional.interpolate(
#                 channel_map, 
#                 size=(224, 224), 
#                 mode='bilinear', 
#                 align_corners=False
#             ).squeeze()
            
#             # Normalize map to [0, 1]
#             if upsampled_map.max() != upsampled_map.min():
#                 normalized_map = (upsampled_map - upsampled_map.min()) / (upsampled_map.max() - upsampled_map.min())
#             else:
#                 normalized_map = torch.zeros_like(upsampled_map)
                
#             normalized_maps.append(normalized_map)
        
#         # Calculate weights for each activation map using score
#         weights = []
#         for norm_map in normalized_maps:
#             # Apply map as mask to the image
#             masked_input = raw_input.clone()
#             for c in range(3):  # For each color channel
#                 masked_input[0, c, :, :] *= norm_map
            
#             # Normalize the masked input
#             masked_input_normalized = transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225]
#             )(masked_input[0]).unsqueeze(0)
            
#             # Forward pass with masked input
#             with torch.no_grad():
#                 mask_logits = self.model(masked_input_normalized)
            
#             # Get score for target class
#             mask_score = torch.nn.functional.softmax(mask_logits, dim=1)[0, class_idx].item()
            
#             # Calculate weight as the difference from base score
#             weight = max(0, mask_score)
#             weights.append(weight)
        
#         # Normalize weights
#         if sum(weights) != 0:
#             weights = [w / sum(weights) for w in weights]
        
#         # Generate Score-CAM by weighted sum of activation maps
#         scorecam = torch.zeros_like(normalized_maps[0])
#         for i, (norm_map, weight) in enumerate(zip(normalized_maps, weights)):
#             scorecam += norm_map * weight
        
#         # Convert to numpy and ensure proper normalization
#         scorecam = scorecam.cpu().numpy()
#         scorecam = np.maximum(scorecam, 0)
#         if np.max(scorecam) > 0:
#             scorecam = scorecam / np.max(scorecam)
        
#         return scorecam, class_idx
    
#     def visualize_scorecam(self, img_path, class_idx=None, save_path=None):
#         """Visualize Score-CAM heatmap overlaid on the original image"""
#         # Generate Score-CAM
#         scorecam, class_idx = self.generate_scorecam(img_path, class_idx)
        
#         # Load original image
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (224, 224))
        
#         # Convert heatmap to color map
#         heatmap = cv2.applyColorMap(np.uint8(255 * scorecam), cv2.COLORMAP_JET)
        
#         # Overlay heatmap on original image
#         overlaid = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        
#         # Display or save the visualization
#         if save_path:
#             cv2.imwrite(save_path, overlaid)
#             print(f"Visualization saved to {save_path}")
        
#         return overlaid, class_idx

class GradCAMPlusPlusGenerator:
    """Grad-CAM++ generator"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.target_module = module
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def generate_gradcam_plus_plus(self, img_path, class_idx=None):
        """Generate Grad-CAM++ for the given image and class"""
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)
        
        # Forward pass
        self.model.eval()
        logits = self.model(input_tensor)
        
        # If class index not specified, use predicted class
        if class_idx is None:
            pred = logits.argmax(dim=1)
            class_idx = pred.item()
        
        # Get class score
        score = logits[0, class_idx]
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Backward pass for the specified class
        score.backward(retain_graph=True)
        
        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Compute the alpha terms (second-order gradients)
        alpha_c = gradients * activations
        alpha_c2 = gradients ** 2 * activations
        
        # Compute the Grad-CAM++ map
        gradcampp = torch.sum(alpha_c, dim=1) + 0.5 * torch.sum(alpha_c2, dim=1)
        
        # Squeeze and move to CPU for further processing
        gradcampp = gradcampp.detach().cpu().numpy().squeeze()
        
        # Apply ReLU to retain positive values
        gradcampp = np.maximum(gradcampp, 0)
        if np.max(gradcampp) > 0:
            gradcampp = gradcampp / np.max(gradcampp)
        
        return gradcampp, class_idx


class GradCAMGenerator:
    """Gradient-weighted Class Activation Map generator"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.target_module = module
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
                
    def generate_gradcam(self, img_path, class_idx=None):
        """Generate Grad-CAM for the given image and class"""
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)
        
        # Forward pass
        self.model.eval()
        logits = self.model(input_tensor)
        
        # If class index not specified, use predicted class
        if class_idx is None:
            pred = logits.argmax(dim=1)
            class_idx = pred.item()
        
        # Get class score
        score = logits[0, class_idx]
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Backward pass for the specified class
        score.backward()
        
        # Generate Grad-CAM
        # Compute the mean gradient for each channel
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weight the feature maps by their corresponding gradients
        weighted_maps = self.feature_maps * gradients
        
        # Sum over the channels to get the heatmap
        gradcam = torch.sum(weighted_maps, dim=1).squeeze()
        gradcam = gradcam.detach().cpu().numpy()
        
        # ReLU and normalize
        gradcam = np.maximum(gradcam, 0)
        if np.max(gradcam) > 0:
            gradcam = gradcam / np.max(gradcam)
        
        return gradcam, class_idx

class FaithfulnessEvaluator:
    """Evaluates the faithfulness of interpretability methods using Average Drop"""
    def __init__(self, model_name='resnet50', target_layer='layer4', method='gmm_actE', threshold=0.5):
        """
        Initialize evaluator
        
        Args:
            model_name: Name of the model to evaluate
            target_layer: Target layer for interpretability method
            method: Interpretability method ('gmm_actE' or 'gradcam')
            threshold: Threshold for saliency map (pixels above this value are preserved)
        """
        self.model_name = model_name
        self.target_layer = target_layer
        self.method = method
        self.threshold = threshold
        
        # Load model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.eval()
        
        # Create interpretability method
        if method == 'gmm_actE':
            self.interpreter = GMMActivationEnergyInterpret(
                self.model, 
                target_layer, 
                num_clusters='auto', 
                cluster_method='bic'
            )
        elif method == 'gradcam':
            self.interpreter = GradCAMGenerator(self.model, target_layer)
        elif method == 'gradcam++':
            self.interpreter = GradCAMPlusPlusGenerator(self.model, target_layer)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Define preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # For storing results - use lists to accumulate results incrementally
        self.results = {
            'original_scores': [],
            'masked_scores': [],
            'class_indices': [],
            'drops': []
        }
        
    def load_image(self, img_path):
        """Load and preprocess image"""
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        
    def predict(self, img):
        """Get model prediction and score for an image"""
        if img is None:
            return None, 0.0
            
        input_tensor = self.preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get predicted class and score
        prob = torch.nn.functional.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_score = prob[0, pred_class].item()
        
        return pred_class, pred_score
    
    def generate_saliency_map(self, img_path, class_idx=None):
        """Generate saliency map using the specified method"""
        if self.method == 'gmm_actE':
            heatmap, class_idx, _ = self.interpreter.generate_heatmap(img_path, class_idx)
            return heatmap, class_idx
        elif self.method == 'gradcam':
            gradcam, class_idx = self.interpreter.generate_gradcam(img_path, class_idx)
            return gradcam, class_idx
        elif self.method == 'gradcam++':
            gradcam, class_idx = self.interpreter.generate_gradcam_plus_plus(img_path, class_idx)
            return gradcam, class_idx
    
    def create_masked_image(self, img_path, saliency_map):
        """Create masked image based on saliency map"""
        # Load original image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path} with OpenCV")
            return None
            
        img = cv2.resize(img, (224, 224))
        
        # Resize saliency map if needed
        saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))
        
        # Create binary mask based on threshold
        mask = (saliency_map > self.threshold).astype(np.float32)
        
        # Expand mask to 3 channels
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
        
        # Apply mask to image
        masked_img = img * mask
        
        # Convert to PIL image for model input
        masked_img = cv2.cvtColor(masked_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        masked_img_pil = Image.fromarray(masked_img)
        
        return masked_img_pil
    
    def evaluate_single_image(self, img_path, save_dir=None):
        """Evaluate faithfulness for a single image"""
        try:
            # Get original prediction
            original_img = self.load_image(img_path)
            if original_img is None:
                return None, None, None, None
                
            pred_class, original_score = self.predict(original_img)
            
            # Generate saliency map
            saliency_map, pred_class = self.generate_saliency_map(img_path, pred_class)
            
            # Create masked image
            masked_img = self.create_masked_image(img_path, saliency_map)
            if masked_img is None:
                return None, None, None, None
                
            # Get prediction on masked image
            _, masked_score = self.predict(masked_img)
            
            # Calculate drop
            drop = max(0, (original_score - masked_score) / original_score) if original_score > 0 else 0
            
            # Save visualization if requested
            if save_dir:
                self._save_visualization(img_path, original_img, saliency_map, masked_img, 
                                        original_score, masked_score, drop, save_dir)
            
            # Cleanup to free memory
            del original_img
            del masked_img
            
            return original_score, masked_score, pred_class, drop
            
        except Exception as e:
            print(f"Error evaluating {img_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def _save_visualization(self, img_path, original_img, saliency_map, masked_img, 
                           original_score, masked_score, drop, save_dir):
        """Save visualization of single image evaluation"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        try:
            # Original image for visualization
            original_np = np.array(original_img)
            original_np = cv2.resize(original_np, (224, 224))
            
            # Saliency map visualization
            saliency_map_resized = cv2.resize(saliency_map, (224, 224))
            
            # Masked image visualization
            masked_img_np = np.array(masked_img)
            
            # Create visualization grid
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title(f'Original (Score: {original_score:.4f})')
            plt.imshow(original_np)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title(f'Saliency Map (Method: {self.method})')
            plt.imshow(saliency_map_resized, cmap='jet')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title(f'Masked (Score: {masked_score:.4f}, Drop: {drop:.4f})')
            plt.imshow(masked_img_np)
            plt.axis('off')
            
            plt.tight_layout()
            basename = os.path.basename(img_path).split('.')[0]
            save_path = os.path.join(save_dir, f"{self.method}_{basename}_eval.png")
            plt.savefig(save_path)
            plt.close()
            
            # Explicitly delete objects to free memory
            del original_np, masked_img_np
            
        except Exception as e:
            print(f"Error saving visualization for {img_path}: {e}")

    def evaluate_dataset(self, data_dir, save_dir=None, sample_size=None, visualize_samples=10, 
                         batch_reporting=10):
        """
        Evaluate faithfulness on a dataset, loading images one by one.
        
        Args:
            data_dir: Directory containing evaluation images
            save_dir: Directory to save visualizations
            sample_size: Number of images to evaluate (None for all)
            visualize_samples: Number of samples to visualize
            batch_reporting: Report progress after processing this many images
        """
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Get list of image files (don't load them)
        image_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        # Limit sample size if specified
        if sample_size is not None:
            sample_size = min(sample_size, len(image_files))
            # Randomly sample to avoid bias
            np.random.seed(42)  # For reproducibility
            image_files = np.random.choice(image_files, sample_size, replace=False).tolist()
        
        # Reset results
        self.results = {
            'original_scores': [],
            'masked_scores': [],
            'class_indices': [],
            'drops': []
        }
        
        # Prepare for evaluation
        total_images = len(image_files)
        successful_evaluations = 0
        print(f"Evaluating {total_images} images using {self.method} method...")
        
        # Track images to visualize
        vis_indices = np.linspace(0, total_images-1, visualize_samples, dtype=int)
        
        # Process images in batches to report progress
        for i, img_path in enumerate(tqdm(image_files)):
            # Determine if this image should be visualized
            vis_dir = save_dir if i in vis_indices else None
            
            # Evaluate single image
            original_score, masked_score, pred_class, drop = self.evaluate_single_image(img_path, vis_dir)
            
            # Store results if evaluation was successful
            if original_score is not None:
                self.results['original_scores'].append(original_score)
                self.results['masked_scores'].append(masked_score)
                self.results['class_indices'].append(pred_class)
                self.results['drops'].append(drop)
                successful_evaluations += 1
            
            # Report batch progress and intermediate results
            if (i + 1) % batch_reporting == 0 or i == total_images - 1:
                # Calculate intermediate results
                if successful_evaluations > 0:
                    avg_drop = np.mean(self.results['drops'])
                    avg_original = np.mean(self.results['original_scores'])
                    avg_masked = np.mean(self.results['masked_scores'])
                    
                    print(f"Processed {i+1}/{total_images} images. "
                          f"Success: {successful_evaluations}. "
                          f"Current Avg Drop: {avg_drop:.4f}")
                else:
                    print(f"Processed {i+1}/{total_images} images. No successful evaluations yet.")
            
            # Force garbage collection to free memory
            if (i + 1) % 20 == 0:
                import gc
                gc.collect()
        
        # Calculate final results
        if successful_evaluations > 0:
            avg_drop = np.mean(self.results['drops'])
            avg_original_score = np.mean(self.results['original_scores'])
            avg_masked_score = np.mean(self.results['masked_scores'])
            
            # Print results
            print(f"\nEvaluation Results for {self.method}:")
            print(f"Successfully evaluated {successful_evaluations}/{total_images} images")
            print(f"Average Original Score: {avg_original_score:.4f}")
            print(f"Average Masked Score: {avg_masked_score:.4f}")
            print(f"Average Drop: {avg_drop:.4f}")
            
            # Save summary visualization
            if save_dir:
                self._save_summary_visualization(save_dir, successful_evaluations, total_images)
            
            return avg_drop, avg_original_score, avg_masked_score
        else:
            print("\nNo successful evaluations. Check image loading and model setup.")
            return 0.0, 0.0, 0.0
    
    def _save_summary_visualization(self, save_dir, successful_evals, total_images):
        """Save summary visualization of results"""
        if len(self.results['drops']) == 0:
            print("No results to visualize.")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Drop distribution
        plt.subplot(2, 2, 1)
        plt.hist(self.results['drops'], bins=20, alpha=0.7, color='blue')
        plt.axvline(np.mean(self.results['drops']), color='red', linestyle='dashed', linewidth=1)
        plt.title(f'Drop Distribution (Avg: {np.mean(self.results["drops"]):.4f})')
        plt.xlabel('Drop')
        plt.ylabel('Count')
        
        # Original vs Masked scores
        plt.subplot(2, 2, 2)
        plt.scatter(self.results['original_scores'], self.results['masked_scores'], alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # y=x line
        plt.title('Original vs Masked Scores')
        plt.xlabel('Original Score')
        plt.ylabel('Masked Score')
        plt.axis('equal')
        plt.axis([0, 1, 0, 1])
        
        # Original score distribution
        plt.subplot(2, 2, 3)
        plt.hist(self.results['original_scores'], bins=20, alpha=0.7, color='green')
        plt.axvline(np.mean(self.results['original_scores']), color='red', linestyle='dashed', linewidth=1)
        plt.title(f'Original Score Distribution (Avg: {np.mean(self.results["original_scores"]):.4f})')
        plt.xlabel('Score')
        plt.ylabel('Count')
        
        # Masked score distribution
        plt.subplot(2, 2, 4)
        plt.hist(self.results['masked_scores'], bins=20, alpha=0.7, color='orange')
        plt.axvline(np.mean(self.results['masked_scores']), color='red', linestyle='dashed', linewidth=1)
        plt.title(f'Masked Score Distribution (Avg: {np.mean(self.results["masked_scores"]):.4f})')
        plt.xlabel('Score')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{self.method}_summary.png"))
        plt.close()
        
        # Save numeric results to file
        with open(os.path.join(save_dir, f"{self.method}_results.txt"), 'w') as f:
            f.write(f"Method: {self.method}\n")
            f.write(f"Threshold: {self.threshold}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Target Layer: {self.target_layer}\n")
            f.write(f"Number of successful evaluations: {successful_evals} / {total_images}\n")
            f.write(f"Average Original Score: {np.mean(self.results['original_scores']):.4f}\n")
            f.write(f"Average Masked Score: {np.mean(self.results['masked_scores']):.4f}\n")
            f.write(f"Average Drop: {np.mean(self.results['drops']):.4f}\n")
            f.write(f"Median Drop: {np.median(self.results['drops']):.4f}\n")
            f.write(f"Standard Deviation of Drop: {np.std(self.results['drops']):.4f}\n")

def compare_methods(data_dir, save_dir='../evaluation_results/', sample_size=50, batch_reporting=10):
    """Compare different interpretability methods with memory-efficient processing"""
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Configuration
    methods = ['gradcam']  # Can add 'gmm_actE', 'gradcam++' as needed
    thresholds = [0.5]  # Can be expanded to test multiple thresholds
    
    # Store results
    results = []
    
    # Evaluate each method
    for method in methods:
        for threshold in thresholds:
            print(f"\n=== Evaluating {method} with threshold {threshold} ===")
            
            # Create method-specific save directory
            method_dir = os.path.join(save_dir, f"{method}_thresh{threshold}")
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            
            # Initialize evaluator
            evaluator = FaithfulnessEvaluator(
                model_name='resnet50',
                target_layer='layer4',
                method=method,
                threshold=threshold
            )
            
            # Evaluate
            avg_drop, avg_original, avg_masked = evaluator.evaluate_dataset(
                data_dir=data_dir,
                save_dir=method_dir,
                sample_size=sample_size,
                batch_reporting=batch_reporting
            )
            
            # Store results
            results.append({
                'method': method,
                'threshold': threshold,
                'avg_drop': avg_drop,
                'avg_original': avg_original,
                'avg_masked': avg_masked
            })
            
            # Force garbage collection
            import gc
            gc.collect()
    
    # Visualize comparison if we have results
    if results:
        plt.figure(figsize=(12, 6))
        
        # Bar chart of average drops
        methods_labels = [f"{r['method']} (t={r['threshold']})" for r in results]
        avg_drops = [r['avg_drop'] for r in results]
        
        plt.bar(methods_labels, avg_drops, color=['blue', 'green'])
        
        for i, v in enumerate(avg_drops):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.title('Comparison of Average Drop Across Methods')
        plt.ylabel('Average Drop')
        plt.ylim(0, max(avg_drops) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "method_comparison.png"))
        plt.close()
        
        # Save comparison results to file
        with open(os.path.join(save_dir, "comparison_results.txt"), 'w') as f:
            f.write("Method Comparison Results\n")
            f.write("-----------------------\n\n")
            
            for r in results:
                f.write(f"Method: {r['method']}\n")
                f.write(f"Threshold: {r['threshold']}\n")
                f.write(f"Average Original Score: {r['avg_original']:.4f}\n")
                f.write(f"Average Masked Score: {r['avg_masked']:.4f}\n")
                f.write(f"Average Drop: {r['avg_drop']:.4f}\n")
                f.write("-----------------------\n\n")
    
    return results

if __name__ == "__main__":
    # Set evaluation data directory
    eval_dir = "../testimgs"  # Directory containing ImageNet validation images
    
    # Run evaluation with memory-efficient processing
    print("Starting faithfulness evaluation...")
    
    # Compare methods
    results = compare_methods(
        data_dir=eval_dir,
        save_dir='../evaluation_results/',
        sample_size=100,  # Adjust based on available images and computation resources
        batch_reporting=10  # Report progress every 10 images
    )
    
    print("\nEvaluation complete!")