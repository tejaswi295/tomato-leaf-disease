import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


class ImageInference:
    """
    Utility class for running inference with the trained models.
    """
    
    def __init__(self, classifier_path=None, generator_path=None, device=None):
        """
        Initialize the inference engine.
        
        Args:
            classifier_path: Path to trained classifier model
            generator_path: Path to trained generator model
            device: torch device ('cuda' or 'cpu')
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = None
        self.generator = None
        
        if classifier_path and os.path.exists(classifier_path):
            from classifier import TomatoDiseaseClassifier
            
            self.classifier = TomatoDiseaseClassifier(num_classes=10, model_name='efficientnet')
            
            print(f"Loading EfficientNet_B0 fine-tuned checkpoint from {classifier_path}...")
            state_dict = torch.load(classifier_path, map_location=self.device, weights_only=True)
            self.classifier.load_state_dict(state_dict)
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()
            
        if generator_path and os.path.exists(generator_path):
            from generator import Generator
            self.generator = Generator().to(self.device)
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            self.generator.eval()
    
    def classify_image(self, image_path, class_names=None):
        """
        Classify a single image as disease or not.
        
        Args:
            image_path: Path to image file
            class_names: List of class names
            
        Returns:
            Dictionary with prediction and confidence
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        if class_names is None:
            class_names = [
                'Bacterial spot', 'Early blight', 'Late blight', 'Leaf Mold',
                'Septoria leaf spot', 'Spider mites', 'Target Spot',
                'Yellow Leaf Curl Virus', 'Mosaic virus', 'Healthy'
            ]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return {
            'class': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {name: prob for name, prob in zip(class_names, probabilities[0].cpu().numpy())}
        }
    
    def generate_images(self, num_images=4, latent_dim=100):
        """
        Generate synthetic tomato leaf disease images.
        
        Args:
            num_images: Number of images to generate
            latent_dim: Dimension of latent vector
            
        Returns:
            Generated images as numpy array
        """
        if self.generator is None:
            raise ValueError("Generator not loaded")
        
        noise = torch.randn(num_images, latent_dim, device=self.device)
        
        with torch.no_grad():
            generated = self.generator(noise)
        
        # Denormalize from [-1, 1] to [0, 1]
        generated = (generated + 1) / 2
        generated = torch.clamp(generated, 0, 1)
        
        # Convert to numpy
        images_np = generated.cpu().numpy()
        images_np = np.transpose(images_np, (0, 2, 3, 1))
        
        return images_np

    def generate_gradcam(self, image_path):
        """
        Generate Grad-CAM heatmap overlaid on the original image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Numpy array of the overlay image (RGB, uint8)
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(original_image).unsqueeze(0).to(self.device).requires_grad_(True)
        
        # Target the last convolutional layer of EfficientNet
        target_layer = self.classifier.backbone.features[-1]
        
        features = []
        gradients = []
        
        def forward_hook(module, input, output):
            features.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            
        if hasattr(target_layer, 'register_full_backward_hook'):
            handle_backward = target_layer.register_full_backward_hook(backward_hook)
        else:
            handle_backward = target_layer.register_backward_hook(backward_hook)
            
        handle_forward = target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        self.classifier.eval()
        output = self.classifier(image_tensor)
        target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.classifier.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        if not gradients or not features:
            print("Warning: Hook failed to capture features/gradients.")
            return np.array(original_image.resize((224, 224)))
            
        grads = gradients[0].cpu().data.numpy()[0]
        fmap = features[0].cpu().data.numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(grads, axis=(1, 2))
        
        # Weighted combination of feature maps
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # Resize using PIL
        cam_img = Image.fromarray(cam)
        try:
            cam_resized = np.array(cam_img.resize((224, 224), Image.Resampling.BILINEAR))
        except AttributeError:
            # Fallback for older PIL versions
            cam_resized = np.array(cam_img.resize((224, 224), Image.BILINEAR))
        
        # Apply colormap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('jet')
        heatmap = cmap(cam_resized)[..., :3]  # RGBA to RGB
        
        # Overlay on original image
        img_np = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0
        
        overlay = heatmap * 0.4 + img_np * 0.6
        overlay = np.uint8(255 * overlay)
        
        return overlay


def load_image_batch(image_dir, image_size=128, normalize=True):
    """
    Load a batch of images from a directory.
    
    Args:
        image_dir: Directory containing images
        image_size: Size to resize images to
        normalize: Whether to normalize to [-1, 1]
        
    Returns:
        Tensor of shape (batch_size, 3, image_size, image_size)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    images = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if os.path.isfile(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                images.append(image)
            except:
                print(f"Failed to load {img_path}")
                continue
    
    if images:
        return torch.stack(images)
    else:
        return None


def save_tensor_images(tensor_images, output_dir, base_name='image'):
    """
    Save tensor images as PNG files.
    
    Args:
        tensor_images: Tensor of shape (batch_size, 3, height, width) with values in [0, 1]
        output_dir: Directory to save images
        base_name: Base name for saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img_tensor in enumerate(tensor_images):
        # Convert tensor to numpy
        img_np = img_tensor.cpu().numpy()
        
        # Handle both normalized and unnormalized inputs
        if img_np.min() >= -1 and img_np.max() <= 1:
            # If values are in [-1, 1], denormalize to [0, 1]
            img_np = (img_np + 1) / 2
        
        # Transpose from CHW to HWC
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # Ensure values are in [0, 1]
        img_np = np.clip(img_np, 0, 1)
        
        # Convert to uint8
        img_np = (img_np * 255).astype(np.uint8)
        
        # Save
        img = Image.fromarray(img_np)
        save_path = os.path.join(output_dir, f'{base_name}_{idx:04d}.png')
        img.save(save_path)
        print(f"Saved: {save_path}")


def get_model_summary(model):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\n{model}")
