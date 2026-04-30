import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


class ImageInference:
    def __init__(self, classifier_path=None, generator_path=None, device=None):

        # ✅ FORCE CPU (VERY IMPORTANT FOR RENDER)
        self.device = torch.device('cpu')

        self.classifier = None
        self.generator = None
        
        print("MODEL PATH:", classifier_path)
        print("FILE EXISTS:", os.path.exists(classifier_path))

        # ✅ LOAD CLASSIFIER
        if classifier_path and os.path.exists(classifier_path):
            from classifier import TomatoDiseaseClassifier
            
            self.classifier = TomatoDiseaseClassifier(num_classes=10, model_name='efficientnet')
            
            print(f"Loading model from {classifier_path}...")

            # ✅ SAFE LOAD (NO weights_only)
            state_dict = torch.load(classifier_path, map_location=self.device)

            self.classifier.load_state_dict(state_dict)
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()

            print("✅ CLASSIFIER LOADED")
        else:
            print("❌ CLASSIFIER FILE NOT FOUND")

        # (generator unchanged)
        if generator_path and os.path.exists(generator_path):
            from generator import Generator
            self.generator = Generator().to(self.device)
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            self.generator.eval()

    # ------------------ CLASSIFIER ------------------

    def classify_image(self, image_path, class_names=None):
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        if class_names is None:
            class_names = [
                'Bacterial spot', 'Early blight', 'Late blight', 'Leaf Mold',
                'Septoria leaf spot', 'Spider mites', 'Target Spot',
                'Yellow Leaf Curl Virus', 'Mosaic virus', 'Healthy'
            ]
        
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return {
            'class': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                name: float(prob)
                for name, prob in zip(class_names, probabilities[0].cpu().numpy())
            }
        }

    # ------------------ GRADCAM (SAFE VERSION) ------------------

    def generate_gradcam(self, image_path):
        if self.classifier is None:
            raise ValueError("Classifier not loaded")

        try:
            print("🔥 GradCAM started")

            original_image = Image.open(image_path).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            image_tensor = transform(original_image).unsqueeze(0).to(self.device)
            image_tensor.requires_grad_(True)

            # ✅ safer layer (avoids crash)
            target_layer = list(self.classifier.backbone.features.children())[-2]

            features = []
            gradients = []

            def forward_hook(module, input, output):
                features.append(output)

            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])

            handle_forward = target_layer.register_forward_hook(forward_hook)

            if hasattr(target_layer, 'register_full_backward_hook'):
                handle_backward = target_layer.register_full_backward_hook(backward_hook)
            else:
                handle_backward = target_layer.register_backward_hook(backward_hook)

            self.classifier.eval()

            output = self.classifier(image_tensor)
            target_class = torch.argmax(output, dim=1).item()

            self.classifier.zero_grad()
            output[0, target_class].backward()

            handle_forward.remove()
            handle_backward.remove()

            # ✅ fallback safety
            if not gradients or not features:
                print("⚠️ GradCAM fallback")
                return np.array(original_image.resize((224, 224)))

            grads = gradients[0].cpu().numpy()[0]
            fmap = features[0].cpu().numpy()[0]

            weights = np.mean(grads, axis=(1, 2))

            cam = np.zeros(fmap.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * fmap[i]

            cam = np.maximum(cam, 0)

            if np.max(cam) != 0:
                cam = cam / np.max(cam)

            cam = Image.fromarray(cam)
            cam = cam.resize((224, 224))
            cam = np.array(cam)

            # ✅ lightweight heatmap (no matplotlib)
            heatmap = np.stack([cam, np.zeros_like(cam), np.zeros_like(cam)], axis=-1)

            img_np = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0

            overlay = heatmap * 0.4 + img_np * 0.6
            overlay = np.uint8(255 * overlay)

            print("✅ GradCAM success")
            return overlay

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("❌ GradCAM crashed → returning original")

            return np.array(Image.open(image_path).resize((224, 224)))
