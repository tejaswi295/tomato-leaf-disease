"""
Simple inference script for disease classification and image generation.
Run: python inference.py --mode classify --image_path <path_to_image>
"""

import argparse
import torch
import os
from pathlib import Path
from utils import ImageInference, save_tensor_images
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Inference with trained models')
    parser.add_argument('--mode', type=str, default='classify', choices=['classify', 'generate'],
                        help='Inference mode: classify or generate')
    parser.add_argument('--image_path', type=str, help='Path to image for classification')
    parser.add_argument('--classifier_path', type=str, default='./classifier_checkpoints/best_classifier.pt',
                        help='Path to trained classifier')
    parser.add_argument('--generator_path', type=str, default='./checkpoints/generator_final.pt',
                        help='Path to trained generator')
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='Output directory for generated images')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize inference
    inference = ImageInference(
        classifier_path=args.classifier_path if os.path.exists(args.classifier_path) else None,
        generator_path=args.generator_path if os.path.exists(args.generator_path) else None,
        device=device
    )
    
    if args.mode == 'classify':
        if not args.image_path:
            print("Error: --image_path required for classification mode")
            return
        
        if not os.path.exists(args.image_path):
            print(f"Error: Image not found at {args.image_path}")
            return
        
        if inference.classifier is None:
            print(f"Error: Classifier not found at {args.classifier_path}")
            return
        
        print(f"\nClassifying: {args.image_path}")
        result = inference.classify_image(args.image_path)
        
        print(f"\nResults:")
        print(f"  Disease: {result['class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"\n  All probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")
    
    elif args.mode == 'generate':
        if inference.generator is None:
            print(f"Error: Generator not found at {args.generator_path}")
            return
        
        print(f"\nGenerating {args.num_images} synthetic images...")
        synthetic_images = inference.generate_images(num_images=args.num_images)
        
        # Save generated images
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        for idx, img in enumerate(synthetic_images):
            img_pil = plt.imshow(img)
            plt.axis('off')
            save_path = os.path.join(args.output_dir, f'generated_{idx:02d}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
            print(f"Saved: {save_path}")
        
        print(f"Generated images saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
