"""
Quick test script to verify all components are working correctly.
Run: python test.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from generator import Generator
        from discriminator import Discriminator
        from classifier import TomatoDiseaseClassifier
        from utils import ImageInference, get_model_summary
        import config
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_generator():
    """Test generator model."""
    print("\nTesting Generator...")
    try:
        from generator import Generator
        device = torch.device('cpu')
        gen = Generator().to(device)
        
        # Test forward pass
        noise = torch.randn(4, 100, device=device)
        output = gen(noise)
        
        assert output.shape == (4, 3, 128, 128), f"Expected shape (4, 3, 128, 128), got {output.shape}"
        print(f"✓ Generator output shape: {output.shape}")
        
        # Test parameter count
        params = sum(p.numel() for p in gen.parameters())
        print(f"✓ Generator parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Generator test failed: {e}")
        return False


def test_discriminator():
    """Test discriminator model."""
    print("\nTesting Discriminator...")
    try:
        from discriminator import Discriminator
        device = torch.device('cpu')
        disc = Discriminator().to(device)
        
        # Test forward pass
        images = torch.randn(4, 3, 128, 128, device=device)
        output = disc(images)
        
        assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"
        print(f"✓ Discriminator output shape: {output.shape}")
        
        # Test parameter count
        params = sum(p.numel() for p in disc.parameters())
        print(f"✓ Discriminator parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Discriminator test failed: {e}")
        return False


def test_classifier():
    """Test classifier model."""
    print("\nTesting Classifier...")
    try:
        from classifier import TomatoDiseaseClassifier
        device = torch.device('cpu')
        clf = TomatoDiseaseClassifier(num_classes=3).to(device)
        
        # Test forward pass
        images = torch.randn(4, 3, 224, 224, device=device)
        output = clf(images)
        
        assert output.shape == (4, 3), f"Expected shape (4, 3), got {output.shape}"
        print(f"✓ Classifier output shape: {output.shape}")
        
        # Test parameter count
        params = sum(p.numel() for p in clf.parameters())
        print(f"✓ Classifier parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Classifier test failed: {e}")
        return False


def test_dcgan_training_step():
    """Test a single training step of DCGAN."""
    print("\nTesting DCGAN training step...")
    try:
        from generator import Generator
        from discriminator import Discriminator
        
        device = torch.device('cpu')
        gen = Generator().to(device)
        disc = Discriminator().to(device)
        
        # Create dummy batch
        real_images = torch.randn(4, 3, 128, 128, device=device)
        noise = torch.randn(4, 100, device=device)
        
        # Forward pass
        fake_images = gen(noise)
        real_output = disc(real_images)
        fake_output = disc(fake_images.detach())
        
        # Loss computation
        criterion = nn.BCELoss()
        real_labels = torch.ones(4, 1, device=device)
        fake_labels = torch.zeros(4, 1, device=device)
        
        d_real_loss = criterion(real_output, real_labels)
        d_fake_loss = criterion(fake_output, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        
        print(f"✓ Discriminator loss: {d_loss.item():.4f}")
        
        # Generator loss
        fake_output = disc(fake_images)
        g_loss = criterion(fake_output, real_labels)
        print(f"✓ Generator loss: {g_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ DCGAN training step failed: {e}")
        return False


def test_cuda_availability():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ CUDA is not available, will use CPU")
    return True


def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    required_dirs = [
        './data',
        './output',
        './checkpoints',
        './classifier_checkpoints'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (will be created during training)")
            all_exist = False
    
    return True  # Don't fail if dirs don't exist yet


def main():
    print("="*60)
    print("DCGAN Project - Component Test Suite")
    print("="*60)
    
    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("Imports", test_imports),
        ("Generator Model", test_generator),
        ("Discriminator Model", test_discriminator),
        ("Classifier Model", test_classifier),
        ("DCGAN Training Step", test_dcgan_training_step),
        ("Directory Structure", test_directory_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to train.")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
