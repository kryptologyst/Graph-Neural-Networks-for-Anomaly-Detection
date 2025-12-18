#!/usr/bin/env python3
"""Simple test script to verify the installation works."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work."""
    try:
        import torch
        print("✓ PyTorch imported successfully")
        
        import torch_geometric
        print("✓ PyTorch Geometric imported successfully")
        
        from src.utils.device import get_device, set_seed
        print("✓ Device utilities imported successfully")
        
        from src.data.dataset import load_dataset
        print("✓ Dataset utilities imported successfully")
        
        from src.models.anomaly_models import create_model
        print("✓ Model utilities imported successfully")
        
        from src.eval.metrics import AnomalyEvaluator
        print("✓ Evaluation utilities imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_device():
    """Test device detection."""
    try:
        from src.utils.device import get_device, set_seed
        
        device = get_device()
        print(f"✓ Device detected: {device}")
        
        set_seed(42)
        print("✓ Seed setting works")
        
        return True
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    try:
        from src.models.anomaly_models import create_model
        
        model = create_model("gae", in_channels=16, hidden_channels=32, out_channels=16)
        print("✓ GAE model created successfully")
        
        model = create_model("dominant", in_channels=16, hidden_channels=32, out_channels=16)
        print("✓ DOMINANT model created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing GNN Anomaly Detection Installation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Test", test_device),
        ("Model Creation Test", test_model_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
