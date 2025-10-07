#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import torch


def test_library_availability():
    """Test that DeepGEMM library can be imported and basic info is accessible."""
    print("Testing DeepGEMM library availability...")
    
    try:
        import deep_gemm
        print(f"  ✅ DeepGEMM imported successfully")
        print(f"  📍 Library path: {deep_gemm.__path__}")
        
        # Check basic functions
        functions = [f for f in dir(deep_gemm) if not f.startswith('_')]
        print(f"  📊 Available functions: {len(functions)}")
        print(f"  🔧 Key functions: {[f for f in functions if 'gemm' in f.lower()][:5]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to import DeepGEMM: {e}")
        return False


def test_testing_module():
    """Test that DeepGEMM testing utilities are available."""
    print("Testing DeepGEMM testing module...")
    
    try:
        import deep_gemm.testing
        print(f"  ✅ Testing module imported successfully")
        
        # Check testing functions
        testing_functions = [f for f in dir(deep_gemm.testing) if not f.startswith('_')]
        print(f"  📊 Available testing functions: {len(testing_functions)}")
        print(f"  🔧 Testing utilities: {testing_functions}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to import DeepGEMM testing: {e}")
        return False


def test_basic_tensor_operations():
    """Test basic tensor operations that don't require specific DeepGEMM calls."""
    print("Testing basic tensor operations...")
    
    try:
        # Test FP8 tensor creation (this validates PyTorch FP8 support)
        a = torch.randn(16, 16, device='cuda', dtype=torch.float32)
        a_fp8 = a.to(torch.float8_e4m3fn)
        print(f"  ✅ FP8 tensor creation successful: {a_fp8.dtype}")
        
        # Test basic CUDA operations
        b = torch.randn(16, 16, device='cuda', dtype=torch.bfloat16)
        c = torch.mm(a.to(torch.bfloat16), b)
        print(f"  ✅ Basic CUDA GEMM successful: {c.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic tensor operations failed: {e}")
        return False


def test_gpu_info():
    """Test GPU information and CUDA availability."""
    print("Testing GPU and CUDA information...")
    
    try:
        if not torch.cuda.is_available():
            print(f"  ❌ CUDA not available")
            return False
            
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name()
        
        print(f"  ✅ CUDA available")
        print(f"  🔢 Device count: {device_count}")
        print(f"  📱 Current device: {current_device}")
        print(f"  🏷️  Device name: {device_name}")
        print(f"  🐍 PyTorch version: {torch.__version__}")
        print(f"  🔥 CUDA version: {torch.version.cuda}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU info test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run minimal DeepGEMM availability test")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")
    args = parser.parse_args()
    
    print("DeepGEMM Minimal Availability Test")
    print("=" * 50)
    
    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Run tests
    tests = [
        ("GPU Info", test_gpu_info),
        ("Library Import", test_library_availability),
        ("Testing Module", test_testing_module),
        ("Basic Tensor Ops", test_basic_tensor_operations),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:18} - {status}")
    
    print(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")
    
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} test(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All availability tests passed!")
        print("DeepGEMM is ready for use!")
        sys.exit(0)


if __name__ == "__main__":
    main()