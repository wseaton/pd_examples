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
import traceback
import torch
import random

# Add DeepGEMM source to Python path for testing modules
sys.path.insert(0, '/opt/deepgemm')
sys.path.insert(0, '/opt/deepgemm/tests')


def run_test_module(module_name, test_file_path):
    """Run a test module and return success/failure."""
    print(f"\n{'='*60}")
    print(f"Running {module_name}")
    print(f"{'='*60}")
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(0)
        random.seed(0)
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Import and run the test module
        spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
            module_name, test_file_path
        )
        module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run main if it exists
        if hasattr(module, '__name__') and module.__name__ == '__main__':
            # The module will run its tests when imported since it has if __name__ == '__main__'
            pass
        
        print(f"\n✅ {module_name} PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ {module_name} FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run DeepGEMM self tests")
    parser.add_argument("--test", type=str, choices=["all", "fp8", "layout"], 
                       default="all", help="Which test to run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")
    args = parser.parse_args()
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("❌ CUDA not available")
        sys.exit(1)
    
    # Test files mapping (only available tests at this commit)
    test_files = {
        "fp8": "/opt/deepgemm/tests/test_fp8.py", 
        "layout": "/opt/deepgemm/tests/test_layout.py"
    }
    
    # Determine which tests to run
    if args.test == "all":
        tests_to_run = test_files.items()
    else:
        tests_to_run = [(args.test, test_files[args.test])]
    
    print("DeepGEMM Self Test")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if deep_gemm is available
    try:
        import deep_gemm
        print(f"DeepGEMM Library Path: {deep_gemm.__path__}")
    except ImportError as e:
        print(f"❌ DeepGEMM not available: {e}")
        sys.exit(1)
    
    # Run tests
    results = {}
    for test_name, test_path in tests_to_run:
        results[test_name] = run_test_module(test_name, test_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:12} - {status}")
    
    print(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")
    
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} test(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()