"""
Environment Validation Script for Mini-R1
Checks GPU, CUDA, disk space, and RAM before training
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_gpu():
    """Check GPU availability and specifications"""
    print_section("GPU Check")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            print("   This project requires a GPU with CUDA support.")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU count: {gpu_count}")
        has_sufficient_vram = False
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"\n   GPU {i}: {props.name}")
            print(f"   VRAM: {vram_gb:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # Current default: 3B full fine-tuning with a reference model
            if vram_gb < 75:
                print(f"   ⚠️  {vram_gb:.2f}GB VRAM is insufficient for the default full fine-tuning config.")
                print("   Use a single 80GB-class GPU, or create and validate a separate PEFT config.")
            else:
                print(f"   ✅ {vram_gb:.2f}GB VRAM is sufficient for the default config.")
                has_sufficient_vram = True
        
        return has_sufficient_vram
        
    except ImportError:
        print("❌ PyTorch not installed!")
        print("   Run: uv sync --locked --no-install-project")
        print("   Use a CUDA 12.8-compatible RunPod image if PyTorch still cannot load.")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def check_disk_space():
    """Check available disk space"""
    print_section("Disk Space Check")
    
    try:
        stat = os.statvfs('/workspace' if os.path.exists('/workspace') else '.')
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        print(f"Available disk space: {free_gb:.2f} GB")
        
        required_gb = 100
        if free_gb < required_gb:
            print(f"⚠️  WARNING: Only {free_gb:.2f}GB free. Recommended: {required_gb}GB+")
            print("   Consider:")
            print("   - Reducing num_samples in dataset prep")
            print("   - Reducing save_total_limit in config")
            return False
        else:
            print(f"✅ Sufficient disk space ({free_gb:.2f}GB available)")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Don't fail on this


def check_ram():
    """Check available RAM"""
    print_section("RAM Check")
    
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = int([x for x in meminfo.split('\n') if 'MemTotal' in x][0].split()[1]) / (1024**2)
        mem_available = int([x for x in meminfo.split('\n') if 'MemAvailable' in x][0].split()[1]) / (1024**2)
        
        print(f"Total RAM: {mem_total:.2f} GB")
        print(f"Available RAM: {mem_available:.2f} GB")
        
        if mem_available < 16:
            print(f"⚠️  WARNING: Only {mem_available:.2f}GB RAM available. Recommended: 16GB+")
            return False
        else:
            print(f"✅ Sufficient RAM ({mem_available:.2f}GB available)")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check RAM: {e}")
        return True  # Don't fail on this


def check_dependencies():
    """Check if required packages are installed"""
    print_section("Dependencies Check")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'trl': 'TRL',
        'peft': 'PEFT',
        'bitsandbytes': 'BitsAndBytes',
    }
    
    all_good = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            version = __import__(package).__version__
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name} not installed!")
            all_good = False
    
    if not all_good:
        print("\n   Install missing packages with: uv sync")
        return False
    
    return True


def check_hf_token():
    """Check if Hugging Face token is configured"""
    print_section("Hugging Face Authentication")
    
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    env_token = os.environ.get('HF_TOKEN')
    
    if token_file.exists() or env_token:
        print("✅ Hugging Face token found")
        return True
    else:
        print("⚠️  WARNING: Hugging Face token not found")
        print("   Run: huggingface-cli login")
        print("   Or set: export HF_TOKEN=your_token")
        return False


def suggest_config():
    """Suggest optimal configuration based on GPU"""
    print_section("Configuration Suggestions")
    
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print("\nCurrent default configuration:\n")

            if vram_gb >= 75:
                print("🎯 80GB-class GPU detected")
                print("   The default Qwen2.5-3B full fine-tuning config is supported.")
            else:
                print("⚠️  The default full fine-tuning config requires a single 80GB-class GPU.")
                print("   Lower-memory PEFT settings require separate tuning and validation.")
    
    except Exception:
        pass


def main():
    """Run all environment checks"""
    print("\n" + "🔍 Mini-R1 Environment Validation".center(80))
    
    results = {
        'gpu': check_gpu(),
        'disk': check_disk_space(),
        'ram': check_ram(),
        'dependencies': check_dependencies(),
        'hf_token': check_hf_token(),
    }
    
    suggest_config()
    
    # Final summary
    print_section("Summary")
    
    all_passed = all(results.values())
    critical_passed = results['gpu'] and results['dependencies']
    
    if all_passed:
        print("✅ All checks passed! Ready to start training.")
        print("\nNext steps:")
        print("  1. bash run_all.sh         # Run full pipeline")
        print("  2. OR run steps manually (see RUNPOD_SETUP.md)")
        return 0
        
    elif critical_passed:
        print("⚠️  Some optional checks failed, but you can proceed.")
        print("   Review warnings above and adjust as needed.")
        print("\nYou can still run:")
        print("  bash run_all.sh")
        return 0
        
    else:
        print("❌ Critical checks failed! Please fix the issues above.")
        print("\nFailed checks:")
        for check, passed in results.items():
            if not passed:
                print(f"  - {check}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
