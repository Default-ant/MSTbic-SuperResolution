import os
import subprocess
import sys
import torch

def main():
    """
    Step 2: Training Launcher
    This script runs the actual model training. It sets up the python path correctly
    and detects whether you have a GPU (helpful for Kaggle) or are running on CPU (local).
    """
    print("\n--- Phase 4: Model Training (MSTbic SwinIR) ---")
    
    # Setup PYTHONPATH so the KAIR library dependencies are found
    cwd = os.getcwd()
    kair_path = os.path.join(cwd, "dependencies", "KAIR")
    src_path = os.path.join(cwd, "src")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{kair_path}:{src_path}:{env.get('PYTHONPATH', '')}"

    # Path to your MSTbic custom training script
    train_script = os.path.join(src_path, "mikrosr", "train", "main_train_psnr_custom.py")
    
    # Path to the new configuration we made
    config_file = os.path.join("options", "train_configs", "MSTbic", "train_div2k_100.json")

    cmd = [sys.executable, train_script, "--opt", config_file]

    print("\nChecking Hardware...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Detected: {gpu_name}. Training will be fast!")
    else:
        print("🐢 CPU Detected. Training will be slow.")
        print("Note: If you run this script in a Kaggle Notebook with a T4 GPU, it will finish much faster!")
        
    print(f"\nLaunching Training with Config: {config_file}")
    print("Command:", " ".join(cmd))
    
    try:
        # Run training loop!
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nTraining manually stopped. You can resume later.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training crashed with error code {e.returncode}")

if __name__ == "__main__":
    main()
