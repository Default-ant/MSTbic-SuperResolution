import os
import json
import shutil
from pathlib import Path

def main():
    print("Setting up local 1-image training demo...")
    
    # 1. Ensure folders exist
    hr_dir = "data/local_demo/HR"
    lr_dir = "data/local_demo/LR"
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # 2. Find any local image to use as a dummy train dataset
    sample_img = "data/raw_lr/lenna.png"
    if not os.path.exists(sample_img):
        for p in Path('.').rglob('*.png'):
            if 'local_demo' not in str(p):
                sample_img = str(p)
                break
                
    if not os.path.exists(sample_img):
        print("Creating a dummy blank image since no PNGs were found.")
        import numpy as np
        import cv2
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(f"{hr_dir}/demo.png", dummy)
        cv2.imwrite(f"{lr_dir}/demo.png", dummy)
    else:
        print(f"Found image: {sample_img}. Copying to demo dataset folders...")
        shutil.copy(sample_img, f"{hr_dir}/demo.png")
        shutil.copy(sample_img, f"{lr_dir}/demo.png")

    # 3. Create extremely minimal local CPU configuration
    config_path = "options/train_configs/MSTbic/test_mstbic.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Force CPU, batch size 1, 0 workers
    config['gpu_ids'] = [] 
    config['datasets']['train']['dataroot_H'] = hr_dir
    config['datasets']['train']['dataroot_L'] = lr_dir
    config['datasets']['train']['dataloader_batch_size'] = 1
    config['datasets']['train']['dataloader_num_workers'] = 0

    config['datasets']['test']['dataroot_H'] = hr_dir
    config['datasets']['test']['dataroot_L'] = lr_dir

    # Shrink training to almost nothing! 
    config['train']['checkpoint_print'] = 1 # Print every iteration
    config['train']['checkpoint_save'] = 10 # Save a permanent .pth every 10 steps
    config['train']['G_scheduler_milestones'] = [5, 10, 15]

    demo_config = "options/train_configs/MSTbic/local_demo.json"
    with open(demo_config, 'w') as f:
        json.dump(config, f, indent=2)

    print("\nLocal 1-Image Dataset and Config generated successfully!")
    print("Run this command to see the training loop run locally on your CPU:")
    print("PYTHONPATH=$PWD/dependencies/KAIR:$PWD/src python3 src/mikrosr/train/main_train_psnr_custom.py --opt options/train_configs/MSTbic/local_demo.json")

if __name__ == "__main__":
    main()
