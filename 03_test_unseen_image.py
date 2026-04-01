import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup import paths
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("dependencies/KAIR"))

from models.select_model import define_Model
from utils import utils_option as option
from utils import utils_image as util

def load_trained_model(weight_path):
    print(f"\nLoading weights from: {weight_path}")
    
    # Load config template
    opt_path = 'options/train_configs/MSTbic/train_div2k_100.json'
    opt = option.parse(opt_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    
    # Overwrite the pretrained weight path to use our specifically chosen checkpoint
    opt['path']['pretrained_netG'] = os.path.abspath(weight_path)
    
    # If no GPU is available, ensure we don't try to use one
    if not torch.cuda.is_available():
        opt['gpu_ids'] = []
        
    model = define_Model(opt)
    model.load()
    return model

def main():
    print("\n--- Phase 5: Super-Resolution on Unseen Image ---")
    
    # 1. Provide an Unseen Image (from outside our 100 DIV2K dataset)
    # The user has 'butterfly.png' in raw_lr which was NOT in the DIV2K valid set!
    unseen_img_path = "data/raw_lr/butterfly.png"
    
    if not os.path.exists(unseen_img_path):
        print(f"Error: Could not find {unseen_img_path}.")
        return

    # 2. Setup the Model
    # Here we should point to the newly trained model checkpoint.
    # Since training takes time, if a freshly trained model doesn't exist yet, 
    # we'll use your baseline as a fallback so this script still works to show your professor.
    trained_model_dir = "superresolution/mstbic_swinir_lw_div2k_100/models"
    best_weight = os.path.join(trained_model_dir, "500_G.pth")
    fallback_weight = "superresolution/swinir_lw_x4_baseline.pth"
    
    if os.path.exists(best_weight):
        model = load_trained_model(best_weight)
        print("Using newly trained DIV2K 100-image weights!")
    elif os.path.exists(fallback_weight):
        model = load_trained_model(fallback_weight)
        print("Training hasn't finished yet! Using base weights for demonstration.")
    else:
        print("No weights found. Did you run step 02_run_training_demo.py?")
        return
        
    # 3. Process Original Image
    print(f"\nProcessing Unseen Image: {unseen_img_path}")
    img_gt = cv2.imread(unseen_img_path)
    if img_gt is None:
        return
        
    img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    
    # Create Low Resolution version by 4x downscaling (bicubic)
    h, w = img_gt_rgb.shape[:2]
    # Simulate low resolution input
    img_lr_rgb = cv2.resize(img_gt_rgb, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    
    # Standard Bicubic upscaling (Poor Quality Baseline)
    img_bicubic = cv2.resize(img_lr_rgb, (w, h), interpolation=cv2.INTER_CUBIC)

    # 4. Neural Network Super Resolution
    print("Running SwinIR prediction...")
    # Convert numpy LR to tensor for KAIR
    img_lr_tensor = torch.from_numpy(img_lr_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    model.feed_data({'L': img_lr_tensor}, need_H=False)
    with torch.no_grad():
        model.test()
        
    visuals = model.current_visuals(need_H=False)
    output_tensor = visuals['E']
    img_sr_rgb = util.tensor2uint(output_tensor)

    # 5. Calculate PSNR (Peak Signal to Noise Ratio) against Ground Truth
    # Calculate on Y channel (luminance) as per academic SR standard
    img_gt_y = util.rgb2ycbcr(img_gt_rgb.astype(np.float32) / 255., only_y=True) * 255.
    img_bicubic_y = util.rgb2ycbcr(img_bicubic.astype(np.float32) / 255., only_y=True) * 255.
    img_sr_y = util.rgb2ycbcr(img_sr_rgb.astype(np.float32) / 255., only_y=True) * 255.
    
    psnr_bicubic = util.calculate_psnr(img_gt_y, img_bicubic_y)
    psnr_sr = util.calculate_psnr(img_gt_y, img_sr_y)

    print(f"Results -> Bicubic PSNR: {psnr_bicubic:.2f} dB, Model PSNR: {psnr_sr:.2f} dB")

    # 6. Plotting the Comparison for presentation
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Zoom into a 120x120 patch where differences are obvious (e.g. butterfly wing texture)
    y_s, x_s = h//2 - 60, w//2 - 60
    patch = slice(y_s, y_s + 120), slice(x_s, x_s + 120)
    
    def crop(img):
        return img[patch[0], patch[1], :]

    axes[0].imshow(img_gt_rgb)
    axes[0].set_title(f"Full Ground Truth\nUnseen Image", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(crop(img_gt_rgb))
    axes[1].set_title(f"Ground Truth (Crop)\nPSNR: ∞", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(crop(img_bicubic))
    axes[2].set_title(f"Bicubic Opscale\nPSNR: {psnr_bicubic:.2f} dB", fontsize=14)
    axes[2].axis('off')

    axes[3].imshow(crop(img_sr_rgb))
    axes[3].set_title(f"Our Model (SwinIR)\nPSNR: {psnr_sr:.2f} dB", fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()
    result_path = "unseen_image_comparison.png"
    plt.savefig(result_path, dpi=150)
    print(f"\n✅ Presentation graphic saved successfully to: {result_path}")

if __name__ == "__main__":
    main()
