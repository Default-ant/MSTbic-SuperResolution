import cv2
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt

# Add the KAIR root to sys.path
sys.path.insert(0, '/home/krish/dip/SuperResolutionMultiscaleTraining/dependencies/KAIR')
sys.path.append('/home/krish/dip/SuperResolutionMultiscaleTraining')

from models.network_swinir import SwinIR
from utils import utils_image as util

def load_model(weights_path, device, upsampler='pixelshuffledirect'):
    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                  img_range=1.0, depths=[6, 6, 6, 6], embed_dim=60, 
                  num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                  upsampler=upsampler, resi_connection='1conv').to(device)
    
    pretrained_model = torch.load(weights_path, map_location=device)
    # Check if weights are nested
    if 'params' in pretrained_model:
        pretrained_model = pretrained_model['params']
    elif 'params_ema' in pretrained_model:
        pretrained_model = pretrained_model['params_ema']
        
    model.load_state_dict(pretrained_model, strict=True)
    model.eval()
    return model

def main():
    device = torch.device('cpu')
    
    # 1. Load Models
    print("Loading models...")
    baseline_path = 'swinir_lw_x4_baseline.pth'
    mstbic_path = '/home/krish/dip/FARB weights/5500_E.pth'
    
    model_baseline = load_model(baseline_path, device, upsampler='pixelshuffledirect')
    model_mstbic = load_model(mstbic_path, device, upsampler='pixelshuffledirect')
    
    # 2. Prepare Image (Baby from Set5)
    img_path = 'dependencies/KAIR/testsets/set5/baby.bmp'
    img_gt_uint8 = cv2.imread(img_path)
    img_gt_rgb = cv2.cvtColor(img_gt_uint8, cv2.COLOR_BGR2RGB)
    
    # Generate LR
    h, w = img_gt_rgb.shape[:2]
    lr_h, lr_w = h // 4, w // 4
    img_lr_rgb = cv2.resize(img_gt_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    # Generate Bicubic Upsampling
    img_bicubic = cv2.resize(img_lr_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Model Inference Helper
    def infer(model, lr_img):
        img_l = torch.from_numpy(np.transpose(lr_img.astype(np.float32) / 255., (2, 0, 1))).float().unsqueeze(0).to(device)
        
        # Padding
        window_size = 8
        _, _, h_old, w_old = img_l.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_l = torch.cat([img_l, torch.flip(img_l, [2])], 2)[:, :, :h_old + h_pad, :]
        img_l = torch.cat([img_l, torch.flip(img_l, [3])], 3)[:, :, :, :w_old + w_pad]
        
        with torch.no_grad():
            output = model(img_l)
        output = output[..., :h_old * 4, :w_old * 4]
        
        return util.tensor2uint(output) # Already in RGB/BGR based on input, matplotlib expects RGB

    print("Running inference...")
    img_sr_baseline = infer(model_baseline, img_lr_rgb)
    img_sr_mstbic = infer(model_mstbic, img_lr_rgb)
    
    # 4. Generate Plot
    print("Generating comparison plot...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    
    # Focus on an eye/face patch
    y, x, s = 150, 150, 150 # Patch coordinates for baby
    def get_patch(im):
        return im[y:y+s, x:x+s, :]

    axes[0].imshow(get_patch(img_gt_rgb))
    axes[0].set_title("Original HR", fontsize=20)
    axes[1].imshow(get_patch(img_bicubic))
    axes[1].set_title("Standard Bicubic", fontsize=20)
    axes[2].imshow(get_patch(img_sr_baseline))
    axes[2].set_title("SwinIR Baseline (32.25 dB)", fontsize=20)
    axes[3].imshow(get_patch(img_sr_mstbic))
    axes[3].set_title("Your MSTbic (5500 Iters)", fontsize=20)
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    out_path = 'final_comparison_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Success! Visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
