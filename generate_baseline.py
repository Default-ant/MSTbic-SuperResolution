import os
import cv2
import torch
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from dependencies.KAIR.models.network_swinir import SwinIR

def download_weights(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading pre-trained weights from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")
    else:
        print("Pre-trained weights already found locally.")

def main():
    # 1. Setup paths
    weights_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
    weights_path = "swinir_lw_x4_baseline.pth"
    img_path = "data/raw_lr/lenna.png"
    out_dir = "/home/krish/.gemini/antigravity/brain/f8e66c96-cd2d-4ced-8d1e-456bdb41cfde/"

    download_weights(weights_url, weights_path)

    # 2. Load the baseline model
    model = SwinIR(
        upscale=4, 
        in_chans=3, 
        img_size=64, 
        window_size=8,
        img_range=1.0, 
        depths=[6, 6, 6, 6], 
        embed_dim=60, 
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2, 
        upsampler='pixelshuffledirect', 
        resi_connection='1conv'
    )
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    # Handle state_dict keys if needed ('params' vs straight dict)
    state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Load HR image
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load Lenna.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 4. Generate LR by bicubic downsampling (simulating the real world)
    H, W, _ = img_rgb.shape
    lr_h, lr_w = H // 4, W // 4
    lr_rgb = cv2.resize(img_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    # 5. Generate Bicubic Upsampling baseline
    bicubic_rgb = cv2.resize(lr_rgb, (lr_w * 4, lr_h * 4), interpolation=cv2.INTER_CUBIC)
    
    # 6. Run SwinIR SR
    # Convert LR to tensor
    lr_t = torch.from_numpy(lr_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    
    print("Running SwinIR Baseline prediction on CPU...")
    with torch.no_grad():
        sr_t = model(lr_t)
        
    sr_rgb = (sr_t.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

    # 7. Plot to compare
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Zoom in on a texture-heavy patch to make differences obvious
    y_start, x_start = 200, 200 # Lenna's hat/feathers
    patch_size = 150
    
    def extract_patch(im):
        return im[y_start:y_start+patch_size, x_start:x_start+patch_size, :]
        
    axes[0].imshow(extract_patch(img_rgb))
    axes[0].set_title("Original HR Image (Ground Truth)", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(extract_patch(bicubic_rgb))
    axes[1].set_title("Standard Bicubic Interpolation (Blurry)", fontsize=16)
    axes[1].axis('off')
    
    axes[2].imshow(extract_patch(sr_rgb))
    axes[2].set_title("Official SwinIR Baseline (Pretrained weights)", fontsize=16)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "swinir_baseline_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_dir}/swinir_baseline_comparison.png")

if __name__ == "__main__":
    main()
