import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def compute_gradient_map(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return grad_mag

def compute_laplacian_pyramid(img_tensor):
    # simulate the laplacian filter as implemented in losses_frequency.py
    lap_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    B, C, H, W = img_tensor.shape
    x_flat = img_tensor.reshape(B * C, 1, H, W)
    out = F.conv2d(x_flat, lap_filter, padding=1)
    return out.reshape(B, C, H, W)

def main():
    hr_path = "/home/krish/dip/SuperResolutionMultiscaleTraining/data/raw_lr/butterfly.png" 
    if not os.path.exists(hr_path):
        print(f"Could not find {hr_path}. Please check the path.")
        
        # fallback to any available png
        from pathlib import Path
        for p in Path('.').rglob('*.png'):
            hr_path = str(p)
            break
            
    img = cv2.imread(hr_path)
    if img is None:
        print("Failed to load image.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Gradient Map (For Patch Selection)
    grad_mag = compute_gradient_map(img_rgb)
    
    # 2. Laplacian Map (For Frequency Loss)
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    lap_out = compute_laplacian_pyramid(img_t)
    lap_img = lap_out[0].abs().mean(dim=0).numpy() # average over channels
    lap_img = np.clip(lap_img * 5, 0, 1) # Enhance contrast visually
    
    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Image", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(grad_mag, cmap='magma')
    axes[1].set_title("2. Sobel Gradient Map\n(Used for Patch Selection)", fontsize=16)
    axes[1].axis('off')
    
    axes[2].imshow(lap_img, cmap='hot')
    axes[2].set_title("3. Laplacian High-Frequency Target\n(Used for Frequency Loss)", fontsize=16)
    axes[2].axis('off')
    
    out_dir = "/home/krish/.gemini/antigravity/brain/f8e66c96-cd2d-4ced-8d1e-456bdb41cfde/"
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mstbic_visualization.png"), dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_dir}/mstbic_visualization.png")

if __name__ == "__main__":
    main()
