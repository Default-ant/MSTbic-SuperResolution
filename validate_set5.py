import cv2
import numpy as np
import torch
import os
import sys

# Add the KAIR root to sys.path
sys.path.insert(0, '/home/krish/dip/SuperResolutionMultiscaleTraining/dependencies/KAIR')
sys.path.append('/home/krish/dip')

from models.network_swinir import SwinIR
from utils import utils_image as util

def main():
    device = torch.device('cpu')
    # Architecture from JSON config
    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                  img_range=1.0, depths=[6, 6, 6, 6], embed_dim=60, 
                  num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                  upsampler='pixelshuffledirect', resi_connection='1conv').to(device)
    
    # Load weights
    model_path = '/home/krish/dip/FARB weights/5500_E.pth'
    pretrained_model = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_model, strict=True)
    model.eval()

    testset_dir = '/home/krish/dip/SuperResolutionMultiscaleTraining/dependencies/KAIR/testsets/set5'
    psnrs_y = []
    
    images = [f for f in os.listdir(testset_dir) if f.endswith('.bmp') or f.endswith('.png')]
    print(f"Evaluating {len(images)} images in Set5 (x4) with Matlab-style resizing...")
    
    for img_name in sorted(images):
        img_path = os.path.join(testset_dir, img_name)
        img_gt_uint8 = util.imread_uint(img_path, n_channels=3)
        h, w = img_gt_uint8.shape[:2]
        
        # Prepare LR using Matlab-style bicubic (if available in KAIR util)
        # KAIR's imresize expects float [0, 1]
        img_gt_f = img_gt_uint8.astype(np.float32) / 255.
        
        # Determine if util.imresize_np exists (KAIR usually has it)
        # If not, use cv2 with a better kernel or just stick to cv2 but fix padding
        img_lr_f = util.imresize_np(img_gt_f, 1/4)
        
        # Mirror Padding (Official SwinIR logic)
        img_l = torch.from_numpy(np.transpose(img_lr_f[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)
        window_size = 8
        _, _, h_old, w_old = img_l.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_l = torch.cat([img_l, torch.flip(img_l, [2])], 2)[:, :, :h_old + h_pad, :]
        img_l = torch.cat([img_l, torch.flip(img_l, [3])], 3)[:, :, :, :w_old + w_pad]
        
        # Inference
        with torch.no_grad():
            output = model(img_l)
        output = output[..., :h_old * 4, :w_old * 4]
        
        # Output to BGR uint8
        img_e_bgr = util.tensor2uint(output)
        img_e_rgb = cv2.cvtColor(img_e_bgr, cv2.COLOR_BGR2RGB)
        
        # Metrics (Y-channel PSNR)
        img_gt_y = util.rgb2ycbcr(img_gt_f, only_y=True) * 255.
        img_e_y = util.rgb2ycbcr(img_e_rgb.astype(np.float32) / 255., only_y=True) * 255.
        cur_psnr_y = util.calculate_psnr(img_gt_y, img_e_y)
        
        psnrs_y.append(cur_psnr_y)
        print(f"{img_name}: PSNR_Y={cur_psnr_y:.2f} dB")

    print(f"---\nAverage PSNR Y: {np.mean(psnrs_y):.2f} dB")

if __name__ == '__main__':
    main()
