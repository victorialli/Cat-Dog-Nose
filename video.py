import cv2
import torch
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp

# --- 1. Setup Models & Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name="resnet18", classes=1).to(device)
model.load_state_dict(torch.load("nose_unet.pt", map_location=device))
model.eval()

# Standard ImageNet Normalization constants
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

# --- 2. Video Paths & Setup ---
video_path = "path_to_your_video.mp4"
output_video_path = "output_nose_video.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define fixed center crop box (adjust crop_size if your cat is too far/close)
crop_size = min(width, height)  # Standard square crop using smaller dimension
cx, cy = width // 2, height // 2
x1, x2 = cx - (crop_size // 2), cx + (crop_size // 2)
y1, y2 = cy - (crop_size // 2), cy + (crop_size // 2)

crop_w = x2 - x1
crop_h = y2 - y1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- 3. Processing Loop ---
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = frame.copy()
        
        # 1. Extract Center Crop
        crop = frame[y1:y2, x1:x2]
        
        # 2. Resize crop to 224x224 and convert to BGR->RGB
        crop_resized = cv2.resize(crop, (224, 224))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Convert to Tensor & Normalize
        img_tensor = torch.tensor(crop_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor = (img_tensor - mean) / std
        
        # 4. Model Prediction
        output = model(img_tensor)
        preds = torch.sigmoid(output) > 0.5
        
        mask_224 = preds.squeeze().cpu().numpy().astype(np.uint8) * 255
        
        # 5. Resize predicted mask back to original crop size
        mask_crop = cv2.resize(mask_224, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        
        # 6. Apply mask directly onto original center-cropped area
        center_region = annotated_frame[y1:y2, x1:x2]
        center_region[mask_crop == 255] = [0, 0, 255]  # Red overlay (BGR)
        
        out_writer.write(annotated_frame)

cap.release()
out_writer.release()
print("Done! Video saved to:", output_video_path)