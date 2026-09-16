from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

# --- 1. Setup Device & Load Model Weights ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet18",        
    encoder_weights=None,           # No need for ImageNet weights since we are loading trained weights
    in_channels=3,                  
    classes=1,                      
).to(device)

# Load your trained model weights
model.load_state_dict(torch.load("nose_unet.pt", map_location=device))
model.eval()  # Set model to evaluation mode
print("Model loaded successfully from nose_unet.pt!")

# --- 2. Configuration & Inference Data ---
batch_size = 8
image_dir = "data/inference_images"  # Folder containing your raw images
output_dir = Path("inference_results")
output_dir.mkdir(parents=True, exist_ok=True)

class InferenceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        # Find all jpg and png images in the folder
        self.samples = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        print(f"Found {len(self.samples)} images for inference.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]

        # Load image and normalize to [0, 1]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.tensor(image).permute(2, 0, 1)

        # Return tensor and the original filename string for saving later
        return image_tensor, img_path.name

inference_dataset = InferenceDataset(image_dir)
inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

# --- 3. Run Inference & Save Visual Overlays ---
with torch.no_grad():
    for images, filenames in inference_loader:
        images = images.float().to(device)

        # Forward pass (Inference)
        outputs = model(images)

        # Convert raw logits to binary predictions using Sigmoid and thresholding
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        # Generate and save visual overlays for each image in the batch
        for i in range(images.shape[0]):
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_mask = preds[i, 0]  # Shape: [H, W]

            # Create overlay (Green overlay for prediction)
            overlay = img_np.copy()
            overlay[pred_mask] = [0, 255, 0]

            alpha = 0.4  # Transparency level
            blended = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

            # Save result image using its original filename
            result_img = Image.fromarray(blended)
            result_img.save(output_dir / f"pred_{filenames[i]}")

print(f"Inference complete! Visual overlay images saved in the '{output_dir}/' folder.")