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
    encoder_weights=None,           # No need for imagenet since we are loading trained weights
    in_channels=3,                  
    classes=1,                      
).to(device)

# Load your trained model weights (Make sure filename matches what you saved!)
model.load_state_dict(torch.load("nose_unet.pt", map_location=device))
model.eval()  # Set model to evaluation mode
print("Model loaded successfully from nose_unet.pt!")

# --- 2. Configuration & Validation Data ---
batch_size = 8
image_dir = "data/test"
mask_dir = "nose_annotations/test"
annotation_color = (255, 0, 0)  # RGB color for the nose annotation
torch.save(model.state_dict(), "nose_unet.pt")
class NoseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, annotation_color):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.annotation_color = annotation_color
        
        self.samples = [
            p for p in sorted(self.image_dir.glob("*.jpg")) 
            if (self.mask_dir / f"{p.stem}.png").exists()
        ]
        print(f"Found {len(self.samples)} valid validation image-mask pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = (np.array(Image.open(mask_path).convert("RGB")) == self.annotation_color).all(axis=-1).astype(np.float32)

        image_tensor = torch.tensor(image).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        return image_tensor, mask_tensor

val_dataset = NoseDataset(image_dir, mask_dir, annotation_color)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- 3. Run Inference & Save Visual Overlays ---
output_dir = Path("validation_results")
output_dir.mkdir(exist_ok=True)

criterion = smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.3, beta=0.7)
val_loss = 0.0

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(val_loader):
        images = images.float().to(device)
        masks = masks.float().to(device)

        # Forward pass
        outputs = model(images)

        if outputs.shape[-2:] != masks.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks, size=outputs.shape[-2:], mode="nearest"
            )

        # Calculate validation loss for metrics tracking
        loss = criterion(outputs, masks)
        val_loss += loss.item()

        # Convert raw logits to binary predictions using Sigmoid and thresholding
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        # Generate and save visual overlays for each image in the batch
        for i in range(images.shape[0]):
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_mask = preds[i, 0]  # Shape: [H, W]

            # Create green overlay
            overlay = img_np.copy()
            overlay[pred_mask] = [0, 255, 0]

            alpha = 0.4  # Transparency
            blended = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

            # Save result image
            img_index = batch_idx * batch_size + i
            result_img = Image.fromarray(blended)
            result_img.save(output_dir / f"val_pred_{img_index}.jpg")

average_val_loss = val_loss / len(val_loader)
print(f"Validation complete! Average Tversky Loss: {average_val_loss:.4f}")
print(f"Visual overlay images saved in the '{output_dir}/' folder.")