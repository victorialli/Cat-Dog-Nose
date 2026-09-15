from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# --- 1. Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


model = smp.Unet(
    encoder_name="resnet18",        
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
).to(device)

batch_size = 16
learning_rate = 0.0001
epochs = 60

image_dir = "data/train2/higher_contrast"
mask_dir = "nose_annotations/train2"
annotation_color = (255, 0, 0)  # RGB color for the nose annotation


# --- 3. Dataset Class ---
class NoseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, annotation_color):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.annotation_color = annotation_color
        
        self.samples = [
            p for p in sorted(self.image_dir.glob("*.jpg")) 
            if (self.mask_dir / f"{p.stem}.png").exists()
        ]
        print(f"Found {len(self.samples)} valid pairs in {image_dir}.")

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


# Initialize Datasets and Loaders
train_dataset = NoseDataset(train_image_dir, train_mask_dir, annotation_color)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = NoseDataset(val_image_dir, val_mask_dir, annotation_color)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# --- 4. Model, Loss, Optimizer ---
model = smp.Unet(
    encoder_name="resnet18",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=1,                      
).to(device)

criterion = smp.losses.TverskyLoss(
    mode="binary", 
    from_logits=True, 
    alpha=0.3, 
    beta=0.7
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter("runs/nose_experiment")


# --- 5. Main Training & Validation Loop ---
for epoch in range(epochs):
    
    # ------------------------------------------
    # TRAINING PHASE
    # ------------------------------------------
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()

        #Step B: Forward pass (model makes its predictions)image_dir = "data/test"
        outputs = model(images)

        if outputs.shape[-2:] != masks.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks, size=outputs.shape[-2:], mode="nearest"
            )

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/Overall", average_loss, epoch) # (or batch index)
   # print(f"Epoch [{epoch + 1}/{epochs}], loss: {average_loss:.4f}")

    # ------------------------------------------
    # VALIDATION PHASE (Weights will NOT change)
    # ------------------------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                masks = torch.nn.functional.interpolate(
                    masks, size=outputs.shape[-2:], mode="nearest"
                )

            val_batch_loss = criterion(outputs, masks)
            val_loss += val_batch_loss.item()

    average_val_loss = val_loss / len(val_loader)

    # Logging
    writer.add_scalar("Loss/Train", average_train_loss, epoch)
    writer.add_scalar("Loss/Val", average_val_loss, epoch)

    print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {average_train_loss:.4f} | Val Loss: {average_val_loss:.4f}")

# Save final trained weights
torch.save(model.state_dict(), "nose_unet.pt")
print("Training complete! Model saved to nose_unet.pt")