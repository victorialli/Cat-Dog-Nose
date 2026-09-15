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

# --- 2. Hyperparameters & Directories ---
batch_size = 16
learning_rate = 0.0001
epochs = 60

train_image_dir = "data/train2"
train_mask_dir = "nose_annotations/train2"

val_image_dir = "data/validation"            
val_mask_dir = "nose_annotations/validation"  

annotation_color = (255, 0, 0)  # RGB color for the nose annotation


# --- 3. Dataset Class (Returns Filename for Tracking) ---
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

        # Return filename string so we know if it's a cat or a dog
        return image_tensor, mask_tensor, img_path.name


# Initialize Single Shuffled Train Loader & Single Validation Loader
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
    train_loss_sum = 0.0
    train_cat_loss_sum = 0.0
    train_dog_loss_sum = 0.0
    
    train_count = 0
    train_cat_count = 0
    train_dog_count = 0

    for images, masks, filenames in train_loader:
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if outputs.shape[-2:] != masks.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks, size=outputs.shape[-2:], mode="nearest"
            )

        total_loss = criterion(outputs, masks)
        total_loss.backward()
        optimizer.step()

        batch_size_actual = images.size(0)
        train_loss_sum += total_loss.item() * batch_size_actual
        train_count += batch_size_actual

        cat_indices = [i for i, f in enumerate(filenames) if f.lower().startswith("cat")]
        dog_indices = [i for i, f in enumerate(filenames) if f.lower().startswith("dog")]

        if cat_indices:
            cat_loss = criterion(outputs[cat_indices], masks[cat_indices])
            train_cat_loss_sum += cat_loss.item() * len(cat_indices)
            train_cat_count += len(cat_indices)

        if dog_indices:
            dog_loss = criterion(outputs[dog_indices], masks[dog_indices])
            train_dog_loss_sum += dog_loss.item() * len(dog_indices)
            train_dog_count += len(dog_indices)

    avg_train = train_loss_sum / max(1, train_count)
    avg_train_cat = train_cat_loss_sum / max(1, train_cat_count) if train_cat_count > 0 else 0.0
    avg_train_dog = train_dog_loss_sum / max(1, train_dog_count) if train_dog_count > 0 else 0.0

    # ------------------------------------------
    # VALIDATION PHASE (Weights will NOT change)
    # ------------------------------------------
    model.eval()
    val_loss_sum = 0.0
    val_cat_loss_sum = 0.0
    val_dog_loss_sum = 0.0
    
    val_count = 0
    val_cat_count = 0
    val_dog_count = 0

    with torch.no_grad():
        for images, masks, filenames in val_loader:
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                masks = torch.nn.functional.interpolate(
                    masks, size=outputs.shape[-2:], mode="nearest"
                )

            total_loss = criterion(outputs, masks)
            batch_size_actual = images.size(0)
            val_loss_sum += total_loss.item() * batch_size_actual
            val_count += batch_size_actual

            cat_indices = [i for i, f in enumerate(filenames) if f.lower().startswith("cat")]
            dog_indices = [i for i, f in enumerate(filenames) if f.lower().startswith("dog")]

            if cat_indices:
                cat_loss = criterion(outputs[cat_indices], masks[cat_indices])
                val_cat_loss_sum += cat_loss.item() * len(cat_indices)
                val_cat_count += len(cat_indices)

            if dog_indices:
                dog_loss = criterion(outputs[dog_indices], masks[dog_indices])
                val_dog_loss_sum += dog_loss.item() * len(dog_indices)
                val_dog_count += len(dog_indices)

    avg_val = val_loss_sum / max(1, val_count)
    avg_val_cat = val_cat_loss_sum / max(1, val_cat_count) if val_cat_count > 0 else 0.0
    avg_val_dog = val_dog_loss_sum / max(1, val_dog_count) if val_dog_count > 0 else 0.0

    # --- Logging to TensorBoard ---
    # 1. Separate individual graphs
    writer.add_scalar("Loss/Train", avg_train, epoch)
    writer.add_scalar("Loss/Train_Cat", avg_train_cat, epoch)
    writer.add_scalar("Loss/Train_Dog", avg_train_dog, epoch)
    
    writer.add_scalar("Loss/Val", avg_val, epoch)
    writer.add_scalar("Loss/Val_Cat", avg_val_cat, epoch)
    writer.add_scalar("Loss/Val_Dog", avg_val_dog, epoch)

    # 2. Overlaid combined graph for Train vs. Val
    writer.add_scalars("Loss/Train_vs_Val", {
        "Train": avg_train,
        "Val": avg_val
    }, epoch)

    print(f"Epoch [{epoch + 1}/{epochs}] | "
          f"Train: {avg_train:.4f} (Cat: {avg_train_cat:.4f}, Dog: {avg_train_dog:.4f}) | "
          f"Val: {avg_val:.4f} (Cat: {avg_val_cat:.4f}, Dog: {avg_val_dog:.4f})")

# Save final trained weights
torch.save(model.state_dict(), "nose_unet.pt")
print("Training complete! Model saved to nose_unet.pt")