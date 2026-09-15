
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import torch.optim as optim
import numpy as np
import Path
import PIL import Image

model = smp.Unet(
    encoder_name="resnet18",        
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

batch_size = 8
learning_rate = 0.001
epochs = 20

image_dir = "data/train"
mask_dir = "nose_annotations/train"
annotation_color = (255, 0, 0)  # RGB color for the nose annotation

#path to data


class NoseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, annotation_color):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.annotation_color = annotation_color
        
        # Scan the image directory for .jpg files that have a matching .png mask
        self.samples = [
            p for p in sorted(self.image_dir.glob("*.jpg")) 
            if (self.mask_dir / f"{p.stem}.png").exists()
        ]
        print(f"Found {len(self.samples)} valid image-mask pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        # Load image and normalize to [0.0, 1.0]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        
        # Convert mask to black-and-white (binary 1.0 for annotation, 0.0 for background)
        mask = (np.array(Image.open(mask_path).convert("RGB")) == self.annotation_color).all(axis=-1).astype(np.float32)

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image).permute(2, 0, 1) # Shape: [C, H, W]
        mask_tensor = torch.tensor(mask).unsqueeze(0)       # Shape: [1, H, W]

        return image_tensor, mask_tensor

train_dataset = NoseDataset(image_dir, mask_dir, annotation_color)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#loss function
criterion = smp.losses.TverskyLoss(
    mode="binary",
    from_logits=True,
    alpha=0.3,
    beta=0.7
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#main training loop

model.train()

for epoch in range(epochs):
    epoch_loss = 0.0 #set model to training mode

    for images, masks in train_loader:
        #Step A: Reset gradients
        images = images.float()
        masks = masks.float()

        optimizer.zero_grad()

        #Step B: Forward pass (model makes its predictions)
        outputs = model(images)

        if outputs.shape[-2:] != masks.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks,
                size = outputs.shape[-2:],
                mode = "nearest"
            )

        #Step C: Calculate error (compare predictions to your binary masks)
        loss = criterion(outputs, masks)

        #Step D: Backward pass (calculate updates/gradients)
        loss.backward()


        #Step E: Update model weights
        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], loss: {average_loss:.4f}")



