import torch
from trouch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet18",        
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

image_dir = "data/train"
mask_dir = "nose_annotations/train"

batch_size = 0
learning_rate = 0
epochs = 0

#pth to data
class NoseDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Fetch file paths for this specific index
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load images
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Convert to tensors and clean binary masks
        image_tensor = TF.to_tensor(image)  # [3, 224, 224]
        mask_np = np.array(mask)
        binary_mask = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.tensor(binary_mask).unsqueeze(0)  # [1, 224, 224]

        return image_tensor, mask_tensor


#loss function
criterion =
optimizer = 


#main training loop

model.train()

for epoch in range(epochs):
    epoch_loss = 0 #set model to training mode

    for images, annotations in train_loader:
        #Step A: Reset gradients
        optimizer.zero_grad()

        #Step B: Forward pass (model makes its predictions)
        outputs = model(images)

        #Step C: Calculate error (compare predictions to your binary masks)
        loss = criterion(outputs, masks)

        #Step D: Backward pass (calculate updates/gradients)
        loss.backwards()
        #Step E: Update model weights
