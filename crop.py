import os
import argparse
from PIL import Image, ImageOps

def process_image(image_path, target_size=224):
    with Image.open(image_path) as img:
        width, height = img.size
        
        # If the image is smaller than the target size, pad it instead of stretching/cropping
        if width < target_size or height < target_size:
            img_processed = ImageOps.pad(
                img, 
                (target_size, target_size), 
                centering=(0.5, 0.5), 
                color=(0, 0, 0)  # Black padding (change to (255, 255, 255) for white)
            )
        else:
            # Standard center crop to square, then resize to target size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            
            img_cropped = img.crop((left, top, right, bottom))
            img_processed = img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
        img_processed.save(image_path)

def process_folder(input_dir, target_size=224):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)
            try:
                process_image(input_path, target_size)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and pad images to square for ML processing.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("-s", "--size", type=int, default=224, help="Target square dimension in pixels (default: 224).")
    
    args = parser.parse_args()
    process_folder(args.input_folder, args.size)