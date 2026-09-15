# /// script
# dependencies = [
#     "pillow",
# ]
# ///

import argparse
from pathlib import Path
from PIL import Image, ImageEnhance

def increase_contrast(input_dir, contrast_factor=1.5):
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory.")
        return

    # Create the output folder named 'higher_contrast' inside the target folder
    output_path = input_path / "higher_contrast"
    output_path.mkdir(exist_ok=True)

    # Supported image formats
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    count = 0
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            try:
                with Image.open(file_path) as img:
                    enhancer = ImageEnhance.Contrast(img)
                    enhanced_img = enhancer.enhance(contrast_factor)
                    
                    # Save to the higher_contrast folder, keeping the original filename
                    output_file = output_path / file_path.name
                    enhanced_img.save(output_file)
                    print(f"Processed: {file_path.name}")
                    count += 1
            except Exception as e:
                print(f"Could not process {file_path.name}: {e}")

    print(f"\nDone! Successfully processed {count} images.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Increase contrast for images in a folder.")
    parser.add_argument("folder", type=str, help="Path to the folder containing images")
    parser.add_argument("--factor", type=float, default=1.5, help="Contrast level (1.0 = original, 1.5 = 50%% more contrast, etc.)")
    
    args = parser.parse_args()
    increase_contrast(args.folder, args.factor)