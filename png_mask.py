import argparse
from pathlib import Path
import cv2
import numpy as np

def fix_annotations_strict(dataset_folder, annotations_folder):
    dataset_path = Path(dataset_folder)
    annotations_path = Path(annotations_folder)
    
    if not annotations_path.exists():
        print(f"Error: The annotations folder '{annotations_path}' does not exist.")
        return

    annotation_files = list(annotations_path.rglob("*.jpg")) + list(annotations_path.rglob("*.jpeg")) + list(annotations_path.rglob("*.png"))

    if not annotation_files:
        print(f"No annotation files found in: {annotations_path}")
        return

    print(f"Processing {len(annotation_files)} annotations with strict filtering...")

    success_count = 0
    for ann_path in annotation_files:
        relative_path = ann_path.relative_to(annotations_path)
        
        source_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = dataset_path / relative_path.with_suffix(ext)
            if candidate.exists():
                source_path = candidate
                break
        
        if source_path is None or not source_path.exists():
            print(f"Could not find original image for {ann_path.name}, skipping.")
            continue

        orig_img = cv2.imread(str(source_path))
        ann_img = cv2.imread(str(ann_path))

        if orig_img is None or ann_img is None:
            print(f"Could not read images for {ann_path.name}, skipping.")
            continue

        if orig_img.shape != ann_img.shape:
            ann_img = cv2.resize(ann_img, (orig_img.shape[1], orig_img.shape[0]))

        # Calculate absolute difference
        diff = cv2.absdiff(orig_img, ann_img)
        
        # High threshold (70) ignores subtle JPEG background noise entirely.
        # We also check that the annotation pixel has a dominant red channel.
        b, g, r = cv2.split(ann_img)
        significant_change = np.any(diff > 70, axis=-1)
        is_reddish = (r.astype(int) - g.astype(int) > 30) & (r.astype(int) - b.astype(int) > 30)
        
        valid_strokes = significant_change & is_reddish

        # Start with a clean copy of the original image background
        result_img = orig_img.copy()
        
        # Paint only the valid brush strokes as 100% pure BGR red (0, 0, 255)
        result_img[valid_strokes] = (0, 0, 255)

        # Define the new .png path
        new_path = ann_path.with_suffix(".png")

        # Save as a lossless PNG
        cv2.imwrite(str(new_path), result_img)

        # Remove old JPEG file
        if ann_path.suffix.lower() in [".jpg", ".jpeg"]:
            ann_path.unlink()

        success_count += 1
        print(f"Fixed: {ann_path.name} -> {new_path.name}")

    print(f"\nSuccessfully fixed {success_count} annotations!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix annotations using high-threshold difference and red validation.")
    parser.add_argument("dataset_dir", type=str, help="Path to the original images folder")
    parser.add_argument("annotations_dir", type=str, help="Path to the annotations folder")
    
    args = parser.parse_args()
    fix_annotations_strict(args.dataset_dir, args.annotations_dir)