from pathlib import Path

import cv2

# q to quit, n to go to the next image, p to go to the previous image, s to save the current annotation, c to clear the current annotation

project_dir = Path(__file__).parent
dataset_dir = project_dir / "data" / "train2"
output_dir = project_dir / "nose_annotations" / "train2"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = sorted(
    path
    for path in dataset_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in image_extensions
)

if not image_paths:
    raise FileNotFoundError(f"No images found in {dataset_dir}")

current_index = 0
image = None
annotated = None
brush_radius = 1
brush_color = (0, 0, 255)
window_name = "Nose Annotator"


def output_path_for(source_path):
    relative_path = source_path.relative_to(dataset_dir)
    return (output_dir / relative_path).with_suffix(".png")


def load_image(index):
    global image, annotated, current_index

    current_index = index
    source_path = image_paths[current_index]
    saved_path = output_path_for(source_path)

    image = cv2.imread(str(source_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {source_path}")

    if saved_path.exists():
        annotated = cv2.imread(str(saved_path))
        if annotated is None:
            annotated = image.copy()
    else:
        annotated = image.copy()

    title = (
        f"Nose Annotator | {current_index + 1}/{len(image_paths)} | "
        f"{source_path.name} | n:next p:previous s:save c:clear q:quit"
    )
    cv2.setWindowTitle(window_name, title)


def save_current():
    source_path = image_paths[current_index]
    saved_path = output_path_for(source_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(saved_path), annotated):
        raise RuntimeError(f"Could not save annotation: {saved_path}")

    print(f"Saved: {saved_path}")


def paint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN or (
        event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
    ):
        cv2.circle(annotated, (x, y), brush_radius, brush_color, -1)


cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1000, 800)
cv2.setMouseCallback(window_name, paint)

load_image(0)


while True:
    cv2.imshow(window_name, annotated)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("s"):
        save_current()

    elif key == ord("n"):
        if current_index < len(image_paths) - 1:
            load_image(current_index + 1)
        else:
            print("Already on the last image.")

    elif key == ord("p"):
        if current_index > 0:
            load_image(current_index - 1)
        else:
            print("Already on the first image.")

    elif key == ord("c"):
        annotated = image.copy()
        print("Cleared current annotation.")

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()