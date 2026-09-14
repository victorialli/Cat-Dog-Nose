from pathlib import Path

import cv2


project_dir = Path(__file__).parent
image_path = project_dir / "cats_dogs_light" / "train" / "cat.1007.jpg"
output_dir = project_dir / "nose_annotations"
output_path = output_dir / image_path.name

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not find image: {image_path}")

annotated = image.copy()
brush_radius = 3
brush_color = (0, 0, 255)  # Red in OpenCV's BGR format
window_name = "Nose Annotator"


def paint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN or (
        event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
    ):
        cv2.circle(annotated, (x, y), brush_radius, brush_color, -1)


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, paint)

while True:
    cv2.imshow(window_name, annotated)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("s"):
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotation to {output_path}")

    elif key == ord("c"):
        annotated = image.copy()
        print("Cleared annotation")

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()