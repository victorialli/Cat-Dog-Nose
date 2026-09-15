# Cat-Dog-Nose
goal: produce a segmentation mask identifying the nose of a dog/cat image

## Run the annotator
Install the dependencies and start the OpenCV window from the project root:

```bash
uv sync
uv run annotator.py
```

The annotator opens `cats_dogs_light/train/(specify test/train/validate), saves to nose_annotations/(specify test/train/validate). Drag over the nose to paint it red, press `s` to save the colored copy to `nose_annotations/`, `c` to clear the paint, `q`/`Esc` to quit, 'n' to advance to the next image, 'p' for the previous image
