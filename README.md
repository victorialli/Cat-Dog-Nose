# Cat-Dog-Nose

## Run the annotator

Install the dependencies and start the OpenCV window from the project root:

```bash
uv sync
uv run annotator.py
```

The annotator opens `cats_dogs_light/train/cat.1007.jpg`. Drag over the nose to paint it red, press `s` to save the colored copy to `nose_annotations/`, `c` to clear the paint, or `q`/`Esc` to quit.