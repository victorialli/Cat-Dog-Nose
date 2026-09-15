# Cat-Dog-Nose

This project trains a binary nose-segmentation model for cat and dog images.

## Project structure

- `data/<split>/` contains source images, such as `train`, `train2`, `test`, and `validation`
- `nose_annotations/<split>/` stores red-labeled nose masks as PNG files
- `annotator.py` opens an image set and lets you paint the nose area in red
- `crop.py` crops/pads images to a square size for model input
- `inc_contrast.py` increases image contrast and saves the output into a `higher_contrast/` folder
- `model.py` trains the segmentation model
- `validate.py` runs inference on the test set and saves overlays

## Environment setup

From the project root:

```bash
uv sync
```

If your environment is missing the model stack, install the Python packages used by the training scripts as well:

```bash
uv pip install pillow numpy torch torchvision segmentation-models-pytorch
```

## 1) Annotate training masks

Run the annotator from the project root:

```bash
uv run annotator.py
```

The script defaults to the dataset under `data/train2` and saves masks under `nose_annotations/train2`.

Controls:

- left click + drag: paint the nose red
- `s`: save the current mask
- `c`: clear the current annotation
- `n`: go to the next image
- `p`: go to the previous image
- `q` or `Esc`: quit

The saved mask files should match the source image filenames and use the same relative paths under the corresponding split folder.

## 2) Prepare the image dataset

If your images are not already square or need standardization, resize and pad them before training:

```bash
uv run crop.py data/train --size 224
```

This overwrites each image in the given folder with a square, padded/cropped version.

To increase image contrast before training or validation:

```bash
uv run inc_contrast.py data/train --factor 1.5
```

This creates a `higher_contrast/` subfolder inside that directory and saves processed images there without modifying the originals.

## 3) Train the model

Train the U-Net model using the image and mask folders:

```bash
uv run model.py
```

The script expects:

- images in `data/train`
- matching masks in `nose_annotations/train`

After training, it saves model weights to `nose_unet.pt`.

## 4) Validate the model

Run inference on the test split:

```bash
uv run validate.py
```

This script:

- loads the trained model from `nose_unet.pt`
- reads images from `data/test`
- reads masks from `nose_annotations/test`
- saves overlay predictions into `validation_results/`

## Recommended workflow

1. Annotate the dataset with `annotator.py`
2. Standardize image size with `crop.py`
3. Optionally improve contrast with `inc_contrast.py`
4. Train with `model.py`
5. Validate with `validate.py`

## Notes

- Each image and mask pair should share the same filename stem, for example `cat_001.jpg` and `cat_001.png`.
- The annotation color is hard-coded as red: `(255, 0, 0)`.
- The project is designed around separate dataset splits such as `train`, `test`, and `validation`.
