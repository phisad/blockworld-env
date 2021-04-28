# Blockworld Random 2D

The dataset contains `3-channel` images of `640 x 640` dimensions.
*Important note: Don't forget to translate bounding boxes and positions, when resizing the images for training!*

The images contain colored boxes of size `48 x 48` which have `digits` on them.

The digits are (almost) centered on the box with font `Consolas` in font size `24` and `bold`.

The labels include:

- **The block_id's in the image.** This is the digit shown on the box
- **The bbox's in the image.** This is the bounding box starting from the upper-left corner of the box.
- **The block_pos's in the image.** This is the center coordinate of the box in the original `640 x 640` image space
- **The block_pos_discrete's in the image.** This is the position of the box regarding a `10 x 10` grid put on the
  original image space. The discrete positions are measured from the box center. If a center falls on an edge, then the
  position is rounded down. The positions are zero-based ranging from 0-9 (incl.).
- **The images file_name's.** The name of the file in the according sub-directory (not having the sub-directory path).

```
{
    "bbox": [x_box: int, yh_box: int],
    "block_id": int,
    "block_pos": [x_box_center: int, y_box_center: int],
    "block_pos_discrete": [x_grid: int, y_grid: int],
    "file_name": str 
}
```

For "single" box variants, there is only a single entry per label info.

For "multi" box variants, ther is a list of entries per label info which are in order (can be zipped).

## Discrete Positions

This is a bit tricky.

If we use too large discrete spaces, which is beneficial for "existince" prediction, then we cannot tell anymore the
relation of the boxes, because they all fall into the same "slot". Still, this might be a good start, because the "
falling into the same slot" should happen not too often on random placement, but actually happens for the interesting
positions of neighborhoods. This might actually be good features for neighborhood predictions then.

If we use too small spaces, which is beneficial for relation predictin (left of, on top of) then we cannot easily
predict the box position, but we have to predict multiple positions for each box (slot overlap). In an extreme case,
this would be object segmentation, where we predict each box class on each pixel. Since the boxes cannot overlap, this
might be actually good features for relation prediction.

Maybe the best is to combine both approaches, the "large" space for "What is there" and "What are neighbors", and the "
small" spaces for the actual "relational prediction".

Object detection algorithm actually try something similar: they propose bounding boxes on different resolutions and "
see" if they fit the ground truth.

## Variants

- single:
    - images with a single box (in sub-directories with the digit name)
    - digits range from 1 to 20 (incl.)
    - the boxes have the same color (green)
    - train: 1000 images for each digit (20,000 in total)
    - dev  : 100 images for each digit (2,000 in total)

- single-colored:
    - images with a single box (in sub-directories with the digit name)
    - digits range from 1 to 9 (incl.)
    - the boxes have different colors (according to their digits)
    - train: 1000 images for each digit (9,000 in total)
    - dev  : 100 images for each digit (900 in total)

- multi:
    - images with multiple boxes (in sub-directories with the box count, starting from 2)
    - digits range from 1 to 20 (incl.)
    - the boxes have the same color (green)
    - train: 1000 images for each count (19,000 in total)
    - dev  : 100 images for each count (1,900 in total)

- multi-colored:
    - images with multiple boxes (in sub-directories with the box count, starting from 2)
    - digits range from 1 to 9 (incl.)
    - the boxes have different colors (according to their digits)
    - train: 1000 images for each count (8,000 in total)
    - dev  : 100 images for each count (800 in total)

## Block Colors

The colors are given as `"digit" : [RED, GREEN, BLUE, ALPHA]` with values between `[0., 1.]`.

```
"block_colors": {
      "1": [1.0, 0.0, 1.0, 1.0],
      "2": [0.9, 0.1, 0.0, 1.0],
      "3": [0.8, 0.2, 1.0, 1.0],
      "4": [0.7, 0.3, 0.0, 1.0],
      "5": [0.6, 0.4, 1.0, 1.0],
      "6": [0.5, 0.5, 0.0, 1.0],
      "7": [0.4, 0.6, 1.0, 1.0],
      "8": [0.3, 0.7, 0.0, 1.0],
      "9": [0.2, 0.8, 1.0, 1.0]
    }
```