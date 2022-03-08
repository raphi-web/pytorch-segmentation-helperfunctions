import torch
from torch.utils import data
import numpy as np

import os
import numpy as np
import albumentations as A
import rasterio
from sklearn.model_selection import train_test_split

import albumentations as A
from rasterio.plot import reshape_as_image, reshape_as_raster

from torch.utils.data import DataLoader


def augmentation():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    return transform


class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 augment=False,
                 nclasses=2

                 ):
        self.inputs = inputs
        self.targets = targets
        self.augment = augment
        self.inputs_dtype = torch.float
        self.targets_dtype = torch.float
        self.nclasses = nclasses

        self.data_list = [i.split("/")[-1] for i in self.inputs]
        self.label_list = [i.split("/")[-1] for i in self.targets]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target

        with rasterio.open(input_ID) as src:
            x = src.read().astype(np.float32)

        with rasterio.open(target_ID) as src:
            y = src.read(1).astype(np.float32)

            if self.nclasses > 1:
                y = [(y == v) for v in range(self.nclasses)]
                y = np.stack(y, axis=0).astype('float')

        # Preprocessing
        if self.augment:
            x = reshape_as_image(x)

            if self.nclasses > 1:
                y = reshape_as_image(y)

            transformed = self.augment(image=x, mask=y)

            x = transformed["image"]
            y = transformed["mask"]

            x = reshape_as_raster(x)

            if self.nclasses > 1:
                y = reshape_as_raster(y)
            else:
                y = np.expand_dims(y, 0)

        # Typecasting
        x = torch.from_numpy(x).type(self.inputs_dtype)
        y = torch.from_numpy(y).type(self.targets_dtype)

        return x, y


def train_folders(input_dir, target_dir):
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".tif")
        ])

    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".tif") and not fname.startswith(".")
        ])

    return input_img_paths, target_img_paths


def train_val_split(inputs, targets, random_seed=42, train_size=0.8, shuffle=True):

    dataset = list(zip(inputs, targets))

    dataset_train, dataset_valid = train_test_split(
        dataset,
        random_state=random_seed,
        train_size=train_size,
        shuffle=shuffle)

    inputs_train, targets_train = list(zip(*dataset_train))
    inputs_valid, targets_valid = list(zip(*dataset_valid))

    return inputs_train, inputs_valid, targets_train, targets_valid


def gen_dataloader(input_dir, target_dir, nclasses=1, batch_size=2, augment=False, input_dtype=torch.float32, target_dtype=torch.float):

    inputs, targets = train_folders(input_dir, target_dir)
    inputs_train, inputs_valid, targets_train, targets_valid = train_val_split(
        inputs, targets)

    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        augment=augment,
                                        nclasses=nclasses,
                                        input_dtype=input_dtype,
                                        target_dtype=target_dtype)

    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        augment=augment,
                                        nclasses=nclasses,
                                        input_dtype=input_dtype,
                                        target_dtype=target_dtype)

    dataloader_training = DataLoader(dataset_train,
                                     batch_size=batch_size,
                                     shuffle=True)

    dataloader_validation = DataLoader(dataset=dataset_valid,
                                       batch_size=batch_size,
                                       shuffle=True)

    return dataloader_training, dataloader_validation


def pad(rast, size):
    result = np.empty((rast.shape[0],) + size)
    for idx, band in enumerate(rast):
        narr = np.zeros(size)
        narr[:band.shape[0], :band.shape[1]] = band
        result[idx] = narr

    return result


def vrt_pad(src, tile_shape):

    rast = src.read()
    _, rows, cols = rast.shape
    left, bottom, right, top = src.bounds

    resolution = (top - bottom) / rows

    tile_height, tile_width = tile_shape
    if rows % tile_height == 0:
        new_rows = rows

    else:
        new_rows = (rows - rows % tile_height) + tile_height

    if cols % tile_width == 0:
        new_cols = cols

    else:
        new_cols = (cols - cols % tile_width) + tile_width

    new_bottom = bottom - (new_rows - rows) * resolution
    new_right = right + (new_cols - cols) * resolution

    profile = src.profile
    profile.update(
        bounds=(left, new_bottom, new_right, top),
        width=new_cols,
        height=new_rows
    )

    return pad(rast, (new_rows, new_cols)), profile


def padTileIterator(rast, func, tile_shape, output_channels=1, mean=False):
    """
    iterates over a raster in tiles of given shape, 
    developed for predicting with klassifiers and semantic segmentation
    this function first padds the raster with zeros than iterates over padded raster
    so that only the inner most pixels of the tile are saved in the output raster. 
    This is done to always use the center pixel results of a classifier prediction of a tile.
    Because detection of Objects along the edges of a tile proves difficult!

    """

    bands, rast_rows, rast_cols = rast.shape
    tile_rows, tile_cols = tile_shape

    # padding equals half of tile size
    pad_offset_row = tile_rows // 2
    pad_offset_col = tile_cols // 2

    # create empty raster with additional padding
    padded_raster = np.zeros(
        (bands, rast_rows + tile_shape[0], rast_cols + tile_shape[1]))

    # replace zeros with mean for experimental prediction results,
    # maby it works better with some classifyers
    if mean:
        for idx, band in enumerate(rast):
            padded_raster[idx, padded_raster[idx, :, :, ] == 0] = band.mean()

    # fill the padded raster with the input raster so that the
    # input raster is aliged in the center of the padded raster
    padded_raster[:, pad_offset_row: -pad_offset_row,
                  pad_offset_col: - pad_offset_col] = rast[:]

    # create empty array to store results
    result_rast = np.zeros((output_channels, rast_rows, rast_cols))

    # calculate the number of tiles needed tile shape of (128,128)
    # in padded raster will be cut down to (64,64) in result raster

    num_tile_rows = rast_rows // pad_offset_row
    num_tile_cols = rast_cols // pad_offset_col

    # iterate over tiles
    for r in range(num_tile_rows):
        row_idx = r * pad_offset_row  # start-row index of input raster
        # start-row index of result raster
        pad_row_idx = r * pad_offset_row + pad_offset_row // 2

        for c in range(num_tile_cols):
            col_idx = c * pad_offset_col
            pad_col_idx = c * pad_offset_col + pad_offset_col // 2

            # get tile from padded_raster, perform input function on it
            result = func(padded_raster[:, pad_row_idx:  pad_row_idx +
                          tile_rows, pad_col_idx: pad_col_idx + tile_rows])

            # keep only the half in the center and add it to result array
            # expl: result-shape (128,128) ---> (32:69,32:69) == (64,64)
            result_rast[:, row_idx:row_idx + pad_offset_row,
                        col_idx:col_idx + pad_offset_col] = result[:, pad_offset_row // 2: - pad_offset_row//2,
                                                                   pad_offset_col // 2: -pad_offset_col // 2]

    return result_rast


def generate_classifier(model):
    def pred_func(x):
        x = torch.unsqueeze(torch.from_numpy(x), 0).to("cuda")

        y = model(x.float())
        y = np.array(y[0].cpu().detach())
        y = np.argmax(y, 0)
        return np.expand_dims(y, 0)

    return pred_func
