import numpy as np
import torch
from skimage.io import imread
from torch.utils import data
from tqdm.notebook import tqdm


class SegmentationDataSet1(data.Dataset):
    """Most basic image segmentation dataset."""

    def __init__(self, inputs: list, targets: list, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y


class SegmentationDataSet2(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        targets: list,
        transform=None,
        use_cache: bool = False,
        pre_transform=None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = imread(str(img_name)), imread(str(tar_name))
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y


class SegmentationDataSet3(data.Dataset):
    """Image segmentation dataset with caching, pretransforms and multiprocessing."""

    def __init__(
        self,
        inputs: list,
        targets: list,
        transform=None,
        use_cache: bool = False,
        pre_transform=None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool() as pool:
                self.cached_data = pool.starmap(
                    self.read_images, zip(inputs, targets, repeat(self.pre_transform))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(str(inp)), imread(str(tar))
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar


class SegmentationDataSet4(data.Dataset):
    """Image segmentation dataset with caching, pretransforms and multiprocessing. Output is a dict."""

    def __init__(
        self,
        inputs: list,
        targets: list,
        transform=None,
        use_cache: bool = False,
        pre_transform=None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool() as pool:
                self.cached_data = pool.starmap(
                    self.read_images, zip(inputs, targets, repeat(self.pre_transform))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return {
            "x": x,
            "y": y,
            "x_name": self.inputs[index].name,
            "y_name": self.targets[index].name,
        }

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(str(inp)), imread(str(tar))
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar


class SegmentationDataSetRandom(data.Dataset):
    """Random image segmentation dataset for testing purposes."""

    def __init__(
        self,
        num_samples,
        size,
        num_classes: int = 4,
        inputs_dtype=torch.float32,
        targets_dtype=torch.long,
    ):
        self.num_samples = num_samples
        self.size = size
        self.num_classes = num_classes
        self.inputs_dtype = inputs_dtype
        self.targets_dtype = targets_dtype
        self.cached_data = []

        # Generate some random input target pairs
        for num in range(self.num_samples):
            inp = torch.from_numpy(np.random.uniform(low=0, high=1, size=size))
            tar = torch.randint(low=0, high=num_classes, size=size[1:])
            self.cached_data.append((inp, tar))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        x, y = self.cached_data[index]

        # Typecasting
        x, y = x.type(self.inputs_dtype), y.type(self.targets_dtype)

        return {
            "x": x,
            "y": y,
            "x_name": f"x_name_{index}",
            "y_name": f"y_name_{index}",
        }
