import logging
from pathlib import Path

import numpy as np
import torch.utils.data as data
from skimage import io

from data.common import sub_mean, set_channel
from data.data_prepare import DataGenerator
from tools.matlabimresize import imresize


class MultiScaleNumpyDataset(data.Dataset):
    def __init__(self, cfg, is_train):
        self.folders = cfg.datasets.train if is_train else cfg.datasets.test
        self.out_channels = cfg.model.out_channels
        datasets = [Path(cfg.datasets.path, folder, 'Augment') for folder in self.folders]
        dataset = datasets[0]
        assert cfg.upscale_factor == 4, f'upscale mast be 4, but now is {cfg.upscale_factor}'
        if cfg.datasets.re_generate:
            DataGenerator(cfg)
        if not dataset.is_dir() or not dataset.joinpath('LR', f'x{cfg.upscale_factor}').is_dir():
            DataGenerator(cfg)
        self.labels = sorted(list((i for d in datasets for i in (d / 'HR' / f'x{cfg.upscale_factor}').iterdir())))
        logging.info(f'Using MultiScaleNumpyDataset, total images:{len(self)}')

    def __getitem__(self, item):
        label = np.load(str(self.labels[item]))
        return set_channel(label, self.out_channels)

    def __len__(self):
        return len(self.labels)


class NumpyDataset(data.Dataset):
    def __init__(self, cfg, is_train):
        self.folders = cfg.datasets.train if is_train else cfg.datasets.test
        self.in_channels = cfg.model.in_channels
        self.out_channels = cfg.model.out_channels
        datasets = [Path(cfg.datasets.path, folder, 'Augment') for folder in self.folders]
        dataset = datasets[0]
        if cfg.datasets.re_generate:
            DataGenerator(cfg)
        if not dataset.is_dir() or not dataset.joinpath('LR', f'x{cfg.upscale_factor}').is_dir():
            DataGenerator(cfg)
        self.images = sorted(list((i for d in datasets for i in (d / 'LR' / f'x{cfg.upscale_factor}').iterdir())))
        self.labels = sorted(list((i for d in datasets for i in (d / 'HR' / f'x{cfg.upscale_factor}').iterdir())))
        logging.info(f'Using NumpyDataset, total images:{len(self)}')

    def __getitem__(self, item):
        image = set_channel(np.load(str(self.images[item])), self.in_channels)
        label = set_channel(np.load(str(self.labels[item])), self.out_channels)
        return image, label

    def __len__(self):
        return len(self.images)


class SimpleTestDataset(data.Dataset):
    def __init__(self, cfg, is_train):
        self.scale = cfg.upscale_factor
        self.in_channels = cfg.model.in_channels
        self.out_channels = cfg.model.out_channels
        self.input_size = cfg.datasets.input_size
        self.method = cfg.datasets.interpolation
        dataset = cfg.datasets.train if is_train else cfg.datasets.test
        self.folders = [Path(cfg.datasets.path, i) for i in dataset]
        self.labels = [i for folder in self.folders for i in folder.iterdir()]
        assert len(self)

    def interpolation(self, image, scale):
        # size = np.array(image.shape[:2]) * scale
        # size = tuple(size.astype(np.int))
        # size_dict = {'height': size[0], 'width': size[1]}
        # if self.method == 'bicubic':
        #     aug = iaa.Resize(size_dict, interpolation=cv2.INTER_CUBIC)
        # else:
        #     aug = iaa.Sequential([
        #         iaa.blur.GaussianBlur(sigma=1.6),
        #         iaa.Resize(size_dict, interpolation=cv2.INTER_NEAREST)
        #     ]
        #     )
        # result_image = aug.augment_image(image)
        result_image = imresize(image, scale)
        return result_image

    def __getitem__(self, item):
        label = set_channel(io.imread(str(self.labels[item])))
        height, width, c = label.shape
        scale = self.scale * 2
        label = label[:height // scale * scale, :width // scale * scale]
        image = self.interpolation(label, scale=1 / self.scale)
        return set_channel(sub_mean(image), self.in_channels), set_channel(sub_mean(label), self.out_channels)

    def __len__(self):
        return len(self.labels)
