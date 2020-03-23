import logging
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm

from config import default_argument_parser, get_cfg, default_setup
from data.common import sub_mean
from tools.matlabimresize import imresize


class DataGenerator(object):
    def __init__(self, cfg):
        self.method = cfg.datasets.interpolation
        self.folders = cfg.datasets.train
        self.dataset_path = Path(cfg.datasets.path)
        self.train_folder = Path(cfg.datasets.train_folder)
        self.scale = cfg.upscale_factor
        self.input_size = cfg.datasets.input_size
        self.channel = 1 if cfg.datasets.gray else 3

        if self.channel == 3:
            self.rgb_mean = np.array((0.4488, 0.4371, 0.4040)).reshape((1, 3, 1, 1))
        else:
            self.rgb_mean = np.array(0.5)
        self.thread_num = cfg.dataloader.num_workers
        self.reshape_size = [1., 0.8, 0.7, 0.6, 0.5]
        self.hr_folder = None
        self.lr_folder = None

        self.pre_process()

    def pre_process(self):
        logging.info(f'start to process dataset:{self.folders}')
        hr_folders = [self.dataset_path / d / 'Augment' / 'HR' / f'x{self.scale}' for d in self.folders]
        lr_folders = [self.dataset_path / d / 'Augment' / 'LR' / f'x{self.scale}' for d in self.folders]
        logging.info(f'Using multing processing with {self.thread_num} threads.')
        for i, (hr_folder, lr_folder) in enumerate(zip(hr_folders, lr_folders)):
            self.hr_folder = hr_folder
            self.lr_folder = lr_folder
            shutil.rmtree(hr_folder, ignore_errors=True)
            shutil.rmtree(lr_folder, ignore_errors=True)
            hr_folder.mkdir(parents=True)
            lr_folder.mkdir(parents=True)
            files = self.dataset_path / self.folders[i] / self.train_folder
            files = list(files.iterdir())
            start = time.perf_counter()
            pool = Pool(self.thread_num)
            for _ in tqdm(pool.imap(self.process, zip(files)), total=len(files)):
                pass
            pool.close()
            pool.join()
            logging.info(f'Finish generate dataset {self.folders[i]}, using {time.perf_counter() - start:.2f}s.')

    def process(self, files):
        file = files[0]
        source = io.imread(str(file))
        for i, size in enumerate(self.reshape_size):
            height, width, channel = source.shape
            scale = self.input_size * self.scale / size
            if height < scale or width < scale:
                continue
            image = imresize(source, scale=size)
            split = self.split_image(image)
            label = self.image_aug(split)
            lr_image = self.interpolation(label, 1 / self.scale)
            self.save(sub_mean(label), self.hr_folder / (file.stem + f'_{i}'))
            self.save(sub_mean(lr_image), self.lr_folder / (file.stem + f'_{i}'))

    def split_image(self, image: np.array):
        height, width, channel = image.shape
        scale = self.input_size * self.scale
        image = image[:height // scale * scale, :width // scale * scale]
        split = image.reshape(height // scale, scale, width // scale, scale, channel)
        split = split.transpose(0, 2, 1, 3, 4).reshape(-1, scale, scale, channel)
        return split

    def image_aug(self, images):
        def aug(image):
            h_flip = np.random.rand() < 0.5
            v_flip = np.random.rand() < 0.5
            rot90 = np.random.rand() < 0.5
            if h_flip:
                image = image[:, ::-1, :]
            if v_flip:
                image = image[::-1, :, :]
            if rot90:
                image = image.transpose(1, 0, 2)
            return image

        return np.array([aug(i) for i in images])

    def save(self, images, name):
        for i, image in enumerate(images):
            np.save(f'{name}_{str(i).zfill(3)}', image)

    def interpolation(self, images, scale):
        result_images = np.array([imresize(i, scale=scale) for i in images])
        return result_images


class PostProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        dataset = cfg.datasets.train or cfg.datasets.test
        self.dataset = Path('dataset', dataset)
        self.process()

    def process(self):
        images = np.load(str(self.dataset / 'images.npy'))
        labels = np.load(str(self.dataset / 'labels.npy'))
        for i, image in enumerate(tqdm(images)):
            if image.var() <= 1:
                color = np.random.randint(0, 255, (3, 1, 1))
                images[i] = ((images[i] * 0) + 1) * color
                labels[i] = ((labels[i] * 0) + 1) * color
        np.save(str(self.dataset / 'images'), images)
        np.save(str(self.dataset / 'labels'), labels)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':
    arg = default_argument_parser().parse_args()
    print(f'Command Line Args: {arg}')
    config = setup(arg)
    generator = DataGenerator(config)
