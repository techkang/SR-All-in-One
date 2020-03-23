import numpy as np

from tools.comm import rgb_to_ycbcr


def augment(*images, h_flip=True, rot=True):
    h_flip = h_flip and np.random.random() < 0.5
    v_flip = rot and np.random.random() < 0.5
    rot90 = rot and np.random.random() < 0.5

    def _augment(img):
        if h_flip:
            img = img[:, ::-1, :]
        if v_flip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)

        return img

    return [_augment(i) for i in images]


def sub_mean(image, channel=3, rgb_range=255):
    if channel == 3:
        rgb_mean = np.array((0.4488, 0.4371, 0.4040), np.float32)
    else:
        rgb_mean = np.array(0.5, np.float32)
    image = image.astype(np.float32) / rgb_range - rgb_mean
    if image.ndim == 4:
        image = image.transpose(0, 3, 1, 2)
    else:
        image = image.transpose(2, 0, 1)
    return image


def add_mean(image, channel=3, rgb_range=255, transpose=False):
    if channel == 3:
        rgb_mean = np.array((0.4488, 0.4371, 0.4040), np.float32)
    else:
        rgb_mean = np.array(0.5, np.float32)
    if transpose:
        if image.ndim == 4:
            image = image.transpose(0, 2, 3, 1)
        else:
            image = image.transpose(1, 2, 0)
    else:
        if image.ndim == 4:
            rgb_mean = rgb_mean.reshape((1, channel, 1, 1))
        else:
            rgb_mean = rgb_mean.reshape((channel, 1, 1))
    image = (image + rgb_mean) * rgb_range
    image = np.clip(image, 0, rgb_range)
    if rgb_range == 255:
        image = image.astype(np.uint8)
    return image


def set_channel(image, n_channels=3):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)

    c = image.shape[2]
    if n_channels == 1 and c == 3:
        image = np.expand_dims(rgb_to_ycbcr(image), 2)
    elif n_channels == 3 and c == 1:
        image = np.concatenate([image] * n_channels, 2)

    return image
