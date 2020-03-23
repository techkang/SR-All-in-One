"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import cv2
import numpy as np
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def rgb_to_ycbcr(img, only_y=True):
    dtype = img.dtype
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if dtype == np.uint8:
        rlt = np.clip(rlt, 0, 255).astype(dtype)
    return rlt


def _convert_image(image, bolder, gray):
    image = image[bolder:-bolder, bolder:-bolder]
    image = image.astype(np.float32)
    if gray and image.shape[2] == 3:
        image = rgb_to_ycbcr(image)
    return image


def cal_psnr(image1, image2, bolder=0, color_range=255, gray=False):
    image1 = _convert_image(image1, bolder, gray)
    image2 = _convert_image(image2, bolder, gray)
    return cv2.PSNR(image1, image2, R=color_range)


def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def cal_ssim(image1, image2, bolder=0, gray=False):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not image1.shape == image2.shape:
        raise ValueError('Input images must have the same dimensions.')
    image1 = _convert_image(image1, bolder, gray)
    image2 = _convert_image(image2, bolder, gray)
    if image1.ndim == 2:
        return ssim(image1, image2)
    elif image1.ndim == 3:
        if image1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(image1, image2))
            return np.array(ssims).mean()
        elif image1.shape[2] == 1:
            return ssim(np.squeeze(image1), np.squeeze(image2))
    else:
        raise ValueError(f'Wrong input image dimensions: {image1.shape}.')
