import logging
import shutil
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorboardX
import torch as t
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
from torch.nn.parallel import DataParallel
from tqdm import tqdm

import model
from data import SimpleTestDataset, NumpyDataset, MultiScaleNumpyDataset
from data.common import add_mean
from tools.checkpointer import Checkpointer
from tools.comm import cal_psnr, cal_ssim
from tools.matlabimresize import imresize


class Trainer:
    def __init__(self, cfg, resume):
        self.loss_thresh = 1000
        self.cfg = cfg
        self.device = t.device(cfg.device)
        self.model = self.build_model(cfg)
        self.output_dir = Path(cfg.output_dir)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.train_loader = self.build_data_loader(cfg, True)
        self.test_loader = self.build_data_loader(cfg, False)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.loss_function = self.build_loss_function(cfg)
        self.checkpointer = Checkpointer(self.model, self.output_dir / 'checkpoint' / cfg.model.name,
                                         optimizer=self.optimizer,
                                         scheduler=self.scheduler)
        self.start_iter = self.resume_or_load(resume)
        folder = self.output_dir / 'tbfile'
        if cfg.tensorboardX.clear_before:
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir()
            self.storage = tensorboardX.SummaryWriter(self.output_dir / 'tbfile')
        else:
            if cfg.tensorboardX.name:
                folder = folder / cfg.tensorboardX.name
            else:
                folder = folder / datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
            self.storage = tensorboardX.SummaryWriter(folder)
        self.iter = self.max_iter = cfg.solver.max_iter
        if cfg.solver.bias_loss:
            self.bias_weight = t.tensor(cfg.solver.bias_weight, device=self.device).reshape(1, 3, 1, 1)
        else:
            self.bias_weight = t.tensor(1, device=self.device)

        self.model.train()

        self._data_loader_iter = None
        self.all_metrics_list = []
        self.all_image_dict = {}
        self.baseline_dict = {}
        self.tqdm = None

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        return (
                self.checkpointer.resume_or_load(self.cfg.model.weights, resume=resume).get(
                    "iteration", self.cfg.start_iter - 1
                )
                + 1
        )

    def build_model(self, cfg):
        net = getattr(model, cfg.model.name)(cfg)
        logging.info('finish initialing model.')
        net.to(self.device)
        logging.info(f'load model to {self.device}')
        if cfg.device == 'cuda' and cfg.num_gpus > 1:
            logging.info('using DataParallel')
            net = DataParallel(net)
        logging.info(f'model: {cfg.model.name} loaded.')
        return net

    def build_optimizer(self, cfg, net):
        optimizer = getattr(t.optim, cfg.optimizer.name)
        return optimizer(net.parameters(), cfg.optimizer.lr)

    def build_data_loader(self, cfg, is_train):
        if is_train:
            dataset = NumpyDataset(self.cfg, is_train=is_train)
            dataloader = data.DataLoader(dataset, cfg.dataloader.batch_size * cfg.num_gpus,
                                         shuffle=True,
                                         num_workers=cfg.dataloader.num_workers,
                                         pin_memory=True, drop_last=True)
        else:
            dataset = SimpleTestDataset(self.cfg, is_train=False)
            dataloader = data.DataLoader(dataset, 1,
                                         shuffle=False,
                                         num_workers=1,
                                         pin_memory=True, drop_last=False)
        return dataloader

    def build_lr_scheduler(self, cfg, optim):
        step_lr = t.optim.lr_scheduler.StepLR
        scheduler = step_lr(optim, cfg.lr_scheduler.step_size,
                            cfg.lr_scheduler.gamma)
        return scheduler

    def build_loss_function(self, cfg):
        loss_function = getattr(F, cfg.solver.loss)
        if self.cfg.solver.ohem_k > 0:
            def ohem(pred, label):
                k = self.cfg.dataloader.batch_size * self.cfg.solver.ohem_k
                loss = getattr(F, cfg.solver.loss)(pred, label, reduction='none')
                return t.topk(loss.reshape(-1), k)[0].mean()

            loss_function = ohem

        return loss_function

    def to(self, obj, device):
        if hasattr(obj, 'to'):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.to(v, device) for k, v in obj.items()}
        elif isinstance(obj, Iterable):
            return type(obj)([self.to(item, device) for item in obj])
        else:
            logging.warning(f'object {obj} can not be moved to device {device}!')
            return obj

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        try:
            if not self._data_loader_iter:
                raise StopIteration
            batch = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.train_loader)
            batch = next(self._data_loader_iter)
        batch = self.to(batch, self.device)
        data_time = time.perf_counter() - start

        lr_image, hr_image = batch
        sr_image = self.model(lr_image)
        losses = self.loss_function(sr_image, hr_image)
        self._detect_anomaly(losses)
        self.store_image(batch, sr_image)

        self.optimizer.zero_grad()
        losses = losses.mean()
        losses.backward()
        postfix_dict = {'loss': f'{losses:.4f}'}
        self.optimizer.step()
        self.tqdm.set_postfix(postfix_dict)

        metrics_dict = {'loss': losses, "data_time": data_time}
        self._write_metrics(metrics_dict)

        self.scheduler.step(None)

    def _detect_anomaly(self, losses):
        if not t.isfinite(losses).all():
            raise FloatingPointError(f"Loss became infinite or NaN at iteration"
                                     f"={self.iter}!\nlosses = {losses}")

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu() if isinstance(v, t.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        self.all_metrics_list.append(metrics_dict)

    def store_image(self, batch: List[t.Tensor], pred: t.Tensor):
        image, label, pred = self.prepare_for_tbx([*batch, pred], transpose=False)
        self.all_image_dict = {'image': image, 'label': label, 'pred': pred}

    def prepare_for_tbx(self, images, squeeze=False, transpose=True):
        if isinstance(images, t.Tensor):
            images = [images]
        images = [image.detach().cpu().numpy() for image in images]
        images = [add_mean(image, transpose=transpose) for image in images]
        if squeeze:
            images = [np.squeeze(image) for image in images]
        return images

    def train(self):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        try:
            self.before_train()
            for self.iter in range(self.start_iter, self.max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
        finally:
            self.after_train()

    def get_baseline(self):
        with t.no_grad():
            self.model.eval()
            folders = iter(self.test_loader.dataset.folders)
            total = 0
            for i, (lr_image, label) in enumerate(self.test_loader):
                if i == total:
                    test_color_psnr = []
                    test_gray_psnr = []
                    test_color_ssim = []
                    test_gray_ssim = []
                    folder = next(folders)
                    total += len(list(folder.iterdir()))
                lr_image, label = self.prepare_for_tbx([lr_image, label], squeeze=True)
                sr_image = imresize(lr_image, self.cfg.upscale_factor)
                test_gray_psnr.append(cal_psnr(sr_image, label, bolder=self.cfg.upscale_factor, gray=True))
                test_color_psnr.append(cal_psnr(sr_image, label, bolder=self.cfg.upscale_factor))
                test_gray_ssim.append(cal_ssim(sr_image, label, bolder=self.cfg.upscale_factor, gray=True))
                test_color_ssim.append(cal_ssim(sr_image, label, bolder=self.cfg.upscale_factor))
                if i == total - 1:
                    self.baseline_dict[f'{folder.stem}_baseline_psnr_gray'] = np.mean(test_gray_psnr)
                    self.baseline_dict[f'{folder.stem}_baseline_psnr_color'] = np.mean(test_color_psnr)
                    self.baseline_dict[f'{folder.stem}_baseline_ssim_gray'] = np.mean(test_gray_ssim)
                    self.baseline_dict[f'{folder.stem}_baseline_ssim_color'] = np.mean(test_color_ssim)
            self.model.train()

    def before_train(self):
        if not self.baseline_dict:
            self.get_baseline()
        logging.info('begin to train model.')
        self.tqdm = tqdm(total=self.cfg.solver.test_interval)
        self.tqdm.update(self.start_iter % self.cfg.solver.test_interval)
        self.tqdm.display('')

    def before_step(self):
        if not self.iter % self.cfg.solver.test_interval:
            self.test()
            self.tqdm.reset()

    def after_step(self):
        self.tqdm.update()
        if not self.iter % self.cfg.tensorboardX.save_freq:
            if "data_time" in self.all_metrics_list[0]:
                data_time = np.max([x.pop("data_time") for x in self.all_metrics_list])
                self.storage.add_scalar("data_time", data_time, self.iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in self.all_metrics_list]) for k in self.all_metrics_list[0].keys()
            }
            total_losses_reduced = metrics_dict['loss']
            self.all_metrics_list = []

            pred = tv.utils.make_grid(t.from_numpy(self.all_image_dict['pred']))
            label = tv.utils.make_grid(t.from_numpy(self.all_image_dict['label']))
            image = t.cat([pred, label], -1)
            self.storage.add_scalar("total_loss", total_losses_reduced, self.iter)
            self.storage.add_image('train_pred_label', image, self.iter)
            for k, v in metrics_dict.items():
                self.storage.add_scalar(k, v, self.iter)
        if not (self.iter + 1) % self.cfg.solver.save_interval:
            self.checkpointer.save(f'{self.cfg.model.name}_SR{self.cfg.upscale_factor}_{self.iter + 1}')

    def test(self):
        logging.info(f'iter: {self.iter} testing...')
        with t.no_grad():
            self.model.eval()
            folders = iter(self.test_loader.dataset.folders)
            total = 0
            for i, (lr_image, label) in enumerate(self.test_loader):
                if i == total:
                    test_color_psnr = []
                    test_gray_psnr = []
                    test_color_ssim = []
                    test_gray_ssim = []
                    folder = next(folders)
                    total += len(list(folder.iterdir()))
                lr_image, label = self.to([lr_image, label], self.device)
                pred = self.model(lr_image)

                pred, label = self.prepare_for_tbx([pred, label], squeeze=True)
                test_color_psnr.append(cal_psnr(pred, label, bolder=self.cfg.upscale_factor))
                test_gray_psnr.append(cal_psnr(pred, label, bolder=self.cfg.upscale_factor, gray=True))
                test_color_ssim.append(cal_ssim(pred, label, bolder=self.cfg.upscale_factor))
                test_gray_ssim.append(cal_ssim(pred, label, bolder=self.cfg.upscale_factor, gray=True))

                if i < self.cfg.tensorboardX.image_num:
                    image_to_show = np.concatenate([pred, label], axis=1)
                    self.storage.add_image(f'test_{folder.stem}_pred_label_{i}', image_to_show,
                                           self.iter, dataformats='HWC')
                if i == total - 1:
                    logging.info(f'{folder.stem:<10} ssim: {np.mean(test_gray_ssim):.4f}, '
                                 f'psnr: {np.mean(test_gray_psnr):.4f}, '
                                 f'baseline: ssim: {self.baseline_dict[f"{folder.stem}_baseline_ssim_gray"]:.4f}, '
                                 f'psnr: {self.baseline_dict[f"{folder.stem}_baseline_psnr_gray"]:.4f}')
                    psnr_dict = {f'{folder.stem}_psnr_gray': np.mean(test_gray_psnr)}
                    ssim_dict = {f'{folder.stem}_ssim_gray': np.mean(test_gray_ssim)}
                    self.storage.add_scalars('test_psnr', psnr_dict, self.iter)
                    self.storage.add_scalars('test_ssim', ssim_dict, self.iter)
            # TODO: delete or uncomment this line
            # self.storage.add_scalars('test_psnr', self.baseline_dict, self.iter)
            self.model.train()

    def after_train(self):
        self.iter += 1
        self.after_step()
        if not self.iter % self.cfg.solver.test_interval:
            self.test()
        logging.info('train finished.')


class MetaTrainer(Trainer):
    def build_data_loader(self, cfg, is_train):
        if is_train:
            dataset = MultiScaleNumpyDataset(self.cfg, is_train=is_train)
            dataloader = data.DataLoader(dataset, cfg.dataloader.batch_size * cfg.num_gpus,
                                         shuffle=True,
                                         num_workers=cfg.dataloader.num_workers,
                                         pin_memory=False, drop_last=True)
        else:
            dataset = SimpleTestDataset(self.cfg, is_train=False)
            dataloader = data.DataLoader(dataset, 1,
                                         shuffle=False,
                                         num_workers=1,
                                         pin_memory=True, drop_last=False)
        return dataloader

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        try:
            if not self._data_loader_iter:
                raise StopIteration
            hr_image = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.train_loader)
            hr_image = next(self._data_loader_iter)
        hr_image = self.to(hr_image, self.device)
        data_time = time.perf_counter() - start

        scale = np.random.randint(2, 3, 1) * 2
        # scale = np.array([4])
        out_h, out_w = hr_image.shape[2:]
        lr_image = t.nn.functional.interpolate(hr_image, (int(out_h / scale), int(out_w / scale)), mode='bicubic',
                                               align_corners=False)
        sr_image = self.model(lr_image, (out_h, out_w, scale))
        losses = self.loss_function(sr_image, hr_image, reduction='none')
        self._detect_anomaly(losses)
        self.store_image([lr_image, hr_image], sr_image)

        self.optimizer.zero_grad()
        if losses.max() < self.loss_thresh:
            self.loss_thresh = losses.max().detach().cpu() * 10
            losses = losses.mean()
            losses.backward()
            postfix_dict = {'loss': f'{losses:.4f}', 'scale': f'{scale[0]:.2f}'}
            self.optimizer.step()
        else:
            postfix_dict = {'max loss': f'{losses.max():.4f}'}
            losses = losses.mean()
        self.tqdm.set_postfix(postfix_dict)

        metrics_dict = {'loss': losses, "data_time": data_time}
        self._write_metrics(metrics_dict)

        self.scheduler.step(None)
