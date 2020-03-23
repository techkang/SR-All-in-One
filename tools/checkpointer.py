import collections
import copy
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t
import torch.nn as nn
from termcolor import colored
from torch.nn.parallel import DataParallel, DistributedDataParallel


def _strip_prefix_if_present(state_dict: collections.OrderedDict, prefix: str):
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def get_missing_parameters_message(keys: list):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters are not in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: list):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint contains parameters not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta")
        for k, v in groups.items()
    )
    return msg


def _group_checkpoint_keys(keys: list):
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: list):
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


class Checkpointer(object):

    def __init__(
            self,
            model: nn.Module,
            save_dir: str = "",
            *,
            save_to_disk: bool = True,
            **checkpointables: object,
    ):
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the `state_dict()` and `load_state_dict()` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.logger = logging.getLogger(__name__)
        self.save_dir = Path(save_dir)
        self.save_to_disk = save_to_disk

    def save(self, name: str, **kwargs: dict):
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if self.save_dir == Path('') or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = name if name.endswith('.pth') else "{}.pth".format(name)
        self.logger.info(f"Saving checkpoint to {self.save_dir / basename}")
        if not self.save_dir.is_dir():
            self.save_dir.mkdir()
        with open(self.save_dir / basename, "wb") as f:
            t.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, path: str):
        """
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info(
                "No checkpoint found. Initializing model from scratch"
            )
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        assert Path(path), "Checkpoint {} not found!".format(path)

        checkpoint = t.load(path, map_location=t.device("cpu"))
        self._load_model(checkpoint)
        for key, obj in self.checkpointables.items():
            if key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = Path(self.save_dir, "last_checkpoint")
        return save_file.is_file()

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = Path(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return Path(self.save_dir, last_saved)

    def get_all_checkpoint_files(self):
        """
        Returns:
            list: All available checkpoint files (.pth files) in target
                directory.
        """
        return [file for file in self.save_dir.iterdir()
                if file.is_file() and file.suffix == ".pth"]

    def resume_or_load(self, path: str, *, resume: bool = True):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
        return self.load(path)

    def tag_last_checkpoint(self, last_filename_basename: str):
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        with open(self.save_dir / 'last_checkpoint', "w") as f:
            f.write(last_filename_basename)

    def _load_model(self, checkpoint):
        """
        Load weights from a checkpoint.
        Args:
            checkpoint (Any): checkpoint contains the weights.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    self.logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(
                            k, shape_checkpoint, shape_model
                        )
                    )
                    checkpoint_state_dict.pop(k)

        incompatible = self.model.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        if incompatible.missing_keys:
            self.logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    def _convert_ndarray_to_tensor(self, state_dict: dict):
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(
                    v, t.Tensor
            ):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(
                        k, type(v)
                    )
                )
            if not isinstance(v, t.Tensor):
                state_dict[k] = t.from_numpy(v)
