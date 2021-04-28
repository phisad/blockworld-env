"""
Created on 20.08.2020

@author: Philipp
"""
from torch.utils import data
import numpy as np

import env
from env import BIN_COUNT


class SequentialSingleRandomBlockEnvDataset(data.IterableDataset):  # noqa
    """
        This dataset produces random positioning of blocks on the Block2d2d env board.
    """

    def __init__(self, task, env, params, device):
        self.env = env
        self.task = task
        self.device = device
        self.num_samples_max = int(params["num_samples"])
        self.num_samples_yield = -1
        self.current_epoch = -1
        self.num_classes = task["num_classes"]

    def __next__(self):
        self.num_samples_yield += 1
        if self.num_samples_yield >= self.num_samples_max:
            raise StopIteration
        block_id = self.current_epoch  # use no-block as zero-label
        if self.current_epoch == 0:
            image = self.env.reset(mode="rgb_array", params={
                env.PARAM_NUM_BLOCKS: 0  # no-blocks
            })
        else:
            image = self.env.reset(mode="rgb_array", params={
                env.PARAM_NUM_BLOCKS: [block_id]
            })
        bbox = [-1, -1]
        block_pos = [-1, -1]
        block_pos_discrete = [-1, -1]
        if self.current_epoch > 0:  # skip no-blocks epoch
            bbox = self.env.get_bbox(block_id)
            block_pos = self.env.get_pos(block_id)
            # Note: We use 10 bins, so we can split the image in 10 parts of 64 x 64 size.
            block_pos_discrete = self.env.get_pos_discrete(block_id, bins=BIN_COUNT)
            # block_color = self.env.get_color(block_id)  # is already a list (not a tuple)
        return image, {"block_id": block_id, "bbox": list(bbox), "block_pos": list(block_pos),
                       "block_pos_discrete": list(block_pos_discrete),
                       # "block_color": block_color
                       }

    def __iter__(self):
        """
        Called once each epoch to iterate over the dataset.

        :return: the samples generator
        """
        self.current_epoch = self.current_epoch + 1
        self.num_samples_yield = -1
        return self


class IncrementalMultiRandomBlockEnvDataset(data.IterableDataset):  # noqa
    """
        This dataset produces random positioning of blocks on the Block2d2d env board.
    """

    def __init__(self, task, env, params, device):
        self.env = env
        self.task = task
        self.device = device
        self.num_samples_max = int(params["num_samples"])
        self.num_samples_yield = -1
        self.current_epoch = 0
        self.num_classes = task["num_classes"]

    def __next__(self):
        self.num_samples_yield += 1
        if self.num_samples_yield >= self.num_samples_max:
            raise StopIteration
        num_blocks = self.current_epoch + 1
        block_ids = np.random.choice(range(1, self.num_classes + 1), size=num_blocks, replace=False)
        image = self.env.reset(mode="rgb_array", params={
            env.PARAM_NUM_BLOCKS: block_ids
        })
        bbox = []
        block_pos = []
        block_pos_discrete = []
        block_ids = block_ids.tolist()
        # block_color = []
        for block_id in block_ids:
            bbox.append(list(self.env.get_bbox(block_id)))
            block_pos.append(list(self.env.get_pos(block_id)))
            # Note: We use 10 bins, so we can split the image in 10 parts of 64 x 64 size.
            block_pos_discrete.append(list(self.env.get_pos_discrete(block_id, bins=BIN_COUNT)))
            # block_color.append(self.env.get_color(block_id))
        return image, {"block_id": block_ids, "bbox": bbox, "block_pos": list(block_pos),
                       "block_pos_discrete": list(block_pos_discrete),
                       # "block_color": block_color
                       }

    def __iter__(self):
        """
        Called once each epoch to iterate over the dataset.

        :return: the samples generator
        """
        self.current_epoch = self.current_epoch + 1
        self.num_samples_yield = -1
        return self
