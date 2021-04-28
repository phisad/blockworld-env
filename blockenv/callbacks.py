import os

import torch
import logging

import torchvision.transforms as transforms

from torchexpresso.callbacks import Callback
from torchexpresso.utils import mkdir_if_not_exists, store_json_to, image_to_patches_2d

logger = logging.getLogger(__file__)


class SaveImageByLabel(Callback):
    """
        Stores the images structured by the labels.

        Images are stored to a sub-directory called images with the directory name is the label.
        Annotations are not necessary, because implicitly given by the directory names.
    """

    def __init__(self, target_directory, split_name, name="save_img_by_lbl_id"):
        super().__init__(name)
        self.target_directory = target_directory
        self.split_name = split_name
        self.num_images = 0
        self.annotations_directory = os.path.join(target_directory, "annotations")
        self.images_directory = os.path.join(target_directory, "images")
        # self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=448)])
        self.transform = transforms.Compose([transforms.ToPILImage()])
        self.images_split_dir = os.path.join(self.images_directory, self.split_name)
        self.current_epoch = 0
        self.annotations = []

        mkdir_if_not_exists(self.images_directory)
        mkdir_if_not_exists(self.annotations_directory)
        mkdir_if_not_exists(self.images_split_dir)

    def on_epoch_start(self, phase, epoch):
        print("Store epoch: " + str(epoch))
        self.current_epoch = epoch
        # Well, we "know" that epoch is the label
        mkdir_if_not_exists(os.path.join(self.images_split_dir, str(epoch)))

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        for image, label in zip(inputs, labels):
            self.num_images = self.num_images + 1
            label_name = str(label["block_id"])
            file_name = "%s.png" % self.num_images
            image_directory = os.path.join(self.images_directory, self.split_name, label_name)
            # Add file name
            label["file_name"] = file_name
            self.annotations.append(label)
            with self.transform(image) as image_pil:
                image_file = os.path.join(image_directory, file_name)
                image_pil.save(image_file)

    def on_epoch_end(self, epoch):
        store_json_to(self.annotations, self.annotations_directory, self.split_name + ".json")


class SaveImagePatchByLabel(Callback):
    """
        Stores the image patch structured by the labels identified by the discrete position.

        Images are stored to a sub-directory called images with the directory name is the label.

        Note: This is actually only an image augmentation with shift translations!
                But we take the actualy values from the images.
    """

    def __init__(self, target_directory, split_name, name="save_img_patch_by_lbl_id"):
        super().__init__(name)
        self.target_directory = target_directory
        self.split_name = split_name
        self.num_images = 0
        self.annotations_directory = os.path.join(target_directory, "annotations")
        self.images_directory = os.path.join(target_directory, "images")
        # self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=448)])
        self.transform = transforms.Compose([transforms.ToPILImage()])
        self.images_split_dir = os.path.join(self.images_directory, self.split_name)
        self.current_label = 0
        self.annotations = []
        self.to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        mkdir_if_not_exists(self.images_directory)
        mkdir_if_not_exists(self.annotations_directory)
        mkdir_if_not_exists(self.images_split_dir)

    def on_epoch_start(self, phase, epoch):
        print("Store epoch: " + str(epoch))
        # Well, we "know" that epoch - 1 is the label
        self.current_label = epoch - 1
        mkdir_if_not_exists(os.path.join(self.images_split_dir, str(self.current_label)))

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        for image, label in zip(inputs, labels):
            self.num_images = self.num_images + 1
            label_name = str(label["block_id"])
            file_name = "%s.png" % self.num_images
            image_directory = os.path.join(self.images_directory, self.split_name, label_name)
            # Add file name
            label["file_name"] = file_name
            self.annotations.append(label)
            image = self.to_tensor(image)
            image_patches = image_to_patches_2d(image, num_patches=10)
            pos_x, pos_y = label["block_pos_discrete"]
            # print("x: %s y: %s" % (pos_x, pos_y))
            # Note: We have to translate the discrete positions from lower-left to upper-left coords
            pos_xt, pos_yt = pos_x, 9 - pos_y
            # print("xt: %s yt: %s" % (pos_xt, pos_yt))
            patch_idx = pos_x + pos_yt * 10  # The patches are "ordered" by height
            # print("idx: " + str(patch_idx))
            image_patch = image_patches[patch_idx]
            with self.transform(image_patch) as image_pil:
                image_file = os.path.join(image_directory, file_name)
                image_pil.save(image_file)

    def on_epoch_end(self, epoch):
        store_json_to(self.annotations, self.annotations_directory, self.split_name + ".json")


class SaveImageByLabelCount(Callback):
    """
        Stores the images structured by the label counts.

        Images are stored to a sub-directory called images with the directory name is the label count.
        Annotations are stored along in the sub-directory 'annotations'.
    """

    def __init__(self, target_directory, split_name, name="save_img_by_lbl_count"):
        super().__init__(name)
        self.target_directory = target_directory
        self.split_name = split_name
        self.num_images = 0
        self.images_directory = os.path.join(target_directory, "images")
        self.annotations_directory = os.path.join(target_directory, "annotations")
        self.images_split_dir = os.path.join(self.images_directory, self.split_name)
        self.annotations = []
        # self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=448)])
        self.transform = transforms.Compose([transforms.ToPILImage()])
        self.current_epoch = 0

        mkdir_if_not_exists(self.images_directory)
        mkdir_if_not_exists(self.annotations_directory)
        mkdir_if_not_exists(self.images_split_dir)

    def on_epoch_start(self, phase, epoch):
        print("Store epoch: " + str(epoch))
        self.current_epoch = epoch
        # Well, we "know" that epoch+1 is the label count
        mkdir_if_not_exists(os.path.join(self.images_split_dir, str(epoch + 1)))

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        for image, label in zip(inputs, labels):
            self.num_images = self.num_images + 1
            label_count = str(len(label["block_id"]))
            file_name = "%s.png" % self.num_images
            # Add file name
            label["file_name"] = file_name
            self.annotations.append(label)
            image_directory = os.path.join(self.images_directory, self.split_name, label_count)
            with self.transform(image) as image_pil:
                image_file = os.path.join(image_directory, file_name)
                image_pil.save(image_file)

    def on_epoch_end(self, epoch):
        store_json_to(self.annotations, self.annotations_directory, self.split_name + ".json")
