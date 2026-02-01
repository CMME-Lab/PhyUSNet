import numpy as np
import torch
import cv2
from torch.utils.data import Dataset as BaseDataset
import os
from transformers import SegformerImageProcessor


class PhyUSDataset(BaseDataset):
    def __init__(self, paths, image_processor, transform=None, device=None):
        # self.scans = scans  # Expecting shape: [Total_images, 1, 256, 256]
        # self.labels = labels  # Expecting shape: [Total_images, 1, 256, 256] (or [Total_images, 256,256])
        self.paths = paths
        self.transform = transform
        self.image_processor = image_processor
        self.id2label = {0: "background", 1: "object"}

    def __len__(self):
        return len(self.paths)

    def normalize_image(self, image):
        """Normalize image values to range [0, 1]"""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image

    def __getitem__(self, idx):
        # expect the paths to be a list of tuples: (input_image_path, mask_image_path, id)
        # --- Load raw data ---
        input_image_path = self.paths[idx][0]
        mask_image_path = self.paths[idx][1]
        id = self.paths[idx][2]
        assert os.path.exists(input_image_path)
        assert os.path.exists(mask_image_path)
        input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

        # Normalize images to [0, 1] range
        input_image = self.normalize_image(input_image)
        mask_image = self.normalize_image(mask_image)

        # --- Convert from CHW to HWC for Albumentations ---
        if len(input_image.shape) == 3 and input_image.shape[0] == 1:
            input_image = np.transpose(input_image, (1, 2, 0))  # becomes [256,256,1]
        elif len(input_image.shape) == 2:
            input_image = input_image[
                ..., np.newaxis
            ]  # Add channel dimension [256,256,1]

        if len(mask_image.shape) == 3 and mask_image.shape[0] == 1:
            mask_image = np.transpose(mask_image, (1, 2, 0))  # becomes [256,256,1]
        elif len(mask_image.shape) == 2:
            mask_image = mask_image[
                ..., np.newaxis
            ]  # Add channel dimension [256,256,1]

        # --- Apply Albumentations transform (if provided) ---
        if self.transform:
            try:
                augmented = self.transform(image=input_image, mask=mask_image)
                input_image = augmented["image"]
                mask_image = augmented["mask"]

            except Exception as e:
                print(
                    f"Warning: Augmentation failed at index {idx}: {e}, using original images"
                )
                # Keep original images if augmentation fails
                pass

        # --- Convert back to torch tensors and switch from HWC to CHW ---
        input_image = (
            torch.from_numpy(input_image).permute(2, 0, 1).float()
        )  # becomes [1,256,256]

        if len(mask_image.shape) == 3:
            mask_image = (
                torch.from_numpy(mask_image).permute(2, 0, 1).float().squeeze(0)
            )
        else:
            mask_image = torch.from_numpy(mask_image).float()

        # --- Process with image_processor ---
        encoded_inputs = self.image_processor(
            input_image, mask_image, return_tensors="pt"
        )

        # --- Adjust pixel_values to have 3 channels ---
        if encoded_inputs["pixel_values"].shape[1] == 1:
            encoded_inputs["pixel_values"] = encoded_inputs["pixel_values"].repeat(
                1, 3, 1, 1
            )
        # Squeeze the batch dimension if you want a single sample
        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key].squeeze(0)

        encoded_inputs["metadata"] = {
            "id": id,
            "input_image_path": input_image_path,
            "mask_image_path": mask_image_path,
        }

        return encoded_inputs


if __name__ == "__main__":
    DATASET_PATH = "/home/user/data/phyusformer_data/physics_data"
    dict_data = np.load(f"{DATASET_PATH}/paths.npz", allow_pickle=True)
    image_processor = SegformerImageProcessor(
        reduce_labels=False,
        do_normalize=False,
        do_rescale=False,
        size={"height": 256, "width": 256},
    )
    train_dataset = PhyUSDataset(
        dict_data["train"], transform=None, image_processor=image_processor
    )
    print("Successfully loaded dataset")
    print("data[pixel_values]: ", train_dataset[0]["pixel_values"].shape)
    # print(train_dataset[0]["pixel_values"].shape)
    print("data[labels]: ", train_dataset[0]["labels"].shape)
    print("data[metadata]", train_dataset[0]["metadata"])
