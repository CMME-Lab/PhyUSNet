"""
Quite similar to the test.py script, but with the cross domain generalization in mind
We will load the trained model on the benchamrak dataset adn then test it on all not only on the
benchmark dataset
"""

# External imports
import os
import random
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import torch  # type: ignore
import cv2  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
from transformers import SegformerImageProcessor  # type: ignore
from PIL import Image  # type: ignore
from matplotlib.patches import Patch
import segmentation_models_pytorch as smp

from tqdm import tqdm
import argparse
import gc


# Internal imports
from eval import compute_hd95

from unet import UNet
from segformer import SegFormer
from swinunet import SwinUnet
from transunet import TransUNet
from swinunet_config import get_config
from swinunet import parser as swin_parser


class ClinicalUltrasoundDataset(Dataset):
    """
    Dataset class for loading and preprocessing Breast Images (BUSI, STU, UDIAT, TN3K).

    Args:
        image_paths (list): List of paths to ultrasound images
        mask_paths (list): List of paths to mask images
        labels (list): List of labels for each image
        transform (callable, optional): Optional transform to be applied to samples
        image_processor (callable, optional): Image processor for segmentation model
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        labels,
        ids,
        transform=None,
        image_processor=None,
        shift_bbox=400,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.image_processor = image_processor
        self.id2label = {0: "background", 1: "object"}
        self.shift_bbox = shift_bbox

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image paths
        x_img_path = self.image_paths[idx]
        x_mask_path = self.mask_paths[idx]
        label = self.labels[idx]

        # Load and convert images
        image = Image.open(x_img_path).convert("RGB")
        image = np.array(image)
        mask = np.array(Image.open(x_mask_path), dtype=np.uint8)
        # convert rgb to gray scale mask
        try:
            if (mask.shape[2] == 3) or (mask.shape[2] == 4):
                # this will raise error in case the shape has only 2 dimensions
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        except:
            pass

        # Resize to standard dimensions
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())

        # Normalize mask
        if mask.max() > 0:
            mask = mask / mask.max()

        # Process with Segformer Image Processor
        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        # Add metadata
        metadata = {
            "image_path": x_img_path,
            "mask_path": x_mask_path,
            "label": label,
            "id": self.ids[idx],
            # "bbox": bbox,
        }
        encoded_inputs["metadata"] = metadata
        return encoded_inputs


def get_formatted_dataset(
    data: list[tuple[str, str, str, str]], H=256, W=256, shift_bbox=400
) -> tuple[ClinicalUltrasoundDataset, DataLoader]:
    """
    This function loads the data into a dataloader
    Args:
        data: list of tuples containing (image_path, mask_path, label, id)
        BATCH_SIZE: int, the batch size
    Returns:
        dataset: torch.utils.data.Dataset, the dataset
        loaded_data: torch.utils.data.DataLoader, the loaded data
    """
    image_processor = SegformerImageProcessor(
        reduce_labels=False,
        do_normalize=False,
        do_rescale=False,
        size={"height": H, "width": W},
    )
    dataset = ClinicalUltrasoundDataset(
        *zip(*data), image_processor=image_processor, shift_bbox=shift_bbox
    )

    return dataset



def filter_multi_lesions(mask):
    """Removes small lesions and keeps only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )

    if num_labels > 1:
        max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Find largest region
        return (labels == max_label).astype(np.uint8)  # Keep only the largest region

    return mask


def get_df(file_name=None, scans_path=None, masks_path=None):
    """
    This function loads the data into a dataframe
    Args:
        file_name: str, the name of the file to load : Example [os.path.join(dataset_path, f"MainPatient/train.txt")]
    Returns:
        df: pandas.DataFrame, the dataframe
    """
    df = pd.read_csv(file_name, header=None, dtype={0: str})
    df["id"] = df[0].astype(str)
    df["image_path"] = df["id"].apply(
        lambda x: os.path.join(scans_path, f"{x}") + ".png"
    )  # Strong Assumption that the images are png files
    df["mask_path"] = df["id"].apply(
        lambda x: os.path.join(masks_path, f"{x}") + ".png"
    )  # Strong Assumption that the masks are png files
    return df


def get_dataset(
    root_us30k_path: str = "",
    visualize: bool = False,
    dataset_key: str = "UDIAT",
    H=256,
    W=256,
    shift_bbox=400,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """
    This function loads the breast_udiat dataset, converts it into a test_loader, and returns the test_dataset and test_loader
    Args:
        root_us30k_path: str, the path to the US30k dataset
        visualize: bool, whether to visualize the dataset
        dataset_key: str, the key of the dataset to load ["BUSI", "STU", "UDIAT", "TN3K", "DDTI", "BUS-BRA"]
    Returns:
        train_dataset: torch.utils.data.Dataset, the train dataset
        val_dataset: torch.utils.data.Dataset, the val dataset
        test_dataset: torch.utils.data.Dataset, the test dataset
        df_train: pandas.DataFrame, the train dataframe
        df_val: pandas.DataFrame, the val dataframe
        df_test: pandas.DataFrame, the test dataframe
    """
    # Dictionary mapping for efficient dataset path lookup
    dataset_paths = {
        "STU": "Breast-STU",
        "BUSI": "Breast-BUSI",
        "UDIAT": "Breast-UDIAT",
        "TN3K": "ThyroidNodule-TN3K",
        "DDTI": "ThyroidNodule-DDTI",
        "BUS-BRA": "BUS-BRA/BUSBRA",
        "OvarianCyst": "OvarianCyst/MMOTU/OTU_2d",
    }

    if dataset_key not in dataset_paths:
        raise ValueError(f"Source dataset {dataset_key} not supported")

    dataset_path = os.path.join(root_us30k_path, dataset_paths[dataset_key])

    assert os.path.exists(dataset_path), "dataset path does not exist"

    if (dataset_key != "BUS-BRA") and (dataset_key != "OvarianCyst"):
        scans_path = os.path.join(dataset_path, "img")
        masks_path = os.path.join(dataset_path, "label")
        assert os.path.exists(scans_path), "scans_path does not exist"
        assert os.path.exists(masks_path), "masks_path does not exist"
        assert os.path.exists(
            os.path.join(dataset_path, f"MainPatient/train.txt")
        ), f"train.txt file does not exist"
        assert os.path.exists(
            os.path.join(dataset_path, f"MainPatient/val.txt")
        ), f"val.txt file does not exist"
        assert os.path.exists(
            os.path.join(dataset_path, f"MainPatient/test.txt")
        ), f"test.txt file does not exist"

        df_train = get_df(
            os.path.join(dataset_path, f"MainPatient/train.txt"), scans_path, masks_path
        )
        df_val = get_df(
            os.path.join(dataset_path, f"MainPatient/val.txt"), scans_path, masks_path
        )
        df_test = get_df(
            os.path.join(dataset_path, f"MainPatient/test.txt"), scans_path, masks_path
        )
        df_train["label"] = dataset_key
        df_val["label"] = dataset_key
        df_test["label"] = dataset_key
    elif dataset_key == "BUS-BRA":
        # not mature source code for BUS-BRA  # will handle it later
        scans_path = os.path.join(dataset_path, "Images")
        masks_path = os.path.join(dataset_path, "Masks")
        df_folds = pd.read_csv(os.path.join(dataset_path, f"5-fold-cv.csv"))
        df = df_folds[df_folds["kFold"] == 1]  # just for testing
        df["id"] = df["ID"].apply(lambda x: x.split("_")[1].replace(".png", ""))

        df["image_path"] = df["id"].apply(
            lambda x: os.path.join(scans_path, f"bus_{x}") + ".png"
        )  # Strong Assumption that the images are png files
        df["mask_path"] = df["id"].apply(
            lambda x: os.path.join(masks_path, f"mask_{x}") + ".png"
        )  # Strong Assumption that the masks are png files
        df_train = df  # replace it with the df_folds
        df_val = df
        df_test = df
        df_train["label"] = dataset_key
        df_val["label"] = dataset_key
        df_test["label"] = dataset_key
    elif dataset_key == "OvarianCyst":
        scans_path = os.path.join(dataset_path, "images")
        masks_path = os.path.join(dataset_path, "annotations")
        df_train = pd.read_csv(
            os.path.join(dataset_path, f"train.txt"), header=None, names=["id"]
        )
        df_val = pd.read_csv(
            os.path.join(dataset_path, f"val.txt"), header=None, names=["id"]
        )
        #
        df_train["image_path"] = df_train["id"].apply(
            lambda x: os.path.join(scans_path, f"{x}.JPG")
        )
        df_train["mask_path"] = df_train["id"].apply(
            lambda x: os.path.join(masks_path, f"{x}_binary_binary.PNG")
        )
        df_val["image_path"] = df_val["id"].apply(
            lambda x: os.path.join(scans_path, f"{x}.JPG")
        )
        df_val["mask_path"] = df_val["id"].apply(
            lambda x: os.path.join(masks_path, f"{x}_binary_binary.PNG")
        )
        df_train["label"] = dataset_key
        df_val["label"] = dataset_key
        # df_test["label"] = dataset_key
        df_test = df_val
    else:
        raise ValueError(f"Dataset {dataset_key} not supported")
    assert df_train is not None, "train dataframe does not exist"
    assert df_val is not None, "val dataframe does not exist"
    assert df_test is not None, "test dataframe does not exist"
    df_train["label"] = dataset_key
    assert os.path.exists(
        df_train["image_path"].values[0]
    ), f"image path does not exist {df['image_path'].values[0]}"
    assert os.path.exists(
        df_train["mask_path"].values[0]
    ), f"mask path does not exist {df_train['mask_path'].values[0]}"
    train_data = df_train[["image_path", "mask_path", "label", "id"]].values
    val_data = df_val[["image_path", "mask_path", "label", "id"]].values
    test_data = df_test[["image_path", "mask_path", "label", "id"]].values

    train_dataset = get_formatted_dataset(train_data, H=H, W=W, shift_bbox=shift_bbox)
    val_dataset = get_formatted_dataset(val_data, H=H, W=W, shift_bbox=shift_bbox)
    test_dataset = get_formatted_dataset(test_data, H=H, W=W, shift_bbox=shift_bbox)

    if visualize:
        plt.figure(figsize=(10, 10))
        sample = next(iter(train_dataset))
        img = sample["pixel_values"].numpy().transpose(1, 2, 0)
        mask = sample["labels"].numpy()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.colorbar(label="Intensity", location="right", shrink=0.5)
        plt.title("Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title("Mask")
        plt.axis("off")
        plt.colorbar(label="Intensity", location="right", shrink=0.5)
        plt.tight_layout()
        plt.suptitle(f"ID: {sample['metadata']['id'][0]}", fontsize=16)
        plt.show()

    return df_train, df_val, df_test, train_dataset, val_dataset, test_dataset


def calculate_metrics_ind(outputs, targets, threshold=0.5):
    """
    Calculates evaluation metrics for binary segmentation per image and averages over the batch.
    Args:
        outputs (torch.Tensor): Model predictions, expected to be probabilities in [0,1], shape [batch_size, 1, H, W]
        targets (torch.Tensor): Ground truth labels, shape [batch_size, 1, H, W]
    Returns:
        metrics_dict (dict): Dictionary containing averaged 'dice', 'iou', 'precision', 'recall', 'accuracy'
    """
    epsilon = 1e-7  # Small constant to avoid division by zero
    # Threshold outputs to obtain binary predictions
    # Downsample the outputs and targets to 256x256
    outputs_bin = (outputs > threshold).float()
    targets_bin = (targets > threshold).float()

    batch_size = outputs.size(0)
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    hd95_scores = []
    # new two metrics for roc curve
    fpr = []
    tpr = []
    # make the loop to be tqdm
    for i in tqdm(range(batch_size)):
        output_i = outputs_bin[i].view(-1)
        target_i = targets_bin[i].view(-1)

        # True Positives, False Positives, True Negatives, False Negatives
        TP = (output_i * target_i).sum()
        FP = (output_i * (1 - target_i)).sum()
        TN = ((1 - output_i) * (1 - target_i)).sum()
        FN = ((1 - output_i) * target_i).sum()

        # Dice Coefficient
        dice = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
        dice_scores.append(dice.item())

        # Intersection over Union (IoU)
        iou = (TP + epsilon) / (TP + FP + FN + epsilon)
        iou_scores.append(iou.item())

        # Precision
        precision = (TP + epsilon) / (TP + FP + epsilon)
        precision_scores.append(precision.item())

        # Recall
        recall = (TP + epsilon) / (TP + FN + epsilon)
        recall_scores.append(recall.item())

        # Accuracy
        accuracy = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)
        accuracy_scores.append(accuracy.item())

        # Compute HD95
        hd95 = compute_hd95(outputs_bin[i].squeeze(0), targets_bin[i].squeeze(0))
        # hd95 = 0
        hd95_scores.append(hd95)

        # Compute FPR and TPR
        # Detach the FP, TN, TP, FN
        FP = FP.detach().cpu().numpy()
        TN = TN.detach().cpu().numpy()
        TP = TP.detach().cpu().numpy()
        FN = FN.detach().cpu().numpy()
        fpr.append(FP / (FP + TN))
        tpr.append(TP / (TP + FN))
        # print the fpr and tpr
        # print(f"FPR: {fpr[-1]}, TPR: {tpr[-1]}")

    # Average over batch
    metrics_dict = {
        "dice": dice_scores,
        "iou": iou_scores,
        "precision": precision_scores,
        "recall": recall_scores,
        "accuracy": accuracy_scores,
        "hd95": hd95_scores,
        "fpr": fpr,
        "tpr": tpr,
    }

    return metrics_dict


def visualize_batch_overlay(
    batch,
    mask,
    num_imgs=1,
    preview=False,
    MODEL_NAME="UNet",
    threshold=0.5,
    save=False,
    save_dir=None,
    batch_metrics: dict = None,
    contour_thickness: int = 2,
    enforce_unique_filename: bool = False,
):
    for idx, (image, gt_mask, pr_mask, x_path, x_label, x_dice, x_hd) in enumerate(
        zip(
            batch["pixel_values"],
            batch["labels"],
            mask,
            batch["metadata"]["image_path"],
            batch["metadata"]["label"],
            batch_metrics["dice"],
            batch_metrics["hd95"],
        )
    ):
        if idx > num_imgs:
            break

        # Convert image to NumPy and normalize for visualization
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8 (0-255)

        gt_mask_np = gt_mask.cpu().numpy().squeeze().astype(np.uint8)  # (H, W)
        pr_mask_np = pr_mask.cpu().numpy().squeeze().astype(np.uint8)  # (H, W)

        # Define Colors for Masks
        gt_color = (0, 255, 0)  # Green for Ground Truth
        pr_color = (255, 0, 0)  # Red for Prediction

        # Step 1: Create Overlay
        overlay = image_np.copy()

        # Fill Ground Truth Mask
        gt_overlay = np.zeros_like(image_np, dtype=np.uint8)
        gt_overlay[gt_mask_np == 1] = gt_color  # Fill ground truth in green

        # Fill Prediction Mask
        pr_overlay = np.zeros_like(image_np, dtype=np.uint8)
        pr_overlay[pr_mask_np == 1] = pr_color  # Fill prediction in red

        # Blend with original image using alpha transparency
        alpha = 0.4  # Transparency level
        overlay = cv2.addWeighted(overlay, 1, gt_overlay, alpha, 0)  # Overlay GT
        overlay = cv2.addWeighted(
            overlay, 1, pr_overlay, alpha, 0
        )  # Overlay Prediction

        # Step 2: Draw Contours (Same Colors as Overlays)
        contours_gt, _ = cv2.findContours(
            gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_pr, _ = cv2.findContours(
            pr_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay, contours_gt, -1, gt_color, thickness=contour_thickness
        )  # GT Contour
        cv2.drawContours(
            overlay, contours_pr, -1, pr_color, thickness=contour_thickness
        )  # Prediction Contour

        # Step 3: Plot the Figure
        plt.figure(figsize=(10, 10), dpi=200)
        plt.imshow(overlay)
        plt.axis("off")

        # Add Legend
        legend_patches = [
            Patch(color=np.array(gt_color) / 255, label="Ground Truth"),
            Patch(color=np.array(pr_color) / 255, label="Prediction"),
        ]
        plt.legend(handles=legend_patches, loc="upper right", fontsize=10)

        # Add title with model name and metrics
        plt.title(
            f"Model: {MODEL_NAME}\nDice: {x_dice:.2f}, HD95: {x_hd:.2f}", fontsize=12
        )

        # Save or display the figure
        # print(x_path)
        if save:
            os.makedirs(save_dir, exist_ok=True)
            if enforce_unique_filename:
                sample_idx = str(x_path[0]).split("\\")[-1].split(".")[0]
            else:
                sample_idx = x_path.split("/")[-1].split(".")[0]

            plt.savefig(f"{save_dir}/sample_idx_{sample_idx}.png", bbox_inches="tight")

        if preview:
            plt.show()
        else:
            plt.close()

# def _

def get_inference(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    post_processing_fns: dict,
    selected_model: str,
    SAVE_DIR: str,
    SAVE_FIGURES: bool,
    PLOTS_PATHS: str,
    benchmark_dataset: str,
) -> pd.DataFrame:
    """
    This function gets the inference for the test loader, and saves the predictions in the SAVE_DIR

    Args:
        test_loader (torch.utils.data.DataLoader): test loader
        model (torch.nn.Module): model
        device (torch.device): device
        post_processing_fns (dict): post processing functions
        selected_model (str): selected model
        SAVE_DIR (str): save directory
        SAVE_FIGURES (bool): save figures

        TODO : Check if sigmoid is already applied in the model since we apply this function for various models

    """
    model.eval()
    list_of_test_metrics_per_batch = []
    for batch in tqdm(test_loader):
        metadata = batch["metadata"]
        with torch.no_grad():
            image = batch["pixel_values"].to(device)  # Shape: (B, 3, H, W)
            logits = model(image)  # Output shape: (B, 2, H, W)
        # Process predictions
        pr_masks = logits.sigmoid()
        mask = (pr_masks > post_processing_fns["threshold"]).to(
            torch.uint8
        )  # Apply thresholding

        # Apply multi-lesion filtering if enabled
        if post_processing_fns["REMOVE_SMALL_LESIONS"]:
            mask_np = mask.cpu().numpy()
            mask_np = np.array(
                [filter_multi_lesions(m.squeeze()) for m in mask_np]
            )  # Efficiently apply filtering
            mask = torch.tensor(mask_np).unsqueeze(1).to(device)

        batch_metrics = calculate_metrics_ind(
            outputs=mask.to(device),
            targets=batch["labels"].to(device),
            threshold=post_processing_fns["threshold"],
        )
        visualize_batch_overlay(
            batch,
            mask,
            num_imgs=image.shape[0],
            preview=False,
            MODEL_NAME=selected_model,
            threshold=post_processing_fns["threshold"],
            save=SAVE_FIGURES,
            save_dir=SAVE_DIR,
            batch_metrics=batch_metrics,
        )

        image_paths = metadata["image_path"]
        mask_paths = list(batch["metadata"]["mask_path"])
        tumor_class = metadata["label"]
        batch_metrics["image_path"] = image_paths
        batch_metrics["mask_path"] = mask_paths
        batch_metrics["tumor_class"] = tumor_class
        pd.set_option("display.float_format", "{:.4f}".format)
        df = pd.DataFrame(batch_metrics)
        list_of_test_metrics_per_batch.append(df)

        # break

    # Save results
    df_test = pd.concat(list_of_test_metrics_per_batch, ignore_index=True)
    # df_test.to_csv(f"{PLOTS_PATHS}/{selected_model}_inference_metrics_{benchmark_dataset}.csv", index=False)
    print("Inference done!")
    # return df
    return df_test


class model_config:
    def __init__(self):
        self.model_name = None
        self.input_channels = 3
        self.num_classes = 1
        self.image_height = None
        self.image_width = None
        self.shift_bbox = 10
        self.device = None


class segformer_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "segformer"
        self.image_height = 256
        self.image_width = 256


class unet_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "unet"
        self.image_height = 256
        self.image_width = 256

class unet_pp_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "unet_pp"
        self.image_height = 256
        self.image_width = 256

class swinunet_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "swinunet"
        self.image_height = 224
        self.image_width = 224


class transunet_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "transunet"
        self.image_height = 256
        self.image_width = 256


def get_model(args):
    if args.model_name == "unet":
        print(f"Using UNet model")
        model = UNet(
            n_channels=args.input_channels, n_classes=args.num_classes, bilinear=True
        ).to(args.device)
    elif args.model_name == "segformer":
        print(f"Using SegFormer model")
        model = SegFormer(
            num_labels=args.num_classes,
            checkpoint="nvidia/mit-b5",
        )
        model.model.to(args.device)
    elif args.model_name == "swinunet":
        print(f"Using SwinUnet model")
        swin_args = swin_parser()
        swin_config = get_config(swin_args)
        model = SwinUnet(
            config=swin_config, img_size=args.image_height, num_classes=args.num_classes
        ).to(args.device)
    elif args.model_name == "transunet":
        print(f"Using TransUNet model")
        model = TransUNet(
            img_dim=args.image_height,
            in_channels=args.input_channels,
            class_num=args.num_classes,
        ).to(args.device)
    elif args.model_name == "unet_pp":
        print(f"Using UNetPlusPlus model")
        # model = UNet(n_channels=args.input_channels, n_classes=args.num_classes, bilinear=True)
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=args.input_channels,
            classes=args.num_classes,
            activation = "sigmoid" 
        )
        model = model.to(args.device)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    return model


def main(args):
    _models_configs = {
        "segformer": segformer_config(),
        "unet": unet_config(),
        "swinunet": swinunet_config(),
        "transunet": transunet_config(),
        "unet_pp":unet_pp_config(),
    }
    ROOT_US30K_PATH = args.root_us30k_path
    RESULTS_DIR = args.results_dir
    BENCHMARK_DATASET = args.benchmark_dataset
    SELECTED_MODEL = args.model_name
    PLOTS_PATHS = os.path.join(
        RESULTS_DIR, BENCHMARK_DATASET, SELECTED_MODEL + "_" + args.test_dataset
    )
    POST_PROCESSING_FN = {
        "REMOVE_SMALL_LESIONS": args.remove_small_lesions,
        "threshold": args.threshold,
    }
    CHECKPOINT_PATH = f"/home/user/data/physformer_data/checkpoints_physformer/{SELECTED_MODEL}/physformer_{SELECTED_MODEL}_clinical_{BENCHMARK_DATASET}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_PATHS, exist_ok=True)

    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    model_config = _models_configs[args.model_name]
    model_config.device = device

    # Get the test loader
    _, _, _, _, _, test_dataset = get_dataset(
        dataset_key=args.test_dataset,
        visualize=False,
        root_us30k_path=ROOT_US30K_PATH,
        H=model_config.image_height,
        W=model_config.image_width,
        shift_bbox=model_config.shift_bbox,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Get the model
    model = get_model(model_config)
    # Load the model checkpoint
    load_path = os.path.join(CHECKPOINT_PATH, "best_model.pth")
    model.load_state_dict(torch.load(load_path)["model_state_dict"])
    print(f"Model loaded from {load_path} Successfully!")
    inference_config = {
        "SAVE_FIGURES": True,
        "ROOT_PATH": "",
        "MODEL_NAME": SELECTED_MODEL,
        "SAVE_DIR": PLOTS_PATHS,
        "device": device,
        "model": model,
        "POST_PROCESSING_FN": POST_PROCESSING_FN,
        "benchmark_dataset": BENCHMARK_DATASET,
        "selected_model": SELECTED_MODEL,
        "PLOTS_PATHS": PLOTS_PATHS,
    }
    # get the inference
    df_results = get_inference(
        test_loader=test_loader,
        model=inference_config["model"],
        device=inference_config["device"],
        post_processing_fns=inference_config["POST_PROCESSING_FN"],
        SAVE_DIR=inference_config["SAVE_DIR"],
        SAVE_FIGURES=inference_config["SAVE_FIGURES"],
        PLOTS_PATHS=inference_config["PLOTS_PATHS"],
        benchmark_dataset=inference_config["benchmark_dataset"],
        selected_model=inference_config["selected_model"],
    )
    # CSV Directory
    csv_dir_path = os.path.join(RESULTS_DIR, BENCHMARK_DATASET)
    os.makedirs(csv_dir_path, exist_ok=True)
    df_results.to_csv(
        f"{csv_dir_path}/{SELECTED_MODEL}_trained_on_{BENCHMARK_DATASET}_tested_on_{args.test_dataset}_metrics.csv",
        index=False,
    )

    return df_results


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--checkpoint_path", type=str, default=None)
    args.add_argument("--remove_small_lesions", type=bool, default=False)
    args.add_argument("--threshold", type=float, default=0.5)
    args.add_argument(
        "--root_us30k_path",
        type=str,
        default="/home/user/data/phyusformer_data/post_miccai_exps/data/US30k/US30K/",
    )
    args.add_argument(
        "--results_dir",
        type=str,
        default="/home/user/data/physformer_data/cross_domain_generalization",
    )
    args.add_argument("--benchmark_dataset", type=str, default="")
    args.add_argument("--plots_paths", type=str, default=None)
    args.add_argument("--device_id", type=int, default=0)
    args.add_argument(
        "--task", type=str, default="clinical", choices=["clinical", "physics"]
    )
    args = args.parse_args()
    RESULTS_DIR = args.results_dir
    # for model_name in ["swinunet", "transunet","segformer"]:
    for model_name in ["unet_pp"]:
        for benchmark_dataset in ["BUSI", "UDIAT", "TN3K", "DDTI", "OvarianCyst"]:
            args.model_name = model_name
            args.benchmark_dataset = benchmark_dataset
            list_of_df_results = []
            for test_dataset in ["BUSI", "UDIAT", "TN3K", "DDTI", "OvarianCyst"]:
                print(
                    f"Testing {model_name} Trained on {benchmark_dataset} Tested on {test_dataset}"
                )
                # args.selected_model = model_name
                args.test_dataset = test_dataset
                df_results = main(args)
                df_results["test_dataset"] = test_dataset
                list_of_df_results.append(df_results)
                gc.collect()
                torch.cuda.empty_cache()
                
            df_results_concat = pd.concat(list_of_df_results, ignore_index=True)
            df_results_concat.to_csv(
                f"{RESULTS_DIR}/{model_name}_model_{benchmark_dataset}_tested_on_all_datasets.csv",
                index=False,
            )
            gc.collect()
