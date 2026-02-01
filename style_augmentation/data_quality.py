from ntpath import sameopenfile
from typing import Any


import os
from re import S
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# Format y-axis ticks to 2 decimal places
from matplotlib.ticker import FormatStrFormatter

# Lets build the config for the paths of all types of synthetic images we have
config = {
    "paths": {
        "physics": "/home/user/data/phyusformer_data/style_transfer/inference_data_latest/synt2realUS_large_multiple_sources/test_latest/images/",
        "pix2pix": "/home/user/haris/pytorch-CycleGAN-and-pix2pix/results/pix2pix_source2target_highres_large_data_inference/pix2pix_source2target_highres/test_latest/images/",
        "cyclegan_without_apsa": "/home/user/data/phyusformer_data/style_transfer/inference_data_latest/synt2realUS_large_multiple_sources/test_latest/images/",
        "cyclegan_with_apsa": "/home/user/data/phyusformer_data/style_transfer/inference_data_roi_aware_model/synt2realUS_large_multiple_sources_roi_aware_v3/test_latest/images/",
    },
    "filter_images_suffix": {
        "physics": "scan_real.png",
        "pix2pix": "_fake_B.png",
        "cyclegan_without_apsa": "scan_fake.png",
        "cyclegan_with_apsa": "scan_fake.png",
    },
    "colors": {
        "real": "#808080",
        "physics": "#F0E68C",
        "pix2pix": "#7BC8F6",
        "cyclegan_without_apsa": "#808000",
        "cyclegan_with_apsa": "#650021",
    },
    # colors = {
    # "Real–Real Split": "#C0C0C0",
    # "Physics-Based": "#F0E68C",
    # "Physics + Pix2Pix": "#7BC8F6",
    # "Physics + CycleGAN": "#808000",
    # "Physics + APSA": "#650021",
    # }
}


class StyleTransferDataset(Dataset):
    def __init__(
        self, image_paths, normalize=True, remove_top_region=False, top_region_height=50
    ):
        self.image_paths = image_paths
        self.images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
        self.normalize = normalize
        self.remove_top_region = remove_top_region
        self.top_region_height = top_region_height

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.images[index]
        paths = self.image_paths[index]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        if self.normalize:
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        else:
            # image =
            # image = image * 255.0
            pass
        # remove the top region of the image if remove_top_region is True
        if self.remove_top_region:
            image = image[self.top_region_height :, :]
        image = torch.from_numpy(image).unsqueeze(0).float()
        return {"images": image, "paths": paths}


def main_fid(args):
    os.makedirs(args.results_path, exist_ok=True)
    device = torch.device(f"cuda:{args.device_id}")

    print("Using device:", device)

    root_real_1 = "/home/user/data/phyusformer_data/style_transfer/synt2realUS_cyclegan_roi_aware_v3/trainB"
    root_real_2 = "/home/user/data/phyusformer_data/style_transfer/synt2realUS_cyclegan_roi_aware_v3/testB"

    real_paths = [
        os.path.join(root_real_1, x) for x in os.listdir(root_real_1) if not "mask" in x
    ]
    real_paths += [
        os.path.join(root_real_2, x) for x in os.listdir(root_real_2) if not "mask" in x
    ]

    if args.experiment_type == "baseline":
        # divide the real data into two groups randomly
        rand_ids = np.random.choice(
            len(real_paths), size=int(len(real_paths) / 2), replace=False
        )
        real_paths_1 = [real_paths[i] for i in rand_ids]
        real_paths_2 = [
            real_paths[i] for i in np.setdiff1d(np.arange(len(real_paths)), rand_ids)
        ]
        real_paths = real_paths_1
        synthetic_paths = real_paths_2

    elif args.experiment_type == "compare":
        synthetic_paths = [
            os.path.join(args.paths, x)
            for x in os.listdir(args.paths)
            if x.endswith(args.filter_images_suffix)
        ]
    else:
        # TODO : raise the exception or handle this conditonal code
        pass

    print(f"Total real images: {len(real_paths)}")
    print(f"Total synthetic images: {len(synthetic_paths)}")

    real_dataset = StyleTransferDataset(real_paths)
    synthetic_dataset = StyleTransferDataset(synthetic_paths)

    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
    synthetic_loader = DataLoader(
        synthetic_dataset, batch_size=args.batch_size, shuffle=False
    )

    fid = FrechetInceptionDistance(
        feature=2048, input_img_size=(256, 256), normalize=True
    ).to(device)

    for batch in tqdm(real_loader, desc="Feeding real images"):
        fid.update(batch["images"].to(torch.uint8).to(device), real=True)

    for batch in tqdm(synthetic_loader, desc="Feeding synthetic images"):
        fid.update(batch["images"].to(torch.uint8).to(device), real=False)

    score = fid.compute()
    print(f" FID Score: {score.item():.2f}")


def main_ks_test(args):
    print("Running KS test")
    print("Args: ", args.__dict__)
    os.makedirs(args.results_path, exist_ok=True)
    args.device = torch.device(f"cuda:{args.device_id}")

    print("Using device:", args.device)

    root_real_1 = "/home/user/data/phyusformer_data/style_transfer/synt2realUS_cyclegan_roi_aware_v3/trainB"
    root_real_2 = "/home/user/data/phyusformer_data/style_transfer/synt2realUS_cyclegan_roi_aware_v3/testB"

    real_paths = [
        os.path.join(root_real_1, x) for x in os.listdir(root_real_1) if not "mask" in x
    ]
    real_paths += [
        os.path.join(root_real_2, x) for x in os.listdir(root_real_2) if not "mask" in x
    ]

    # print(f"Total real images: {len(real_paths)}")

    if args.experiment_type == "baseline":
        # divide the real data into two groups randomly
        rand_ids = np.random.choice(
            len(real_paths), size=int(len(real_paths) / 2), replace=False
        )
        real_paths_1 = [real_paths[i] for i in rand_ids]
        real_paths_2 = [
            real_paths[i] for i in np.setdiff1d(np.arange(len(real_paths)), rand_ids)
        ]
        real_paths = real_paths_1
        synthetic_paths = real_paths_2

    elif args.experiment_type == "compare":
        args.paths = config["paths"][args.synthetic_type]
        args.filter_images_suffix = config["filter_images_suffix"][args.synthetic_type]

        synthetic_paths = [
            os.path.join(args.paths, x)
            for x in os.listdir(args.paths)
            if x.endswith(args.filter_images_suffix)
        ]

    else:
        # TODO : raise the exception or handle this conditonal code
        pass

    print(f"Total real images: {len(real_paths)}")
    print(f"Total synthetic images: {len(synthetic_paths)}")

    # print(f"Normalize: {args.normalize}")
    print("real paths: ", real_paths[:5])
    print("synthetic paths: ", synthetic_paths[:5])
    # total_length =
    # args.sample_size = 100
    # sample_ids = np.random.choice(len(real_paths), size=args.sample_size, replace=False)
    # sampled_real_paths = [real_paths[i] for i in sample_ids]
    # sampled_synthetic_paths = [synthetic_paths[i] for i in sample_ids]
    sampled_real_paths = real_paths[: args.sample_size]
    sampled_synthetic_paths = synthetic_paths[1000 : 1000 + args.sample_size]
    if args.synthetic_type == "physics":
        print("Removing the top region of the synthetic images")
        args.remove_top_region = True
        args.top_region_height = 50
    real_dataset = StyleTransferDataset(sampled_real_paths,
        normalize=args.normalize,
        remove_top_region=args.remove_top_region,
    )
    synthetic_dataset = StyleTransferDataset(
        sampled_synthetic_paths,
        normalize=args.normalize,
        remove_top_region=args.remove_top_region,
    )

    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
    synthetic_loader = DataLoader(
        synthetic_dataset, batch_size=args.batch_size, shuffle=False
    )
    print(
        "min and max of the real pixel distribution: ",
        next(iter(real_loader))["images"].min(),
        next(iter(real_loader))["images"].max(),
    )
    print(
        "min and max of the synthetic pixel distribution: ",
        next(iter(synthetic_loader))["images"].min(),
        next(iter(synthetic_loader))["images"].max(),
    )

    print(
        "unique value of synthetic pixel distribution: ",
        np.unique(next(iter(synthetic_loader))["images"]),
    )
    real_pixel_distribution = (
        get_pixel_distribution(real_loader).view(-1).detach().cpu().numpy()
    )
    synthetic_pixel_distribution = (
        get_pixel_distribution(synthetic_loader).view(-1).detach().cpu().numpy()
    )
    # print("Real pixel distribution shape: ", real_pixel_distribution.shape)
    # print("Synthetic pixel distribution shape: ", synthetic_pixel_distribution.shape)
    # draw_pixel_distribution(real_pixel_distribution, synthetic_pixel_distribution)
    # sanity_check(real_loader, save_text="real_sanity_check.png")
    # sanity_check(synthetic_loader, save_text="synthetic_sanity_check.png")
    # lets apply the ks test
    # ks_score = ks_2samp(real_pixel_distribution, synthetic_pixel_distribution)
    # print("KS test score: ", ks_score.pvalue)
    # print("KS test statistic: ", ks_score.statistic)
    # print("KS test critical value: ", ks_score.critical_value)
    # Sample 20K pixels from each distribution
    # N = min(len(real_pixel_distribution), len(synthetic_pixel_distribution), 20000)
    # real_sample = np.random.choice(real_pixel_distribution, size=N, replace=False)
    # synthetic_sample = np.random.choice(synthetic_pixel_distribution, size=N, replace=False)

    perform_distribution_tests(
        real_pixel_distribution,
        synthetic_pixel_distribution,
        args.results_path,
        config_color=config["colors"][args.synthetic_type],
    )


def sanity_check(loader, save_text: str = "sanity_check.png"):
    sample = next(iter(loader))
    print("Sample shape: ", sample["images"].shape)
    # create a grid and plot the images in the grid
    grid_size = int(np.sqrt(sample["images"].shape[0]))
    print("Grid size: ", grid_size)
    plt.figure(figsize=(10, 5))
    for i in range(sample["images"].shape[0]):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(sample["images"][i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(sample["paths"][i], fontsize=4, loc="left")

    plt.tight_layout()
    plt.savefig(os.path.join(args.results_path, save_text))
    plt.show()


def draw_pixel_distribution(real_pixel_distribution, synthetic_pixel_distribution):
    plt.figure(figsize=(10, 5))
    plt.hist(real_pixel_distribution, bins=100, alpha=0.5, label="Real")
    plt.hist(synthetic_pixel_distribution, bins=100, alpha=0.5, label="Synthetic")
    plt.legend()
    plt.savefig(os.path.join(args.results_path, "pixel_distribution.png"))
    plt.show()


def get_pixel_distribution(loader):
    pixel_distribution = torch.tensor([]).to(args.device)
    for batch in tqdm(loader, desc="Getting pixel distribution"):
        images = batch["images"].to(args.device)
        images = images.view(images.size(0), -1).float()
        pixel_distribution = torch.cat([pixel_distribution, images], dim=0)
    print("Pixel distribution shape: ", pixel_distribution.shape)
    print(
        "modified view for the pixel distribution : ",
        pixel_distribution[:1].view(-1).shape,
    )
    return pixel_distribution


import seaborn as sns
from scipy.stats import wasserstein_distance, anderson_ksamp
from scipy.special import kl_div


def perform_distribution_tests(
    real_pixel_distribution,
    synthetic_pixel_distribution,
    save_path,
    config_color="#650021",
):
    print("Performing distribution tests")
    print("Real pixel distribution shape: ", real_pixel_distribution.shape)
    print("Synthetic pixel distribution shape: ", synthetic_pixel_distribution.shape)

    # Histogram-based KL Divergence (requires same bin edges).
    # We build histograms on a common range inferred from both distributions,
    # then use scipy.special.kl_div elementwise and sum over bins.
    data_min = min(real_pixel_distribution.min(), synthetic_pixel_distribution.min())
    data_max = max(real_pixel_distribution.max(), synthetic_pixel_distribution.max())
    hist_r, bins = np.histogram(
        real_pixel_distribution,
        bins=args.bin_size,
        range=(data_min, data_max),
        density=True,
    )
    hist_s, _ = np.histogram(
        synthetic_pixel_distribution,
        bins=args.bin_size,
        range=(data_min, data_max),
        density=True,
    )
    # KL(real || synthetic) using scipy.special.kl_div, see:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html
    kl_per_bin = kl_div(hist_r + 1e-8, hist_s + 1e-8)
    kl_value = np.sum(kl_per_bin)

    # Compute KS, Wasserstein, Anderson-Darling
    ks_result = ks_2samp(real_pixel_distribution, synthetic_pixel_distribution)
    wasserstein = wasserstein_distance(
        real_pixel_distribution, synthetic_pixel_distribution
    )
    anderson_result = anderson_ksamp(
        [real_pixel_distribution, synthetic_pixel_distribution]
    )

    # Print results
    print("=== Distribution Similarity Tests ===")
    print(f"KL Divergence         : {kl_value:.4f}")
    print(f"Wasserstein Distance  : {wasserstein:.4f}")
    print(
        f"KS Statistic          : {ks_result.statistic:.4f} (p-value: {ks_result.pvalue:.10e})"
    )
    print(
        f"Anderson-Darling Stat : {anderson_result.statistic:.4f} (p < {anderson_result.significance_level:.4f})"
    )

    # Plot histogram and KDE
    # font of the text in the plot sgherif and sans-serif are not working so we use Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"axes.labelsize": 14})
    plt.rcParams.update({"axes.titlesize": 14})
    plt.rcParams.update({"legend.fontsize": 14})
    plt.rcParams.update({"xtick.labelsize": 14})
    plt.rcParams.update({"ytick.labelsize": 14})
    plt.rcParams.update({"figure.titlesize": 14})
    plt.rcParams.update({"figure.figsize": (3.5, 2.5)})

    plt.figure()
    ax = sns.kdeplot(
        real_pixel_distribution,
        color="#808080",
        label="Real",
        alpha=0.7,
        linewidth=2.0,
        shade=True,  # shaded area under the curve,
        # clip=(1, 255),
        bw_adjust=args.bw_adjust,
    )
    sns.kdeplot(
        synthetic_pixel_distribution,
        color=config_color,
        label="Synthetic",
        alpha=0.4,
        linewidth=2.0,
        shade=True,  # shaded area under the curve
        ax=ax,
        # clip=(1, 255),
        bw_adjust=args.bw_adjust,
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    # Axis labels and legend (IEEE-like styling)
    # ax.set_xlabel("Pixel Intensity", )
    # ax.set_ylabel("Density",)
    # Add text box with statistical test results near the legend (top-right)
    # Place clearly below the legend to avoid overlap, right-aligned
    # ax.text(
    #     0.98,
    #     0.80,
    #     f"KS p-value: {ks_result.pvalue:.4e}",
    #     fontsize=16,
    #     transform=ax.transAxes,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    # )
    # ax.text(
    #     0.98,
    #     0.74,
    #     f"Wasserstein Distance: {wasserstein:.4f}",
    #     fontsize=16,
    #     transform=ax.transAxes,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    # )
    # ax.text(
    #     0.98,
    #     0.68,
    #     f"Anderson-Darling Stat: {anderson_result.statistic:.4f}",
    #     fontsize=16,
    #     transform=ax.transAxes,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    # )
    ax.legend(loc="upper right")
    # lets limit the y-axis to 0.000 to 0.025
    ax.set_ylim(0.000, 0.022)
    ax.set_yticks([0.010,0.020])
    ax.set_xticks([0,255])
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # Minimal whitespace around figure
    plt.tight_layout()
    # plt.x_axis()
    plt.savefig(
        os.path.join(save_path, f"kde_distribution_{args.synthetic_type}.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return {
        "kl_divergence": kl_value,
        "wasserstein_distance": wasserstein,
        "ks_statistic": ks_result.statistic,
        "ks_pvalue": ks_result.pvalue,
        "anderson_statistic": anderson_result.statistic,
        "anderson_significance_level": anderson_result.significance_level,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--paths",
        type=str,
        default="/home/user/data/phyusformer_data/style_transfer/inference_data_roi_aware_model/synt2realUS_large_multiple_sources_roi_aware_v3/test_latest/images/",
        help="Path to the synthetic image directory",
    )
    parser.add_argument(
        "--filter_images_suffix",
        type=str,
        default="_fake.png",
        help="Filter images suffix",
    )
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--results_path", type=str, default="./results")
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="compare",
        choices=["compare", "baseline"],
    )
    parser.add_argument(
        "--command_help", action="store_true", help="Command line arguments help"
    )
    parser.add_argument("--mode", type=str, default="fid", choices=["fid", "ks"])
    parser.add_argument(
        "--normalize", type=bool, default=False, help="Normalize the pixel values"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Sample size for the distribution tests",
    )
    parser.add_argument(
        "--synthetic_type",
        type=str,
        default="physics",
        choices=["cyclegan_without_apsa", "cyclegan_with_apsa", "pix2pix", "physics"],
        help="Synthetic type",
    )
    parser.add_argument(
        "--bin_size", type=int, default=20, help="Bin size for the histogram"
    )
    parser.add_argument(
        "--bw_adjust",
        type=float,
        default=0.1,
        help="Bandwidth adjustment for the KDE plot",
    )
    parser.add_argument(
        "--remove_top_region",
        type=bool,
        default=False,
        help="Remove the top region of the synthetic images",
    )
    # if args.help:
    args = parser.parse_args()
    if args.command_help:
        print(
            "For the Baseline experiment, run the following command: python data_quality.py --device_id 1 --batch_size 1024 --experiment_type 'baseline'"
        )
        print("-------------------------------------")
        print(
            "For the comparison experiment for Real vs Pix2Pix, run the following command: python data_quality.py \n\
            --device_id 1 --batch_size 1024 --experiment_type 'compare' --paths '/home/user/data/phyusformer_data/style_transfer/inference_data_latest/synt2realUS_large_multiple_sources/test_latest/images/'"
        )
        print(
            "For the Compare experiment for Real vs CycleGAN, run the following command: python data_quality.py \n\
             --device_id 1 --batch_size 1024 --experiment_type 'compare' \n\
            --paths '/home/user/data/phyusformer_data/style_transfer/inference_data_roi_aware_model/synt2realUS_large_multiple_sources_roi_aware_v3/test_latest/images/'"
        )
        print(
            "for the comparison experiment of Real vs APSA, run the following command : python data_quality.py --device_id 1 --batch_size 512"
        )

        # quit and break the program
        exit()
    # # set_seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # # now we run the main function for 5 times with different seeds which randomly groups the 10000 synthetic images into 5 groups
    if args.mode == "fid":
        main_fid(args)
    elif args.mode == "ks":
        main_ks_test(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    # main_ks_test(args)
