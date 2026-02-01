import os
import torch
import segmentation_models_pytorch as smp

try:
    from .unet import UNet
    from .segformer import SegFormer
    from .swinunet import SwinUnet
    from .transunet import TransUNet
    from .swinunet_config import get_config
    from .swinunet import parser as swin_parser
except ImportError:
    from unet import UNet
    from segformer import SegFormer
    from swinunet import SwinUnet
    from transunet import TransUNet
    from swinunet_config import get_config
    from swinunet import parser as swin_parser


def get_model(configs):
    if configs.model_name == "unet":
        print(f"Using UNet model")
        model = UNet(
            n_channels=configs.input_channels, n_classes=configs.num_classes, bilinear=True
        )
        # model = smp.Unet(
        #     encoder_name="efficientnet-b5",
        #     encoder_weights="imagenet",
        #     in_channels=args.input_channels,
        #     classes=args.num_classes,
        #     activation = "sigmoid"
        # )
    #         model = UNet(
    #             n_channels=args.input_channels, n_classes=args.num_classes, bilinear=True
    #         ).to(args.device)
    elif configs.model_name == "unet_pp":
        print(f"Using UNetPlusPlus model")
        # model = UNet(n_channels=args.input_channels, n_classes=args.num_classes, bilinear=True)
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=configs.input_channels,
            classes=configs.num_classes,
            activation="sigmoid",
        )
    elif configs.model_name == "segformer":
        print(f"Using SegFormer model")
        model = SegFormer(
            num_labels=configs.num_classes,
            checkpoint=configs.loaded_checkpoint_path,
            config = configs.loaded_model_config
        )
        model.to(configs.device)
    elif configs.model_name == "swinunet":
        print(f"Using SwinUnet model")
        swin_args = swin_parser()
        swin_config = get_config(swin_args)
        model = SwinUnet(
            config=swin_config, img_size=configs.image_height, num_classes=configs.num_classes
        )
    elif configs.model_name == "transunet":
        print(f"Using TransUNet model")
        model = TransUNet(
            img_dim=configs.image_height,
            in_channels=configs.input_channels,
            class_num=configs.num_classes,
        )
    else:
        raise ValueError(f"Invalid model name: {configs.model_name}")
    return model


if __name__ == "__main__":

    class config:
        def __init__(self):
            self.model_name = "segformer"
            self.input_channels = 3
            self.num_classes = 1
            self.image_height = 256
            self.image_width = 256
            self.device = torch.device("cuda:0")
            self.checkpoint = "/home/user/data/physformer_data/checkpoints_physformer/segformer/physformer_segformer_v1"

    args = config()
    model = get_model(args)
    # Load the model checkpoint
    load_path = os.path.join(args.checkpoint, "best_model.pth")
    print(f"Loading model from {load_path}")
    model.load_state_dict(torch.load(load_path)["model_state_dict"])
    print(f"Model loaded successfully")
    model.to(args.device)
    # print(model)
    x = torch.randn(1, 3, 256, 256)
    x = x.to(args.device)
    y = model(x)[0]
    print(y.shape)
