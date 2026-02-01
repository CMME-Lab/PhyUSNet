"""
Pretraining the model on the physics dataset

"""

# Imports
import argparse
import config
import numpy as np
import wandb
import os
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
from pytorch_lightning.loggers import WandbLogger
from transformers import SegformerImageProcessor
# Internal imports
from utils.dataset import PhyUSDataset
from utils.model import get_model
from utils.loss import CombinedLoss
from utils.train import Trainer


# Main function to train the model on the physics dataset 

def main(args):
    # Few default params
    if args.experiment_name is None:    
        args.experiment_name = f"physformer_{args.model_name}"
    args.save_dir = f"/home/user/data/physformer_data/checkpoints_physformer/{args.model_name}/{args.experiment_name}"
    args.wandb_name = args.experiment_name

    try:
        args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Error initializing device: {e}")
        args.device = torch.device(f"cuda:{config.DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    try:
        wandb_writer = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config=args,
        )
    except Exception as e:
        print(f"Error initializing wandb writer: {e}")
        wandb_writer = None

    paths = np.load(args.dataset_path, allow_pickle=True)
    image_processor = SegformerImageProcessor(
        reduce_labels=False,
        do_normalize=False,
        do_rescale=False,
        size={"height": args.image_height, "width": args.image_width},
    )
    # trim_length = 1000
    # train_dataset = PhyUSDataset(paths=paths["train"][:trim_length], image_processor=image_processor)
    # val_dataset = PhyUSDataset(paths=paths["val"][:trim_length], image_processor=image_processor)
    # test_dataset = PhyUSDataset(paths=paths["test"][:trim_length], image_processor=image_processor)
    
    train_dataset = PhyUSDataset(paths=paths["train"], image_processor=image_processor)
    val_dataset = PhyUSDataset(paths=paths["val"], image_processor=image_processor)
    test_dataset = PhyUSDataset(paths=paths["test"], image_processor=image_processor)
    
    # Print the shapes of the datasets
    print("Training Data :", len(train_dataset))
    print("Validation Data :", len(val_dataset))
    print("Test Data :", len(test_dataset))
    print("Dataset loaded successfully")
    # Print the shapes of the datasets
    model = get_model(args).to(args.device)
    # get the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # get the loss function
    criterion = CombinedLoss(
        lambda_weight=args.lambda_weight,
        smooth=args.smooth,
        from_logits=args.from_logits,
    )
    # Log all the configs (support WandbLogger or raw wandb run)
    if wandb_writer:
        try:
            run = wandb_writer.experiment if hasattr(wandb_writer, "experiment") else wandb_writer
            cfg = vars(args) if hasattr(args, "__dict__") else args
            run.config.update(cfg)
            run.config.update(
                {
                    "lambda_weight": args.lambda_weight,
                    "smooth": args.smooth,
                    "from_logits": args.from_logits,
                    "device": str(args.device),
                    "save_dir": args.save_dir,
                }
            )
        except Exception:
            pass
    # get the trainer
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        writer=(wandb_writer.experiment if hasattr(wandb_writer, "experiment") else wandb_writer),
        device=args.device,
        save_dir=args.save_dir,
        # scheduler
    )

    # train the model
    if not args.test_only:
        trainer.fit()
    else:
        load_path = os.path.join(args.save_dir, "best_model.pth")
        if not os.path.exists(load_path):
            print(f"Checkpoint not found at {load_path}")
            return None
        model.load_state_dict(torch.load(load_path)["model_state_dict"])
        print(f"Model loaded from {load_path} Successfully!")
        trainer.test(model)
    wandb_writer.experiment.finish()
    return model


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, default=config.DATASET_PATH)
    args.add_argument("--model_name", type=str, default=config.MODEL_NAME)
    args.add_argument("--experiment_name", type=str, default=None)
    args.add_argument("--lambda_weight", type=float, default=config.LAMBDA_WEIGHT)
    args.add_argument("--smooth", type=float, default=config.SMOOTH)
    args.add_argument("--from_logits", type=bool, default=config.FROM_LOGITS)
    args.add_argument("--device_id", type=str, default=config.DEVICE_ID)
    args.add_argument("--save_dir", type=str, default='./checkpoints')
    args.add_argument("--wandb_project", type=str, default=config.WANDB_PROJECT)
    args.add_argument("--wandb_entity", type=str, default=config.WANDB_ENTITY)
    args.add_argument("--wandb_name", type=str, default=None)
    # args.add_argument("--wandb_mode", type=str, default=config.WANDB_MODE)
    args.add_argument("--wandb_mode", type=str, default="disabled")
    args.add_argument("--step_size", type=int, default=config.STEP_SIZE)
    args.add_argument("--gamma", type=float, default=config.GAMMA)
    args.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    args.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args.add_argument("--epochs", type=int, default=config.EPOCHS)
    args.add_argument("--image_height", type=int, default=config.IMAGE_HEIGHT)
    args.add_argument("--image_width", type=int, default=config.IMAGE_WIDTH)
    args.add_argument("--input_channels", type=int, default=config.INPUT_CHANNELS)
    args.add_argument("--num_classes", type=int, default=config.NUM_CLASSES)
    args.add_argument("--train_metrics_print_frequency", type=int, default=config.TRAIN_METRICS_PRINT_FREQUENCY)
    args.add_argument("--patience_epoch", type=int, default=config.PATIENCE_EPOCH)
    args.add_argument("--test_only", action="store_true", help="Run inference only")
    args.add_argument("--threshold", type=float, default=config.THRESHOLD, help="Threshold for binary segmentation")
    args = args.parse_args()
    model = main(args)
    # print(dataset[0])
