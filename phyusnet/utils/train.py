from ast import arg
import os
import time

import tqdm
import numpy as np
import torch
from utils.eval import get_metrics
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb
from torchvision.utils import make_grid
import pandas as pd

# random seed
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        writer,
        device,
        save_dir,
        args,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = epochs
        self.writer = writer
        self.device = device
        self.save_dir = save_dir
        self.args = args

    def _can_log(self):
        return self.writer is not None and hasattr(self.writer, "log")

    def _safe_log(self, data: dict):
        if self._can_log():
            try:
                self.writer.log(data)
            except Exception:
                pass

    def _log_table(self, key: str, columns: list, rows: list):
        if not self._can_log():
            return
        try:
            table = wandb.Table(
                columns=columns, data=[[str(c) for c in r] for r in rows]
            )
            self.writer.log({key: table})
        except Exception:
            pass

    def _log_run_metadata(self):
        # Hyperparameters/config
        try:
            config_items = []
            for k, v in vars(self.args).items():
                config_items.append([k, v, type(v).__name__])
            self._log_table(
                "hyperparameters", ["Parameter", "Value", "Type"], config_items
            )
        except Exception:
            pass

        # Dataset sizes
        try:
            train_len = (
                len(self.train_loader.dataset)
                if hasattr(self.train_loader, "dataset")
                else len(self.train_loader)
            )
            val_len = (
                len(self.val_loader.dataset)
                if hasattr(self.val_loader, "dataset")
                else len(self.val_loader)
            )
            rows = [["Training Set", train_len], ["Validation Set", val_len]]
            self._log_table("dataset_information", ["Dataset", "Size"], rows)
        except Exception:
            pass

    def _setup_wandb_metrics(self):
        if not self._can_log():
            return
        try:
            # Define step metrics for clean timelines
            wandb.define_metric("train/epoch")
            wandb.define_metric("train/*", step_metric="train/epoch")
            wandb.define_metric("val/epoch")
            wandb.define_metric("val/*", step_metric="val/epoch")
        except Exception:
            pass

    def fit(self):

        model = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        best_val_loss = float("inf")
        global_step = 0
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.args.step_size, gamma=self.args.gamma
        )
        # One-time metadata logging
        self._log_run_metadata()
        self._setup_wandb_metrics()
        count_patience_epochs = 0
        model.train()
        for epoch in range(self.num_epochs):
            if count_patience_epochs < self.args.patience_epoch:
                running_loss = 0.0
                running_dice_loss = 0.0
                running_ce_loss = 0.0
                start_time = time.time()
                list_of_dice_scores = []
                list_of_iou_scores = []
                list_of_precision_scores = []
                list_of_recall_scores = []
                list_of_accuracy_scores = []
                list_of_hd95_scores = []
                train_bar = tqdm(
                    total=len(self.train_loader),
                    desc=f"Train [{epoch + 1}/{self.num_epochs}]",
                    unit="batch",
                    leave=False,
                    dynamic_ncols=True,
                    mininterval=0.5,
                    maxinterval=1.0,
                    smoothing=0.1,
                    bar_format="{l_bar}{bar} | {elapsed}<{remaining} | {rate_fmt} | {postfix}",
                )
                pending_updates = 0
                update_every = int(
                    getattr(self.args, "train_metrics_print_frequency", 10) or 10
                )
                for batch_idx, (data) in enumerate(self.train_loader):
                    inputs = data["pixel_values"]
                    targets = data["labels"]
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # Forward pass
                    outputs = model(inputs)
                    loss, dice_loss, ce_loss = loss_fn(
                        outputs, targets.to(torch.float32)
                    )
                    # loss = torch.randn(1).to(self.device)
                    # dice_loss = torch.randn(1).to(self.device)
                    # ce_loss = torch.randn(1).to(self.device)
                    # Backward pass and optimization

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_dice_loss += dice_loss.item()
                    running_ce_loss += ce_loss.item()

                    # Log training loss
                    if self.args.from_logits:
                        outputs = torch.sigmoid(outputs)
                    else:
                        outputs = outputs
                    metrics = get_metrics(outputs, targets)
                    # Place holder for wandb writer
                    global_step += 1

                    # Append metrics to list
                    list_of_dice_scores.append(metrics["dice"])
                    list_of_iou_scores.append(metrics["iou"])
                    list_of_precision_scores.append(metrics["precision"])
                    list_of_recall_scores.append(metrics["recall"])
                    list_of_accuracy_scores.append(metrics["accuracy"])
                    list_of_hd95_scores.append(metrics["hd95"])

                    # Update tqdm postfix with running averages and latest metrics
                    avg_loss_so_far = running_loss / (batch_idx + 1)
                    avg_dice_loss_so_far = running_dice_loss / (batch_idx + 1)
                    avg_ce_loss_so_far = running_ce_loss / (batch_idx + 1)
                    current_lr = optimizer.param_groups[0].get("lr", 0.0)
                    pending_updates += 1
                    if ((batch_idx + 1) % update_every == 0) or (
                        batch_idx + 1 == len(self.train_loader)
                    ):
                        if update_every > 0:
                            train_bar.set_postfix(
                                loss=f"{avg_loss_so_far:.4f}",
                                dice=f"{metrics['dice']:.3f}",
                                iou=f"{metrics['iou']:.3f}",
                                dl=f"{avg_dice_loss_so_far:.4f}",
                                cel=f"{avg_ce_loss_so_far:.4f}",
                                lr=f"{current_lr:.1e}",
                            )
                        train_bar.update(pending_updates)
                        pending_updates = 0

                # Update the scheduler
                scheduler.step()
                avg_loss = running_loss / len(self.train_loader)
                avg_dice_loss = running_dice_loss / len(self.train_loader)
                avg_ce_loss = running_ce_loss / len(self.train_loader)

                elapsed_time = time.time() - start_time
                # Calculate average metrics
                metrics_epoch = {
                    "dice": np.mean(list_of_dice_scores),
                    "iou": np.mean(list_of_iou_scores),
                    "precision": np.mean(list_of_precision_scores),
                    "recall": np.mean(list_of_recall_scores),
                    "accuracy": np.mean(list_of_accuracy_scores),
                }
                tqdm.write(
                    f"Epoch [{epoch + 1}/{self.num_epochs}] | Loss: {avg_loss:.4f} | Dice Loss: {avg_dice_loss:.4f} | CE Loss: {avg_ce_loss:.4f} | Time: {elapsed_time:.2f}s"
                )
                if epoch % self.args.train_metrics_print_frequency == 0:
                    tqdm.write(
                        f"Train Metrics | Dice: {metrics_epoch['dice']:.4f} | IoU: {metrics_epoch['iou']:.4f} | "
                        f"Precision: {metrics_epoch['precision']:.4f} | Recall: {metrics_epoch['recall']:.4f} | "
                        f"Accuracy: {metrics_epoch['accuracy']:.4f}"
                    )

                # Validation
                val_loss = self.validate(epoch)

                # Log training epoch metrics
                self._safe_log(
                    {
                        "train/loss": avg_loss,
                        "train/dice_loss": avg_dice_loss,
                        "train/ce_loss": avg_ce_loss,
                        "train/dice": float(metrics_epoch["dice"]),
                        "train/iou": float(metrics_epoch["iou"]),
                        "train/precision": float(metrics_epoch["precision"]),
                        "train/recall": float(metrics_epoch["recall"]),
                        "train/accuracy": float(metrics_epoch["accuracy"]),
                        "train/epoch_time_s": float(elapsed_time),
                        "train/epoch": int(epoch),
                        "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                        "batch_size": int(getattr(self.args, "batch_size", 0) or 0),
                    }
                )

                # Save model checkpoint
                best_val_loss = self.save_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    save_dir=self.args.save_dir,
                )

                if val_loss > best_val_loss:
                    count_patience_epochs += 1
                    print(
                        f"Model is not improving, early stopping warning - Patience Count: {count_patience_epochs} / {self.args.patience_epoch}"
                    )
                else:
                    count_patience_epochs = 0
            else:
                print(
                    f"Early Stopping at epoch {epoch} with best validation loss {best_val_loss:.4f} due to model not improving for {self.args.patience_epoch} epochs"
                )
                break

    def validate(self, epoch):
        """
        Validate the model
        """
        val_loss = 0.0
        self.model.eval()
        val_loss = 0.0
        running_dice_loss = 0.0
        running_ce_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data) in enumerate(self.val_loader):
                inputs = data["pixel_values"]
                targets = data["labels"]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss, dice_loss, ce_loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()
                running_dice_loss += dice_loss.item()
                running_ce_loss += ce_loss.item()
                # Log validation loss
                # This function expects the probabilities of outputs
                if self.args.from_logits:
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = outputs
                metrics = get_metrics(outputs, targets)
                # Keep last batch for sample logging
                last_inputs, last_targets, last_outputs = inputs, targets, outputs

        self._safe_log(
            {
                "val/dice": metrics["dice"],
                "val/iou": metrics["iou"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "val/accuracy": metrics["accuracy"],
                "val/hd95": metrics["hd95"],
                "val/epoch": epoch,
            }
        )

        avg_val_loss = val_loss / len(self.val_loader)
        avg_dice_loss = running_dice_loss / len(self.val_loader)
        avg_ce_loss = running_ce_loss / len(self.val_loader)
        # log the losses and metrics to the writer
        self._safe_log(
            {
                "val/loss": avg_val_loss,
                "val/dice_loss": avg_dice_loss,
                "val/ce_loss": avg_ce_loss,
                "val/epoch": epoch,
            }
        )
        # Log a few validation samples as images (once per epoch)
        try:
            if self._can_log():
                # Prepare predictions
                if last_outputs.dim() == 4 and last_outputs.size(1) > 1:
                    up_logits = last_outputs
                    preds = up_logits.softmax(dim=1).argmax(dim=1)
                else:
                    logits_or_probs = (
                        last_outputs[:, 0] if last_outputs.dim() == 4 else last_outputs
                    )
                    if self.args.from_logits:
                        probs = logits_or_probs
                    else:
                        probs = torch.sigmoid(logits_or_probs)
                    
                    preds = (probs > 0.5).long()

                # Select up to 4 samples
                num_samples = min(4, preds.size(0))
                images = last_inputs[:num_samples].detach().cpu()
                gt = last_targets[:num_samples].detach().cpu()
                pr = preds[:num_samples].detach().cpu()

                # Convert masks to 3 channels for visualization
                gt_rgb = (
                    gt.unsqueeze(1).repeat(1, 3, 1, 1)
                    if gt.dim() == 3
                    else gt.repeat(1, 3, 1, 1)
                )
                pr_rgb = (
                    pr.unsqueeze(1).repeat(1, 3, 1, 1)
                    if pr.dim() == 3
                    else pr.repeat(1, 3, 1, 1)
                )

                # Build a grid [img, gt, pred] per sample
                triplets = []
                for i in range(num_samples):
                    triplets.extend([images[i], gt_rgb[i].float(), pr_rgb[i].float()])
                grid = make_grid(triplets, nrow=3, normalize=True, padding=2)
                self._safe_log(
                    {
                        "val/samples": wandb.Image(
                            grid.numpy().transpose(1, 2, 0),
                            caption=f"Epoch {epoch+1}: Input | GT | Pred",
                        ),
                        "val/epoch": epoch,
                    }
                )
        except Exception:
            pass
        tqdm.write(
            f"Validation [{epoch + 1}/{self.num_epochs}] | Loss: {avg_val_loss:.4f} | Dice Loss: {avg_dice_loss:.4f} | CE Loss: {avg_ce_loss:.4f}"
        )
        # Print all metrics values after each epoch
        tqdm.write(
            f"Val Metrics | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        return avg_val_loss

    def test(self, model):

        with torch.no_grad():
            for batch_idx, (data) in tqdm(enumerate(self.test_loader)):
                inputs = data["pixel_values"]
                targets = data["labels"]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                # This function expects the probabilities of outputs
                if self.args.from_logits:
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = outputs
                metrics = get_metrics(outputs, targets, threshold=self.args.threshold)

        print(
            "Dice: ",
            metrics["dice"],
            "IoU: ",
            metrics["iou"],
            "Precision: ",
            metrics["precision"],
            "Recall: ",
            metrics["recall"],
            "Accuracy: ",
            metrics["accuracy"],
            "HD95: ",
            metrics["hd95"],
            "FPR: ",
            metrics["fpr"],
            "TPR: ",
            metrics["tpr"],
        )
        # df = pd.DataFrame(metrics)
        # df.to_csv(f"{self.args.save_dir}/metrics.csv", index=False)

        # log metrics in table
        self._log_table(
            "test_metrics",
            ["Metric", "Value"],
            [
                ["Dice", metrics["dice"]],
                ["IoU", metrics["iou"]],
                ["Precision", metrics["precision"]],
                ["Recall", metrics["recall"]],
                ["Accuracy", metrics["accuracy"]],
                ["HD95", metrics["hd95"]],
                ["FPR", metrics["fpr"]],
                ["TPR", metrics["tpr"]],
            ],
        )

        return metrics

    def save_model_checkpoint(
        self, model, optimizer, epoch, val_loss, best_val_loss, save_dir="./weights"
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, f"best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(
                f"Best Model saved at {checkpoint_path} with validation loss {val_loss:.4f}"
            )
        return best_val_loss
