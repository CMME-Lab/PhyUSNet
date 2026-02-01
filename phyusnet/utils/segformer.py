from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn as nn
import torch
import torch.nn.functional as F


class SegFormer(nn.Module):
    def __init__(
        self,
        num_labels=1,
        ignore_mismatched_sizes=True,
        return_dict=False,
        checkpoint=None,
        config=None
    ):
        super(SegFormer, self).__init__()
        # self.config = config
        self.num_labels = num_labels
        self.checkpoint = checkpoint
        if config is None:
            self.model = SegformerForSemanticSegmentation(
                config=SegformerConfig(num_labels=num_labels)
            )
        else:
            self.model = SegformerForSemanticSegmentation(
                config=config
            )
        # id2label={0: "background", 1: "foreground"},label2id={"background": 0, "foreground": 1},
        # ))
        print("Loading Segformer")
        print("Num Labels: ", num_labels)
        print("Checkpoint: ", checkpoint)
        print("Ignore Mismatched Sizes: ", ignore_mismatched_sizes)
        print("Return Dict: ", return_dict)
        if checkpoint is not None:
            self.model = self.model.from_pretrained(
                checkpoint,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                return_dict=return_dict,
                # num_labels=num_labels,
            )
        else:
            self.model = self.model.from_pretrained(
                "nvidia/mit-b5",
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                return_dict=return_dict,
                num_labels=num_labels,
            )

    def forward(self, x):
        out = self.model(x)[0]
        # lets resize the output to the original size
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    model = SegFormer(
        # checkpoint="/home/user/haris/physics_medsam/phyusnet/checkpoints/task_specific_models/segformer_clinical_BUSI"
        # checkpoint=None
    ).to(torch.device("cuda:0"))
    model.load_state_dict(torch.load("/home/user/haris/physics_medsam/phyusnet/checkpoints/task_specific_models/segformer_clinical_BUSI/best_model.pth")['model_state_dict'])
    model.train()
    y = model(x.to(torch.device("cuda:0")))
    # # print the total number of parameters
    # print(sum(p.numel() for p in model.model.parameters() if p.requires_grad))
    # # print the total number of trainable parameters in encoder
    # print(sum(p.numel() for p in model.model.encoder.parameters() if p.requires_grad))
    # # print the total number of trainable parameters in decoder
    # print(sum(p.numel() for p in model.model.decoder.parameters() if p.requires_grad))
    # print(y.shape)
    # print(model.model)
    print(y.shape)