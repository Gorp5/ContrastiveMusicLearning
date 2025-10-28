#!/usr/bin/env python
import argparse
import torch
from torch.utils.data import DataLoader

from models.PositionalEmbeddings import AttentionClamping
from utils.Config import Config
from data.data_utils import StreamViewDataset
from utils import misc
from models.Myna import Myna
from training.contrastive_training import train_contrastive


def parse_args():
    parser = argparse.ArgumentParser(description="Train Myna contrastive model")

    # Dataset paths
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test dataset")

    # Model / training params
    parser.add_argument("--model_name", type=str, default="Myna-CLS-2D-ALIBI", help="Name of the model / save folder")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    # Model hyperparameters
    parser.add_argument("--latent_space", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp_dim", type=int, default=1536)
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser.add_argument("--use_cls", action="store_true")
    parser.add_argument("--positional_encoding", type=str, default="2D-ALIBI", choices=["2D-ALIBI", "1D-ALIBI", "sinusoidal"])
    parser.add_argument("--attention_clamping", type=str, default="None", choices=["normalize", "cap_tanh", "cap_rational", "mask_rescale"])

    parser.add_argument("--chunk_size", type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup config
    config = Config(
        save_path=f"trained_models\\{args.model_name}\\",
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        dtype=getattr(torch, args.dtype),
    )

    # Datasets
    train_dataset = StreamViewDataset(args.train_path, chunk_size=args.chunk_size, views=2)
    test_dataset = StreamViewDataset(args.test_path, chunk_size=args.chunk_size, views=2)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    clamping = None
    if args.attention_clamping is not "None":
        if args.attention_clamping == "normalize":
            clamping = AttentionClamping(method="normalize", learnable=True)
        elif args.attention_clamping == "cap_tanh":
            clamping = AttentionClamping(method="cap_tanh", learnable=True)
        elif args.attention_clamping == "cap_rational":
            clamping = AttentionClamping(method="cap_rational", learnable=True)
        elif args.attention_clamping == "mask_rescale":
            clamping = AttentionClamping(method="mask_rescale", learnable=True)


    # Model
    model = Myna(
        image_size=(128, args.chunk_size),
        channels=1,
        patch_size=(16, 16),
        latent_space=args.latent_space,
        d_model=args.d_model,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        mask_ratio=args.mask_ratio,
        use_cls=args.use_cls,
        positional_encoding=args.positional_encoding,
        clamping=clamping
    )

    print(f"{misc.model_size(model)} Parameters")

    # Train
    train_contrastive(
        model,
        test_dataloader,
        train_dataloader,
        config,
        convex=args.convex,
        start_epoch=0,
        views=args.views
    )


if __name__ == "__main__":
    main()