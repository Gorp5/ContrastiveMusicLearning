import argparse
import torch


from AblationEvaluation import masking_ratio
from contrastive_training import train_contrastive
from data.data_utils import StreamViewDataset
from models.Myna import Myna
from utils.Config import Config

masking_ratios = [0.25, 0.5, 0.75, 0.9]
training_chunk_lengths = [128, 256, 512, 1024, 2048]
embedding_strategy = ["alibi_2d_learned", "alibi_1d", "alibi_2d", "rope_1d", "rope_2d", "sinusoidal_raster", "learned_x", "none", "sinusoidal_xy", "rope_double_frequency"]
# alibi_x             = [True,  True,  True,  False, False, False, False, False, False, False]
# alibi_y             = [True,  False, True,  False, False, False, False, False, False, False]
# alibi_learned_slopes= [True,  False, False, False, False, False, False, False, False, False]
# rope_x              = [False, False, False, True,  True,  False, False, False, False, True]
# rope_y              = [False, False, False, False, True,  False, False, False, False, True]
# sinusoidal_raster   = [False, False, False, False, False, True,  False, False, False, False]
# sinusoidal_x        = [False, False, False, False, False, False, False, False, True,  False]
# sinusoidal_y        = [False, False, False, False, False, False, False, False, True,  False]
# learned_x           = [False, False, False, False, False, False, True,  False, False, False]
# learned_y           = [False, True,  False, True,  False, False, True,  False, False, False]

BASE_CONFIG = dict(
    alibi_x=False,
    alibi_y=False,
    alibi_learned_slopes=False,
    rope_x=False,
    rope_y=False,
    sinusoidal_raster=False,
    sinusoidal_x=False,
    sinusoidal_y=False,
    learned_x=False,
    learned_y=False,
)

embedding_configs = [
    dict(name="alibi_2d_learned", alibi_x=True, alibi_y=True, alibi_learned_slopes=True),
    dict(name="alibi_1d",         alibi_x=True),
    dict(name="alibi_2d",         alibi_x=True, alibi_y=True),
    dict(name="rope_1d",          rope_x=True),
    dict(name="rope_2d",          rope_x=True, rope_y=True),
    dict(name="sinusoidal_raster", sinusoidal_raster=True),
    dict(name="learned_x",        learned_x=True, learned_y=True),
    dict(name="none"),
    dict(name="sinusoidal_xy",    sinusoidal_x=True, sinusoidal_y=True),
    dict(name="rope_double_frequency", rope_x=True, rope_y=True),
]

embedding_strategy_params = []
def determine_based_on_id(id):
    masking_ratio_index = id % 4
    training_length_index = (id // 4) % 5
    type_index = (id // 20) % 10

    config = BASE_CONFIG.copy()
    config.update(embedding_configs[type_index])

    return masking_ratios[masking_ratio_index], training_chunk_lengths[training_length_index], config

import sys
import os

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--latent_projection_method", type=str, required=False, default="cls")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--rope_base", type=int, required=False, default=4096)
    parser.add_argument("--use_time_chunking", type=bool, required=False, default=False)

    id = sys.argv[1]
    masking_ratio, training_chunk_lengths, params = determine_based_on_id(id)

    use_alibi_x = params["alibi_x"]
    use_alibi_y = params["alibi_y"]
    use_learned_alibi_slopes = params["alibi_learned_slopes"]
    use_rope_x = params["rope_x"]
    use_rope_y = params["rope_y"]
    use_sinusoidal_raster = params["sinusoidal_raster"]
    use_sinusoidal_x = params["sinusoidal_x"]
    use_sinusoidal_y = params["sinusoidal_y"]
    use_learned_x = params["learned_x"]
    use_learned_y = params["learned_y"]

    print(f"Running task {id}: {params["name"]}:{id % 20}")

    args = parser.parse_args()

    per_gpu_batch = args.batch_size

    config = Config(
        save_path=args.save_dir,
        num_epochs=args.epochs,
        learning_rate=3e-4,
        weight_decay=1e-4,
        num_workers=2,
        batch_size=per_gpu_batch,
        eval_batch_size=per_gpu_batch,
        dtype=torch.float32
    )

    patch_size = (16, 16)
    if args.use_time_chunking:
        patch_size = (128, 1)

    # Determines which model is trained
    model = Myna(
        image_size=(128, args.chunk_length),
        channels=1,
        patch_size=(16, 16),
        latent_space=128,
        d_model=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        mask_ratio=args.mask_ratio,
        latent_projection_method=args.latent_projection_method,
        use_sinusoidal_x=use_sinusoidal_x,
        use_sinusoidal_y=use_sinusoidal_y,
        use_sinusoidal_raster=use_sinusoidal_raster,
        use_learned_encoding_y=use_learned_y,
        use_learned_encoding_x=use_learned_x,
        use_rope_x=use_rope_x,
        use_rope_y=use_rope_y,
        use_rope_double_frequency=False,
        use_learned_alibi_slopes=use_learned_alibi_slopes,
        use_alibi_x=use_alibi_x,
        use_alibi_y=use_alibi_y,
        rope_base=args.rope_base
    )

    train_dataset = StreamViewDataset(args.dataset_dir, views=2, chunk_size=args.chunk_length)
    test_dataset = StreamViewDataset(args.dataset_dir, views=2, chunk_size=args.chunk_length)

    train_contrastive(model, test_dataset, train_dataset, config)