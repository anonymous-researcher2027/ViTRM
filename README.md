# ViTRM: Vision Tiny Recursion Model

Code accompanying the submission. ViTRM brings recursive reasoning in the style of Tiny Recursive Models (TRM) to image classification. A small transformer block is applied over and over to three token streams:

- **x**: the patch embeddings of the image;
- **z**: a latent state with one token per patch;
- **y**: a single global answer token.

One recursion step updates the latent state `n` times with `z ← f([x; y; z])`, then updates the answer once with `y ← f([y; z])`. Both updates use the **same** block `f`, made of 3 transformer layers. Class logits and a halting logit are read from `y`.

Training uses **deep supervision**. Each batch runs up to `N_sup` recursion steps, and every step takes its own optimizer step on `cross-entropy + BCE(halt, prediction is correct)`. The state `(y, z)` is detached between steps, and the batch stops early once the predicted halting probability exceeds a threshold. At inference, `T` recursion steps are chained.

The repository also contains the ViT and ResNet baselines, trained with the same pipeline on CIFAR-10 and CIFAR-100.


## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). It requires Python ≥ 3.9 and a machine with one NVIDIA GPU (training also runs on CPU, but slowly).

```bash
# 1. Install uv if needed (see https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the virtual environment (.venv) and install the package with its dependencies
uv sync --extra dev

# 3. (Optional) check the installation: runs on CPU in under a minute and needs no dataset
uv run pytest
```

On Linux, the default PyTorch wheels from PyPI include CUDA support. If your GPU driver needs a different CUDA build, follow uv's [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

All commands below are run with `uv run`, which uses the project environment; activating `.venv` first and dropping the prefix works too. CIFAR-10/100 are downloaded automatically to `./data` on first use (or ahead of time with `uv run vitrm download`). Pass `--data-root <dir>` to use another location.

## Usage

### Train

```bash
uv run vitrm train --config configs/vitrm/cifar10/batch_64.yaml
```

The same command trains the baselines, for example `--config configs/vit-small/cifar100/batch_128.yaml`. Each run writes to `outputs/<run-name>/`:

| File | Content |
| --- | --- |
| `config.yaml` | the exact configuration used |
| `history.json` | per-epoch train loss/accuracy/F1, test accuracy/F1, learning rate, epoch time |
| `summary.json` | best test accuracy, best test F1, best epoch, number of parameters |
| `best.pt` | checkpoint with the best test accuracy (for ViTRM, these are the EMA weights, which are the ones evaluated) |

Useful options (see `uv run vitrm train --help`):

- `--device cuda|cpu` sets the device (default `auto`: the GPU if available).
- `--run-name`, `--output-dir` set where results go.
- `--epochs N` overrides the number of epochs.
- `--num-workers N` sets the number of data-loading processes.
- `--no-mlflow` turns off experiment tracking.
- `--max-batches N` stops each epoch early, for quick debugging.

Quick sanity check, which takes a few minutes:

```bash
uv run vitrm train --config configs/vitrm/cifar10/batch_128.yaml --epochs 2 --max-batches 50 --no-mlflow
```

To run several experiments one after another on the GPU, loop over configs, for example every batch size of ViTRM on CIFAR-10:

```bash
for config in configs/vitrm/cifar10/batch_*.yaml; do
    uv run vitrm train --config "$config"
done
```

### Recursion-depth grid search (ViTRM)

This trains one ViTRM for each pair of `n` (latent steps) and `N_sup` (supervision steps). Every other setting comes from the config.

```bash
uv run vitrm grid-search --config configs/vitrm/cifar10/grid_search.yaml \
    --n-latent-steps 1 --n-latent-steps 2 --n-latent-steps 3 --n-latent-steps 6 --n-latent-steps 12 \
    --n-supervision 1 --n-supervision 2 --n-supervision 4 --n-supervision 8 --n-supervision 16
```

The values shown are the defaults. The 25 trials run sequentially. Results go to `outputs/<run-name>/n<n>-nsup<N_sup>/`.

### Evaluate and benchmark inference

```bash
uv run vitrm evaluate --checkpoint outputs/<run-name>/best.pt           # test accuracy and macro F1
uv run vitrm evaluate --checkpoint outputs/<run-name>/best.pt --benchmark \
    --batch-size 64 --output results/<run-name>.json              # + latency and throughput
```

The benchmark runs 10 warm-up batches, then times the forward pass of every test batch (using CUDA events on GPU). It reports the mean, std, median, min and max latency per batch, the latency per sample, and the throughput.

### Experiment tracking

Metrics and parameters are logged to MLflow in `./mlruns` (experiment `<dataset>-classification`). Browse them with `uv run mlflow ui`. To log to a tracking server instead, set `MLFLOW_TRACKING_URI`.

## Configuration reference

Every hyperparameter is written explicitly in the YAML files. Unknown or misspelled keys are rejected.

| Key | Meaning |
| --- | --- |
| `dataset.name`, `dataset.root` | `cifar10` / `cifar100`; download directory |
| `model.family` | `vitrm`, `vit` or `resnet` |
| `model.dim`, `num_heads`, `mlp_hidden_dim` | width, attention heads and SwiGLU hidden size of the 3-layer ViTRM block |
| `model.n_latent_steps` | **n**: latent updates of `z` per recursion step |
| `model.t_deep` | **T**: recursion steps chained at inference |
| `model.n_supervision` | **N_sup**: maximum deep-supervision steps per training batch |
| `model.halt_threshold` | stop the supervision steps of a batch when the mean halting probability exceeds this value |
| `model.dropout`, `latent_dropout`, `token_dropout` | dropout in the block, on `z` before each latent update, on the patch tokens `x` |
| `training.learning_rate`, `weight_decay`, `betas` | AdamW hyperparameters |
| `training.warmup_epochs`, `min_lr` | linear warmup from 0, then cosine decay to `min_lr` (stepped once per epoch) |
| `training.ema_decay` | EMA of the weights used for evaluation (`null` disables it; ViTRM uses 0.999) |
| `training.randaugment` | RandAugment (2 ops, magnitude 9) on top of random crop + horizontal flip |
| `training.mixup_cutmix`, `mixup_alpha`, `cutmix_alpha` | for each batch, apply Mixup or CutMix with equal probability |
| `training.seed` | random seed (cuDNN is set to deterministic mode) |

The ViT baseline fields (`dim`, `depth`, `heads`, `mlp_dim`, `dropout`, `emb_dropout`) follow [`vit-pytorch`](https://github.com/lucidrains/vit-pytorch). ResNets are the torchvision architectures (`depth` 18/34/50), randomly initialised.

Both datasets are normalised with the CIFAR-10 channel statistics. Reported metrics are top-1 accuracy and macro F1 on the official test split.
