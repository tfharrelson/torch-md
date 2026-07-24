from __future__ import annotations

import argparse
from tqdm import tqdm
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from upath import UPath

from torch_md.data.adapters import ParquetSink
from torch_md.data.config import SourcesConfig
from torch_md.data.sources import load_omol25
from torch_md.datasets import _collate_fn, create_dataset
from torch_md.mlip.transformers import SimpleMLIP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PARQUET_DIR = DEFAULT_DATA_DIR / "omol25"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "simple_mlip.pt"


# TODO: sigh unsloppify this later after i have time
# we should use a better library for making a cli tool
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SimpleMLIP on OMOL25 Calculation parquet shards."
    )
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--profile", type=bool, default=False)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on total optimizer steps for quick smoke tests.",
    )
    return parser.parse_args()


def ensure_parquet_data(parquet_dir: Path, accelerator: Accelerator) -> None:
    if any(parquet_dir.glob("*.parquet")):
        accelerator.print(f"Using existing parquet data in {parquet_dir}")
        return

    accelerator.print(
        f"No parquet shards found in {parquet_dir}; loading OMOL25 source"
    )
    src_config = SourcesConfig.get().omol25  # type: ignore
    sink = ParquetSink(UPath(parquet_dir))
    load_omol25(sink, src_config)
    accelerator.print(f"Wrote OMOL25 parquet shards to {parquet_dir}")


def build_mlip_inputs(
    batch: dict[str, Any], device: torch.device
) -> list[torch.Tensor]:
    inputs: list[torch.Tensor] = []
    for positions, masses in zip(batch["positions"], batch["masses"], strict=True):
        positions = positions.to(device=device, dtype=torch.float32)
        masses = masses.to(device=device, dtype=torch.float32)

        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(
                f"positions must have shape (num_atoms, 3), got {positions.shape}"
            )
        if masses.ndim != 1 or masses.shape[0] != positions.shape[0]:
            raise ValueError(
                "masses must have shape (num_atoms,) and match positions, "
                f"got masses={masses.shape}, positions={positions.shape}"
            )

        # TODO: replace mass with atomic number once Calculation stores it.
        inputs.append(torch.cat((positions, masses.unsqueeze(-1)), dim=-1))

    return inputs


def predict_energy_batch(model: nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([model.forward(molecule) for molecule in inputs])


def main() -> None:
    args = parse_args()
    if args.profile is True:
        pfr = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=2,
                warmup=2,
                active=3,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/profiler"),
        )
    else:
        pfr = nullcontext()
    with pfr as prof:
        accelerator = Accelerator()
        config = SourcesConfig.get()

        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        if args.log_every < 1:
            raise ValueError("--log-every must be at least 1")

        if accelerator.is_main_process:
            ensure_parquet_data(args.parquet_dir, accelerator)
        accelerator.wait_for_everyone()

        dataset = create_dataset(args.parquet_dir)  # .with_format("torch")
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataloader = DataLoader(
            dataset,  # type: ignore
            batch_size=config.omol25.batch_size,
            collate_fn=_collate_fn,
            num_workers=4,
        )
        model = SimpleMLIP(d_model=args.d_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        accelerator.print(
            "Starting training: "
            f"epochs={args.epochs}, batch_size={config.omol25.batch_size}, lr={args.lr}, "
            f"d_model={args.d_model}, device={accelerator.device}"
        )

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            epoch_steps = 0
            accelerator.print(f"Epoch {epoch}/{args.epochs} started")

            pbar = tqdm(dataloader)
            for batch in pbar:
                optimizer.zero_grad()

                inputs = build_mlip_inputs(batch, accelerator.device)
                targets = batch["energy"].to(
                    device=accelerator.device, dtype=torch.float32
                )
                predictions = predict_energy_batch(model, inputs)
                loss = F.mse_loss(predictions, targets)

                accelerator.backward(loss)
                optimizer.step()
                if prof is not None:
                    prof.step()

                loss_value = float(loss.detach().item())
                running_loss += loss_value
                epoch_steps += 1
                global_step += 1

                if global_step == 1 or global_step % args.log_every == 0:
                    avg_loss = running_loss / epoch_steps
                    pbar.set_description(
                        f"step={global_step} epoch={epoch} "
                        f"batch_loss={loss_value:.6f} avg_epoch_loss={avg_loss:.6f}"
                    )

                if args.max_steps is not None and global_step >= args.max_steps:
                    accelerator.print(f"Stopping early at max_steps={args.max_steps}")
                    break
                if args.profile and global_step >= 10:
                    break
            if args.profile and global_step >= 10:
                break

            if epoch_steps == 0:
                raise RuntimeError("No training batches were produced from the dataset")

            accelerator.print(
                f"Epoch {epoch}/{args.epochs} finished; "
                f"avg_loss={running_loss / epoch_steps:.6f}"
            )

            if args.max_steps is not None and global_step >= args.max_steps:
                break

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        accelerator.save(unwrapped_model.state_dict(), args.output_path)
        accelerator.print(f"Saved model parameters to {args.output_path}")


if __name__ == "__main__":
    main()
