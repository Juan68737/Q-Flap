"""Metric logging.

The training loop emits clean scalars through this Logger; it does not print
metrics itself or know whether TensorBoard is installed. If TensorBoard is
available, scalars go to a `tb/` subdir of the run; otherwise it degrades to a
tidy console line. Swap this out for W&B without touching the loop.
"""

from pathlib import Path


class Logger:
    def __init__(self, run_dir, use_tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception:
                self.writer = None  # tensorboard not installed -> console only

    def log_scalars(self, step: int, **scalars) -> None:
        if self.writer is not None:
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, step)
        line = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in scalars.items())
        print(f"[step {step:>9}] {line}")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
