import logging
import os

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, iteration, losses, path, config):
    """Save model checkpoint to disk.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        iteration: Current training iteration
        losses: Dict with 'train' and 'val' losses (or None)
        path: Path to save checkpoint
        config: Dict with model configuration
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": losses["train"] if losses else None,
        "val_loss": losses["val"] if losses else None,
        "config": config,
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint from disk.

    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        path: Path to checkpoint file
        device: Device to map tensors to

    Returns:
        Starting iteration number (0 if no checkpoint found)
    """
    if not os.path.exists(path):
        return 0
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    logger.info(
        f"Loaded checkpoint from {path} at iteration {iteration}, "
        f"train_loss={checkpoint.get('train_loss')}, val_loss={checkpoint.get('val_loss')}"
    )
    return iteration


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None

    def get_iter(fname):
        if fname == "checkpoint_final.pt":
            return float("inf")
        try:
            return int(fname.replace("checkpoint_", "").replace(".pt", ""))
        except ValueError:
            return -1

    checkpoints.sort(key=get_iter, reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])
