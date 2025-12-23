import click
import torch
from smart_open import open as smart_open

from checkpoint import get_latest_checkpoint
from model import GPTLanguageModel

# load vocab from same source as training
with smart_open(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "r",
    encoding="utf-8",
) as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def decode(ls):
    return "".join([itos[i] for i in ls])


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GPTLanguageModel(
        vocab_size=config["vocab_size"],
        n_embed=config["n_embed"],
        block_size=config["block_size"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=0.0,  # no dropout during inference
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@click.command()
@click.option(
    "--checkpoint",
    "-c",
    default=None,
    help="Path to checkpoint file. Defaults to latest in checkpoints/",
)
@click.option(
    "--tokens",
    "-t",
    default=100000,
    type=int,
    help="Number of tokens to generate",
)
@click.option(
    "--device",
    "-d",
    default="mps",
    help="Device to run on (mps, cuda, cpu)",
)
def generate(checkpoint, tokens, device):
    """Generate text from a trained model checkpoint."""
    if checkpoint is None:
        checkpoint = get_latest_checkpoint("checkpoints")
        if checkpoint is None:
            raise click.ClickException("No checkpoint found in checkpoints/")

    click.echo(f"Loading checkpoint: {checkpoint}")
    model, ckpt_data = load_model_from_checkpoint(checkpoint, device)

    click.echo(f"Checkpoint from iteration {ckpt_data['iteration']}")
    if ckpt_data.get("train_loss"):
        click.echo(
            f"Train loss: {ckpt_data['train_loss']:.4f}, Val loss: {ckpt_data['val_loss']:.4f}"
        )

    click.echo(f"Generating {tokens} tokens...")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=tokens)

    generated_text = decode(output[0].tolist())
    click.echo("\n" + "=" * 80)
    click.echo(generated_text)


if __name__ == "__main__":
    generate()
