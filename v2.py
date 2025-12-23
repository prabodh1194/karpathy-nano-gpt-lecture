import logging
import os
import time

from smart_open import open
import torch

from checkpoint import save_checkpoint
from model import GPTLanguageModel

# configure logging with microseconds
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# hyperparameters
batch_size = 64 // 2  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = "mps"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
checkpoint_dir = "checkpoints"
checkpoint_every_n_evals = 3  # save every N evals
# ------------

torch.manual_seed(1337)

with open(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "r",
    encoding="utf-8",
) as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(ls):
    return "".join(
        [itos[i] for i in ls]
    )  # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# model config for checkpointing
model_config = {
    "n_embed": n_embed,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
    "vocab_size": vocab_size,
}

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embed=n_embed,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device,
)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

eval_count = 0
for iter in range(max_iters):
    t1 = time.time()

    # evaluate and checkpoint periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        t2 = time.time()
        logger.info(
            f"step {iter}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}, eval_time={t2 - t1:.2f}s"
        )
        eval_count += 1
        if eval_count % checkpoint_every_n_evals == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{iter}.pt")
            save_checkpoint(model, optimizer, iter, losses, ckpt_path, model_config)

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    t2 = time.time()

    logger.info(f"step {iter}: optim_time={t2 - t1:.2f}s")

# save final checkpoint
final_losses = estimate_loss()
final_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
save_checkpoint(
    model, optimizer, max_iters, final_losses, final_ckpt_path, model_config
)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
logger.info(
    f"Generated text:\n{decode(m.generate(context, max_new_tokens=500)[0].tolist())}"
)
