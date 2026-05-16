import torch
import torch.nn as nn
import time
import argparse

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from model import LTSM

def sec_to_full(s):
    s = int(s)
    hr = s // 3600
    mins = (s % 3600) // 60
    sec = s % 60
    return f"{f'{hr}h ' if hr > 0 else ''}{f'{mins}m ' if mins > 0 else ''}{sec}s"

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str)
parser.add_argument("--epochs", type=int, default=16)
parser.add_argument("--save_every", type=int)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.ion()
fig, ax = plt.subplots()
losses = []
line, = ax.plot(losses)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")

X_train = torch.load("data/processed/x_train.pt")
y_train = torch.load("data/processed/y_train.pt")

print("Loaded dataset")

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = LTSM(input_size=21, hidden_dim=512, output_size=14, num_layers=8, dropout=0.1)
model = model.to(device)

print(f"Loaded model, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

_epoch, _step = 0, 0
if args.load:
    loaded_file = torch.load(f"models/checkpoints/{args.load}", weights_only=False)
    model.load_state_dict(loaded_file.get("model_state_dict"))
    optimizer.load_state_dict(loaded_file.get("optimizer_state_dict"))
    _epoch, _step = loaded_file.get("epoch", 0), loaded_file.get("step", 0)

    print("Loaded checkpoint model states")

step = 0
max_steps = len(loader) * args.epochs
t = 0

model.train()
scaler = torch.amp.GradScaler("cuda")

print(f"Starting training, {args.epochs} epochs, {max_steps:,} steps, {f'saving every {args.save_every} steps' if args.save_every else ''}")

for epoch in range(args.epochs):
    if epoch < _epoch: continue
    for x_batch, y_batch in loader:
        step += 1
        if step < _step: continue
        t = time.time_ns() / 1e+9

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        elapsed = (time.time_ns() / 1e+9) - t
        losses.append(loss.item())

        if args.save_every and step % args.save_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "step": step
            }, f"models/checkpoints/ep_{epoch + 1}_st_{step}.pt")

        if step % 20 == 0:
            print(f"Epoch: {epoch + 1} / {args.epochs}\tStep: {step:,} / {max_steps:,}\tLoss: {loss.item():,.4f}\tAvg. loss: {(sum(losses[-1000:]) / len(losses[-1000:])):,.4f}\tSpeed: {(1 / elapsed):,.2f} steps/s\tETA: {sec_to_full((max_steps - step) * elapsed)}")

            line.set_xdata(range(len(losses)))
            line.set_ydata(losses)

            ax.relim()
            ax.autoscale_view()
            
            fig.canvas.draw()
            fig.canvas.flush_events()

print("Training finished, saving model state")
torch.save(model.state_dict(), f"models/ep_{args.epochs}_st_{step}.pt")
print("Done")