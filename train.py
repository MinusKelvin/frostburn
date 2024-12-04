#!/usr/bin/env python

INITIAL_LR = 0.001
SUPER_BATCHES = 400
LR_DROPS = [200, 350]
WEIGHT_DECAY = 1e-6

import ctypes, subprocess, sys, os, json
from time import time, strftime

# start compiling data loader before importing torch since that takes a while
compiling = subprocess.Popen(
    "RUSTFLAGS='-C target-cpu=native' cargo build --release -p dataload",
    shell=True,
)

try:
    import torch
except ImportError:
    print("please install the appropriate version of torch 2.3.0", file=sys.stderr)
    sys.exit(1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.Linear(768, 512)
        self.l1 = torch.nn.Linear(1024, 1)

    def clip(self):
        self.l1.weight.data = self.l1.weight.data.clamp(-127/64, 127/64)

    def forward(self, stm, nstm):
        stm = self.ft(stm)
        nstm = self.ft(nstm)
        x = torch.cat((stm, nstm), dim=1)

        x = torch.clamp(x, 0, 1)
        x = x * x
        x = self.l1(x)

        return torch.sigmoid(x)


if compiling.wait() != 0:
    print("failed to compile data loader", file=sys.stderr)
    sys.exit(1)

data_loader = ctypes.cdll.LoadLibrary("target/release/libdataload.so")
data_loader.create.restype = ctypes.c_void_p
data_loader.destroy.argtypes = [ctypes.c_void_p]
data_loader.next_batch.restype = ctypes.c_int64
data_loader.next_batch.argtypes = \
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
gpu = torch.device("cuda")
batch_size = data_loader.batch_size()

def batch_stream():
    loader = data_loader.create()
    stm_idx = torch.zeros(batch_size * 32, 2, dtype=torch.long)
    nstm_idx = torch.zeros(batch_size * 32, 2, dtype=torch.long)
    one = torch.ones(batch_size * 32, dtype=torch.float)
    targets = torch.zeros(batch_size, 1, dtype=torch.float)
    try:
        while True:
            nidx = data_loader.next_batch(
                loader,
                stm_idx.data_ptr(),
                nstm_idx.data_ptr(),
                targets.data_ptr(),
            )

            stm = torch.sparse_coo_tensor(stm_idx[:nidx, :].t(), one[:nidx], (batch_size, 768))
            stm = stm.to(gpu).to_dense()
            nstm = torch.sparse_coo_tensor(nstm_idx[:nidx, :].t(), one[:nidx], (batch_size, 768))
            nstm = nstm.to(gpu).to_dense()

            yield stm, nstm, targets.to(gpu)
    finally:
        data_loader.destroy(loader)

model = Model().to(gpu)
opt = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

B_PER_SB = 6104
recent_losses = torch.zeros(B_PER_SB).to(gpu)

start = time()

train_id = strftime("%Y-%m-%d-%H-%M-%S")

train_loss = []

for i, (stm, nstm, targets) in enumerate(batch_stream()):
    opt.zero_grad()
    prediction = model(stm, nstm)
    loss = torch.mean(torch.abs(prediction - targets) ** 2)
    loss.backward()
    opt.step()

    with torch.no_grad():
        model.clip()

    recent_losses[i % B_PER_SB] = loss.detach()

    iter_num = i + 1
    if iter_num % B_PER_SB == 0:
        sb = iter_num // B_PER_SB
        if sb in LR_DROPS:
            opt.param_groups[0]["lr"] /= 10

        print_loss = torch.mean(recent_losses).item()
        durr = time() - start
        speed = iter_num * batch_size / durr
        mins = int(durr) // 60
        secs = int(durr) % 60
        print(f"\r{sb:>8}/{SUPER_BATCHES}   {speed:>5.0f} pos/s   loss: {print_loss:.6f}   time: {mins:2}:{secs:02}    ", end="", file=sys.stderr)
        train_loss.append(print_loss)

        if sb == SUPER_BATCHES:
            break

print(file=sys.stderr)

os.makedirs("nets", exist_ok=True)
with open(f"nets/{train_id}.json", "w") as f:
    json.dump({
        name: param.detach().cpu().tolist()
        for name, param in model.named_parameters()
    }, f)

print(train_id)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; not generating loss graph")
    sys.exit(0)

plt.plot(range(SUPER_BATCHES), train_loss)
for drop in LR_DROPS:
    plt.axvline(drop, ls=":")
plt.savefig("loss.png")
