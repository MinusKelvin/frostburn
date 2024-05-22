#!/usr/bin/env python

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
    print("please install the appropriate version of torch 2.3.0")
    sys.exit(1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.Linear(768, 256)
        self.l1 = torch.nn.Linear(512, 1)

    def clip(self):
        pass

    def forward(self, stm, nstm):
        stm = self.ft(stm)
        nstm = self.ft(nstm)
        x = torch.cat((stm, nstm), dim=1)

        x = torch.clamp(x, 0, 1)
        x = self.l1(x)

        return torch.sigmoid(x)


if compiling.wait() != 0:
    print("failed to compile data loader")
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
opt = torch.optim.Adam(model.parameters(), lr=0.01)

PRINT_ITERS = 100
recent_losses = torch.zeros(PRINT_ITERS).to(gpu)

ITERS = 100_000
lr_drops = [75_000, 95_000]

start = time()

train_id = strftime("%Y-%m-%d-%H-%M-%S")

for i, (stm, nstm, targets) in enumerate(batch_stream()):
    if i in lr_drops:
        opt.param_groups[0]["lr"] /= 10
    if i == ITERS:
        break

    opt.zero_grad()
    prediction = model(stm, nstm)
    loss = torch.mean(torch.abs(prediction - targets) ** 2)
    loss.backward()
    opt.step()

    with torch.no_grad():
        model.clip()

    recent_losses[i % PRINT_ITERS] = loss

    if (i + 1) % PRINT_ITERS == 0:
        print(f"\r{i+1:>8}/{ITERS}    {i * batch_size / (time() - start):>5.0f} pos/s    loss: {torch.mean(recent_losses).item():.6f}   ", end="")

print()

os.makedirs("nets", exist_ok=True)
with open(f"nets/{train_id}.json", "w") as f:
    json.dump({
        name: param.detach().cpu().tolist()
        for name, param in model.named_parameters()
    }, f)

print(train_id)
