
import numpy as np
import torch
from tqdm import tqdm

def train_epoch(net, dl, crit, opt, device, ConfusionMatrix, get_score_sum):
    net.train()
    total_loss = 0
    cm_sum = np.zeros((2, 2))

    for t1, t2, l in dl:
        t1, t2, l = t1.to(device), t2.to(device), l.to(device)

        opt.zero_grad()
        out = net(t1, t2)

        loss = crit(out, l)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        cm_sum += ConfusionMatrix(2, out, l)

    return total_loss / len(dl), get_score_sum(cm_sum)


def val_epoch(net, dl, device, ConfusionMatrix, get_score_sum):
    net.eval()
    cm_sum = np.zeros((2, 2))

    with torch.no_grad():
        for t1, t2, l in tqdm(dl, desc="Validation"):
            t1, t2, l = t1.to(device), t2.to(device), l.to(device)
            cm_sum += ConfusionMatrix(2, net(t1, t2), l)

    return get_score_sum(cm_sum)
