
from tqdm import tqdm 
from torch.cuda.amp import autocast
import torch

def train_loop(cfg, model, train_loader, criterion, epoch, scaler):
    print(f'\nEpoch {epoch + 1}')
    model.train()

    tbar = tqdm(train_loader)

    losses = []
    for i, (tar_tensor, in_tensor) in enumerate(tbar):
        # print(tar_tensor.size(), in_tensor.size())
        # Forward through model
        output = model(in_tensor, tar_tensor)

        # Comput loss            
        loss = criterion(w_output, tar_tensor)
        scaler.scale(loss).backward()

        # Optimize step
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()

        # Record loss
        tbar.set_description("Train loss: %.9f, learning rate: %.6f" % (
            loss, optimizer.param_groups[-1]['lr']))
        losses.append(loss)

    losses = torch.stack(losses)
    tbar.set_description("Train loss: %.9f, learning rate: %.6f" % (
            torch.mean(losses), optimizer.param_groups[-1]['lr']))


