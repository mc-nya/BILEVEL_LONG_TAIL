import torch.nn.functional as F
from utils import topk_corrects
import torch
def train_epoch(train_loader, model, loss_fun, optimizer, cur_epoch, ITER_LR=None, ARCH_EPOCH=0, val_loader=None, val_optimizer=None, val_loss=None,dy=None,ly=None):
    """Performs one epoch of bilevel optimization."""

    # Enable training mode
    model.train()
    if val_loader:
        trainB_iter = iter(val_loader)
    total_correct=0.
    total_sample=0.
    total_loss=0.
    for cur_iter, (inputs, labels) in enumerate(train_loader):

        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        
        # Update architecture
        # if cur_epoch + cur_iter / len(train_loader[0]) >= ARCH_EPOCH:
        try:
            inputsB, labelsB = next(trainB_iter)
        except StopIteration:
            trainB_iter = iter(val_loader)
            inputsB, labelsB = next(trainB_iter)
        inputsB, labelsB = inputsB.cuda(non_blocking=True), labelsB.cuda(non_blocking=True)
        preds = model(inputsB)
        loss = val_loss(preds, labelsB,dy,ly)
        val_optimizer.zero_grad()
        
        loss.backward()
        val_optimizer.step()
        #print(dy,ly)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels,dy,ly)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        # Update the parameters
        optimizer.step()
        # Compute the errors
        mb_size = inputs.size(0)
        ks = [1,5]  # rot only has 4 classes
        top1_correct, top5_correct = topk_corrects(preds, labels, ks)
        
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct, top5_correct = top1_correct.item(), top5_correct.item()
        total_correct+=top1_correct
        total_sample+=mb_size
        total_loss+=loss*mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample*100.}')

import numpy as np

@torch.no_grad()
def eval_epoch(data_loader, model, loss_fun, cur_epoch, text,dy=None,ly=None):
    model.eval()
    correct=0.
    total=0.
    loss=0.
    class_correct=np.zeros(10,dtype=float)
    class_total=np.zeros(10,dtype=float)

    for cur_iter, (inputs, labels) in enumerate(data_loader):
        targets=labels.numpy()

        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        logits = model(inputs)
        preds = logits.data.max(1)[1]
        #print(logits,preds, targets==preds)
        mb_size = inputs.size(0)
        # if not dy is None:
        #     print(my_cross_entropy(logits,labels,dy,ly))
        loss += loss_fun(logits, labels,dy,ly).item()*mb_size
        total+=mb_size
        correct+=preds.eq(labels.data.view_as(preds)).sum().item()
    print(f'TEST {text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.}')

def my_cross_entropy(logits,targets,dy,ly):
    x=torch.transpose(torch.transpose(logits,0,1)*dy[targets],0,1)+ly
    return F.cross_entropy(x,targets)