import torch.nn.functional as F
from utils.metrics import topk_corrects
import torch
from torch.autograd import grad
import numpy as np
def gather_flat_grad(loss_grad):
    #cnt = 0
    #for g in loss_grad:
    #    g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
    #    cnt = 1
    return torch.cat([p.contiguous().view(-1) for p in loss_grad if not p is None]) #g_vector

def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        #gradient=grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True)
        #print(gradient)
        #print(d_train_loss_d_w)
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
        
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner

def train_epoch(cur_epoch, model, in_loader, in_criterion , in_optimizer, in_logit_adjust=None, in_params=None,
    is_out=False, out_loader=None, out_optimizer=None, out_criterion=None, out_logit_adjust=None, out_params=None,out_posthoc=False,
    ITER_LR=None, ARCH_EPOCH=0,num_classes=10,ARCH_INTERVAL=1,ARCH_TRAIN_SAMPLE=1,ARCH_VAL_SAMPLE=1):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    model.train()
    if is_out:
        print('lr: ',in_optimizer.param_groups[0]['lr'],'  arch lr: ',out_optimizer.param_groups[0]['lr'])
        out_iter = iter(out_loader)
        in_iter_alt=iter(in_loader)
    else:
        print('lr: ',in_optimizer.param_groups[0]['lr'])
        
    total_correct=0.
    total_sample=0.
    total_loss=0.
    arch_interval=20
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 3*num_classes
    use_reg=True
    d_train_loss_d_w = torch.zeros(num_weights).cuda()

    for cur_iter, (in_data, in_targets) in enumerate(in_loader):
        #print(cur_iter)

        # Transfer the data to the current GPU device
        in_data, in_targets = in_data.cuda(non_blocking=True), in_targets.cuda(non_blocking=True)
        # Update architecture
        if is_out and not out_posthoc:# and cur_epoch>=ARCH_EPOCH:
            model.train()
            out_optimizer.zero_grad()

            if cur_iter%ARCH_INTERVAL==0:
                for cur_iter_alt in range(ARCH_TRAIN_SAMPLE):
                    try:
                        in_data_alt, in_targets_alt = next(in_iter_alt)
                    except StopIteration:
                        in_iter_alt = iter(in_loader)
                        in_data_alt, in_targets_alt = next(in_iter_alt) 
                    in_data_alt, in_targets_alt = in_data_alt.cuda(non_blocking=True), in_targets_alt.cuda(non_blocking=True)
                    in_optimizer.zero_grad()
                    in_preds=model(in_data_alt)
                    in_loss=in_criterion(in_preds,in_targets_alt,in_params) 
                    d_train_loss_d_w+=gather_flat_grad(grad(in_loss,model.parameters(),create_graph=True))
                    #print(cur_iter_alt)
                d_train_loss_d_w/=ARCH_TRAIN_SAMPLE
                d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()

                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        out_data, out_targets = next(out_iter)
                    except StopIteration:
                        out_iter = iter(out_loader)
                        out_data, out_targets = next(out_iter) 
                #for _,(out_data,out_targets) in enumerate(out_loader):
                    out_data, out_targets = out_data.cuda(non_blocking=True), out_targets.cuda(non_blocking=True)
                    model.zero_grad()
                    in_optimizer.zero_grad()
                    out_preds = model(out_data)
                    out_loss = out_criterion(out_preds,out_targets,out_params)
                    d_val_loss_d_theta += gather_flat_grad(grad(out_loss, model.parameters(), retain_graph=use_reg))
                    # if use_reg:
                    #     direct_grad+=gather_flat_grad(grad(out_loss, get_trainable_hyper_params(out_params), allow_unused=True))
                    #     direct_grad[direct_grad != direct_grad] = 0
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                direct_grad/=ARCH_VAL_SAMPLE
                preconditioner = d_val_loss_d_theta
                
                preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 1.0,
                                                                5, model)
                indirect_grad = gather_flat_grad(
                    grad(d_train_loss_d_w, get_trainable_hyper_params(out_params), grad_outputs=preconditioner.view(-1),allow_unused=True))
                hyper_grad=indirect_grad#+direct_grad
                out_optimizer.zero_grad()
                assign_hyper_gradient(out_params,-hyper_grad,num_classes)
                out_optimizer.step()
                d_train_loss_d_w = torch.zeros(num_weights).cuda()
        
        if is_out and out_posthoc:
            try:
                out_data, out_targets = next(out_iter)
            except StopIteration:
                out_iter = iter(out_loader)
                out_data, out_targets = next(out_iter) 
            out_data, out_targets = out_data.cuda(non_blocking=True), out_targets.cuda(non_blocking=True)
            out_preds=model(out_data)
            out_preds=out_logit_adjust(out_preds,params=out_params)
            out_loss=out_criterion(out_preds,out_targets,out_params)
            out_optimizer.zero_grad()
            out_loss.backward()
            out_optimizer.step()


        # Perform the forward pass
        in_preds = model(in_data)
        if not in_logit_adjust is None:
            in_preds=in_logit_adjust(in_preds,in_params)
        # Compute the loss
        loss = in_criterion(in_preds, in_targets, in_params)
        # Perform the backward pass
        in_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        in_optimizer.step()

        # Compute the errors
        mb_size = in_data.size(0)
        ks = [1] 
        top1_correct = topk_corrects(in_preds, in_targets, ks)[0]
        
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct+=top1_correct
        total_sample+=mb_size
        total_loss+=loss*mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample*100.}')



def train_epoch_DA(cur_epoch, model, in_loader, in_criterion , in_optimizer, in_logit_adjust=None, in_params=None,
    is_out=False, out_loader=None, out_optimizer=None, out_criterion=None, out_logit_adjust=None, out_params=None,out_posthoc=False,
    ITER_LR=None, ARCH_EPOCH=0,num_classes=10,ARCH_INTERVAL=1,ARCH_TRAIN_SAMPLE=1,ARCH_VAL_SAMPLE=1):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    model.train()
    if is_out:
        print('lr: ',in_optimizer.param_groups[0]['lr'],'  arch lr: ',out_optimizer.param_groups[0]['lr'])
        out_iter = iter(out_loader)
        in_iter_alt=iter(in_loader)
    else:
        print('lr: ',in_optimizer.param_groups[0]['lr'])
        
    total_correct=0.
    total_sample=0.
    total_loss=0.
    arch_interval=20
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 3*num_classes
    use_reg=True
    d_train_loss_d_w = torch.zeros(num_weights).cuda()

    for cur_iter, (in_data, in_targets) in enumerate(in_loader):
        #print(cur_iter)

        # Transfer the data to the current GPU device
        in_data, in_targets = in_data.cuda(non_blocking=True), in_targets.cuda(non_blocking=True)
        # Update architecture
        if is_out and not out_posthoc:# and cur_epoch>=ARCH_EPOCH:
            model.train()
            out_optimizer.zero_grad()

            if cur_iter%ARCH_INTERVAL==0:
                for cur_iter_alt in range(ARCH_TRAIN_SAMPLE):
                    try:
                        in_data_alt, in_targets_alt = next(in_iter_alt)
                    except StopIteration:
                        in_iter_alt = iter(in_loader)
                        in_data_alt, in_targets_alt = next(in_iter_alt) 
                    in_data_alt, in_targets_alt = in_data_alt.cuda(non_blocking=True), in_targets_alt.cuda(non_blocking=True)
                    in_optimizer.zero_grad()
                    in_preds=model(in_data_alt)
                    in_loss=in_criterion(in_preds,in_targets_alt,in_params) 
                    d_train_loss_d_w+=gather_flat_grad(grad(in_loss,model.parameters(),create_graph=True))
                    #print(cur_iter_alt)
                d_train_loss_d_w/=ARCH_TRAIN_SAMPLE
                d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()

                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        out_data, out_targets = next(out_iter)
                    except StopIteration:
                        out_iter = iter(out_loader)
                        out_data, out_targets = next(out_iter) 
                #for _,(out_data,out_targets) in enumerate(out_loader):
                    out_data, out_targets = out_data.cuda(non_blocking=True), out_targets.cuda(non_blocking=True)
                    model.zero_grad()
                    in_optimizer.zero_grad()
                    out_preds = model(out_data)
                    out_loss = out_criterion(out_preds,out_targets,out_params)
                    d_val_loss_d_theta += gather_flat_grad(grad(out_loss, model.parameters(), retain_graph=use_reg))
                    # if use_reg:
                    #     direct_grad+=gather_flat_grad(grad(out_loss, get_trainable_hyper_params(out_params), allow_unused=True))
                    #     direct_grad[direct_grad != direct_grad] = 0
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                direct_grad/=ARCH_VAL_SAMPLE
                preconditioner = d_val_loss_d_theta
                
                preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 1.0,
                                                                5, model)
                indirect_grad = gather_flat_grad(
                    grad(d_train_loss_d_w, get_trainable_hyper_params(out_params), grad_outputs=preconditioner.view(-1),allow_unused=True))
                hyper_grad=indirect_grad#+direct_grad
                out_optimizer.zero_grad()
                assign_hyper_gradient(out_params,-hyper_grad,num_classes)
                out_optimizer.step()
                d_train_loss_d_w = torch.zeros(num_weights).cuda()
        
        if is_out and out_posthoc:
            try:
                out_data, out_targets = next(out_iter)
            except StopIteration:
                out_iter = iter(out_loader)
                out_data, out_targets = next(out_iter) 
            out_data, out_targets = out_data.cuda(non_blocking=True), out_targets.cuda(non_blocking=True)
            out_preds=model(out_data)
            out_preds=out_logit_adjust(out_preds,params=out_params)
            out_loss=out_criterion(out_preds,out_targets,out_params)
            out_optimizer.zero_grad()
            out_loss.backward()
            out_optimizer.step()


        # Perform the forward pass
        in_preds = model(in_data)
        if not in_logit_adjust is None:
            in_preds=in_logit_adjust(in_preds,in_params)
        # Compute the loss
        loss = in_criterion(in_preds, in_targets, in_params)
        # Perform the backward pass
        in_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        in_optimizer.step()

        # Compute the errors
        mb_size = in_data.size(0)
        ks = [1] 
        top1_correct = topk_corrects(in_preds, in_targets, ks)[0]
        
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct+=top1_correct
        total_sample+=mb_size
        total_loss+=loss*mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample*100.}')

@torch.no_grad()
def eval_epoch(data_loader, model, criterion, cur_epoch, text, params=None, logit_adjust=None, num_classes=10,class_wise=False):
    model.eval()
    correct=0.
    total=0.
    loss=0.
    class_correct=np.zeros(num_classes,dtype=float)
    class_total=np.zeros(num_classes,dtype=float)

    for cur_iter, (data, targets) in enumerate(data_loader):
        data, targets = data.cuda(), targets.cuda(non_blocking=True)
        logits = model(data)
        if not logit_adjust is None:
            logits=logit_adjust(logits,params)
            
        preds = logits.data.max(1)[1]
        mb_size = data.size(0)
        loss+=criterion(logits,targets,params).item()*mb_size

        total+=mb_size
        correct+=preds.eq(targets.data.view_as(preds)).sum().item()

        #preds=preds.cpu().numpy()
        #targets=targets.cpu().numpy()
        if class_wise:
            for i in range(num_classes):
                indexes=np.where(targets.cpu().numpy()==i)[0]
                class_total[i]+=indexes.size
                class_correct[i]+=preds[indexes].eq(targets[indexes].data.view_as(preds[indexes])).sum().item()
            #print(class_total,class_correct)
    text=f'TEST {text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.}'
    if class_wise:
        text=f'TEST {text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.} Class wise = {class_correct/class_total*100.}'
    print(text)
    return text,loss/total,correct/total*100.

def loss_adjust_cross_entropy(logits,targets,params):
    #assert(len(params)==2)
    dy=params[0]
    ly=params[1]
    #wy=params[2]
    #x=logits*torch.exp(dy)+ly
    #print(F.sigmoid(dy))
    x=logits*F.sigmoid(dy)+ly
    #x=torch.transpose(torch.transpose(logits,0,1)*dy[targets],0,1)+ly
    loss=F.cross_entropy(x,targets)
    #loss=wy[targets]*F.cross_entropy(x,targets)
    return loss

def cross_entropy(logits,targets,params):
    #print(logits.shape,targets)
    return F.cross_entropy(logits,targets)

def logit_adjust_ly(logits,params):
    #assert(len(params)==2)
    dy=params[0]
    ly=params[1]
    x=logits*dy-ly
    return x

def get_trainable_hyper_params(params):
    return[param for param in params if param.requires_grad]

def assign_hyper_gradient(params,gradient,num_classes):
    i=0
    for para in params:
        if para.requires_grad:
            para.grad=gradient[i:i+num_classes].clone()
            i+=num_classes
