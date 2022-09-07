import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import *
from .ND_Crossentropy import *

class TALWrapper(nn.Module):
    def __inti__(self, weights=None, do_bg=True, class_num=1):
        assert weights is not None
        self.class_weights = weights
        self.do_bg = do_bg
        self.class_num = class_num
    def forward(self, outputs, targets):
        index_ = []
        background = [0]
        weight = []
        for i in range(1, self.class_num):
            if len(targets[targets==i] > 0):
                index_.append(i)
                weight.append(self.class_weights[i])
            else:
                background.append(i)
        # transform weight so it can be used by crossentropy...
        weight = torch.tensor(weight).type_as(outputs).float()
        weight[0] = 0.
        # for second loss component we must modulate tensors...
        back_ = outputs.clone()
        back_ = back_[:,background]
        outs_ = outputs.clone()
        outs_ = outs_[:,index_]
        back_ = torch.mean(back_,dim=1)
        warnings.warnings(f"{outs_.size()} v. {back_.size()}")
        outs_ = torch.cat([back_.unsqueeze(0), outs_], dim=1)
        targ = targets.clone()
        targ[targ>0] = 0
        for i, val in enumerate(index_):
            targ[targets==val] = i+1
        warnings.warn(f"Number of labels used are {len(weight)}")
        warnings.warn(f"{len(background)} in background were excluded.")
        ce_kwargs = {'weight': weight, "k": 15, "ignore_index":0}
        tversky_kwargs = {'batch_dice':False,'do_bg':False, 'smooth':1, 'square':False}
        loss = FocalTversky_and_topk_loss(tversky_kwargs, ce_kwargs)
        loss_ = loss(outs_, targ)
        return loss_
