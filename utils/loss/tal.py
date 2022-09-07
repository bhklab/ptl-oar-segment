import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import *
from .ND_Crossentropy import *

class TALWrapper(nn.Module):
    def __init__(self, weight=None, do_bg=True):
        super(TALWrapper, self).__init__()
        assert weight is not None
        ce_kwargs = {'weight': weight, "k": 15, "ignore_index": 0}
        tversky_kwargs = {'batch_dice': False, 'do_bg': do_bg, 'smooth': 1, 'square': False}
        self.loss = FocalTversky_and_topk_loss(tversky_kwargs, ce_kwargs)
    def forward(self, outputs, targets, mask):
        loss_ = self.loss(outputs, targets, mask)
        return loss_

        # if counts is None:
        # index_ = []
        # background = [0]
        # weight = [0.]
        # for i in range(1, self.class_num):
        #     if counts is not None:
        #         if counts[0][i] == 1:
        #             index_.append(i)
        #             weight.append(self.class_weights[i])
        #         else:
        #             background.append(i)
        #     else:
        #         if len(targets[targets==i] > 0):
        #             index_.append(i)
        #             weight.append(self.class_weights[i])
        #         else:
        #             background.append(i)
        # else:
        #     bool_counts = (counts == 1)
        #     counts_ = np.where(bool_counts)[0]
        # transform weight so it can be used by crossentropy...
        # weight = torch.tensor(weight).type_as(outputs).float()
        # assert len(weight) == len(counts)
        # # for second loss component we must modulate tensors...
        # back_ = outputs.clone()
        # back_ = back_[:,background]
        # outs_ = outputs.clone()
        # outs_ = outs_[:,index_]
        # back_ = torch.mean(back_,dim=1)
        # warnings.warn(f"{outs_.size()} v. {back_.size()}")
        # outs_ = torch.cat([back_.unsqueeze(0), outs_], dim=1)
        # targ = targets.clone()
        # targ[targ>0] = 0
        # for i, val in enumerate(index_):
        #     targ[targets==val] = i+1
        # warnings.warn(f"Number of labels used are {len(weight)}")
        # warnings.warn(f"{len(background)} in background were excluded.")

