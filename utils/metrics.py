import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from scipy.spatial.distance import directed_hausdorff as hausd
from typing import Optional
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


################################################################################
################################################################################
######################### Define Accuracy Metrics ##############################

def getAcc(input, target, pat_ids, norm=True):
    # first
    # input should be B, L
    assert len(target) == len(pat_ids)
    batch = input.size()[0]
    targ = target.cpu().numpy()
    ids = pat_ids.cpu().numpy()
    # apply softmax to input...
    # we could also take the value with the highest activation ENERGY...
    input = torch.softmax(input, dim=1) # is this needed?
    input = input.cpu().detach().numpy()
    # used to be 0 and plus...
    correct = []
    for i, label in enumerate(input):
        prediction = np.argmax(label)
        if np.int(prediction) == np.int(targ[i]):
            correct.append(1)
        else:
            correct.append(0)

    pat = []
    # set initial values to account for patient level classification acc...
    count = 1.
    pat_count = correct[0]
    assert len(correct) == len(input)

    for i, idx in enumerate(ids):
        # for every set of labels in bath
        if i == 0:
            patient = idx

        if idx == patient:
            # check if id is the same
            count+=1.
            pat_count += np.int(np.array(correct[i]))
        else:
            a = np.float(pat_count/count)
            if a > 0.5:
                pat.append(1)
            else:
                pat.append(0)
            # reset counter
            count = 1.
            pat_count = correct[i]

        # if last slice
        if i == np.int(len(ids) - 1):
            a = np.float(pat_count/count)
            if a > 0.5:
                pat.append(1)
            else:
                pat.append(0)

    total = np.sum(np.array(correct))
    pat_total = np.sum(np.array(pat))
    # correct at the slice level
    # normalized count vs total count
    slice = total/batch if norm == True else total
    pat = pat_total/len(pat) if norm == True else pat_total

    return slice, pat

def getMetrics(pred, target, multi=False):

    '''
    1. DICE Similarity metric that measures prediction vs ground truth contour overlap
    Can validate this with Deepind's Dice calculation.

    2. Hausendorf distance between two surfaces (ND arrays)

    '''
    shape = target.shape
    class_num = shape[1]

    # should be (B, C, H, W)
    assert pred.shape == shape
    smooth = 1.
    dices = []
    hauss = []

    for i, slice in enumerate(pred):
        channel_dice = []
        channel_hu = []
        for j, channel in enumerate(slice):
            # pred = channel
            targ = target[i, j]
            # return dice
            # complete overlap in image space...
            # volumetric dice...
            dice = 1 - dice_coeff(torch.from_numpy(channel), torch.from_numpy(targ))
            # return directed hausdorff distance
            # hu = hausd(channel, targ)
            if len(channel.shape) == 3:

                # probably a better way to calculate this at the volume level

                hu = 0

                for z, ex in enumerate(channel):
                    hu += hausd(ex, targ[z])[0]

                hu /= len(channel)
                channel_hu.append(hu)

            else:
                hu = hausd(channel, targ)
                channel_hu.append(hu[0])

            channel_dice.append(dice.item())
            # first dimenstion is the value, [1:] indices of the points that generated that distance...


        dices.append(channel_dice)
        hauss.append(channel_hu)

    batch_dice = np.mean(np.stack(dices), axis=0)
    batch_hu = np.mean(np.stack(hauss), axis=0)

    return batch_dice, batch_hu

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    For use in Average Hausendorff loss
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def dice_coeff(logits, target):

    '''
    https://github.com/pytorch/pytorch/issues/1249
    Assums prediction was passed through sigmoid & thresholded.
    For use in SoftDiceLoss...
    '''
    smooth = 1.
    num = logits.size(0)
    m1 = logits.contiguous().view(num, -1)  # Flatten
    m2 = target.contiguous().view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    # only cares about non-zero values.
    return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    # def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    #     dice=0
    #     for index in range(numLabels):
    #         dice -= dice_coef(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])
    #     return dice

################################################################################
# Define Losses
################################################################################

################################################################################
# Combined Loss
################################################################################

class CombinedLoss(nn.Module):

    '''
    link: https://github.com/EKami/carvana-challenge/blob/original_unet/src/nn/losses.py
    Make a new combined loss to include disease site classification...
    '''
    def __init__(self, topk=True, percent=0.2, new=False, single_loss='dice' ):
        super(CombinedLoss, self).__init__()
        self.topk=topk
        self.dice = SoftDiceLoss()
        self.new_loss = NewLoss(new=new, single_loss=single_loss)
        self.topk_loss = BinaryTopKCELoss(top_k=percent)
        self.bce_loss = BinaryCrossEntropyLoss2d()

    def forward(self, logits, targets):

        # bce = BinaryCrossEntropyLoss2d().forward(logits, targets)
        # if self.topk is False else None, probably something wrong with topK implementation...
        # loss  = BinaryTopKCELoss().forward(logits, targets) if self.topk == True else BinaryCrossEntropyLoss2d().forward(logits, targets) # FocalLoss().forward(logits, targets)
        # loss += NewLoss().forward(logits, targets)
        # cleaner way to implement loss?
        # FocalLoss().forward(logits, targets)
        # Question, should we add a weighting to these losses?
        loss  = self.topk_loss(logits, targets) if self.topk == True else self.bce_loss(logits, targets)
        # loss *= 0.4
        loss += self.dice(logits, targets) # *.6
        # loss += self.new_loss(logits, targets)
        # How can we do a weighted hausendorf??

        return loss

################################################################################
################# General Lossess for combined Loss above #################

class PartialDataLoss(nn.Module ):

    '''
    Link: https://github.com/EKami/carvana-challenge/blob/original_unet/src/nn/losses.py
    '''

    def __init__(self, weights=None):

        super(PartialDataLoss, self).__init__()
        self.weight = weights
        # will contain the weights for all the model classes...

    def forward(self, logits, targets):
        # once targets recieved...
        # for loop that calculates loss for 2 class for every label present in target...
        # standardize by dividing by number of classes present...
        max_ = torch.max(targets)
        count = 0
        losses  = []

        for i in range(max_):

            targ = targets.clone()
            total = (targ==i).sum()

            if total > 1:
                background = (targets!=i).sum()
                weight = torch.tensor([background/(background+total), total/(background+total)], dtype=torch.float)
                loss = nn.CrossEntropyLoss(weight=weight)
                new_logits = logits[:,i]
                targ[targ==i] = 100
                targets[targ!=i] = 0
                targ[targ==100] = 1
                value = loss(new_logits, targ)
                losses.append(value)

        return torch.stack(losses).mean()

class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        """
        Binary cross entropy loss 2D
        Args:
            weight:
            size_average:
        link: https://github.com/EKami/carvana-challenge/blob/original_unet/src/nn/losses.py
        """
        super(BinaryCrossEntropyLoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction=reduction)

    def forward(self, logits, targets):
        # probs = torch.sigmoid(logits)
        # can also use method deployed in dicecoeff...
        logits = logits.contiguous().view(-1)  # Flatten
        targets = targets.contiguous().view(-1)  # Flatten
        return self.bce_loss(logits, targets)


class SoftDiceLoss(nn.Module):
    '''
    Link: https://github.com/EKami/carvana-challenge/blob/original_unet/src/nn/losses.py
    '''
    def __init__(self, weights=None):
        super(SoftDiceLoss, self).__init__()
        self.weight = weights

    def forward(self, logits, targets):
        # convert arrays to tensor(s) .clone().detach().requires_grad_(True)
        # logits = torch.tensor(logits).type(torch.FloatTensor).requires_grad_(True)
        # targets = torch.tensor(targets).type(torch.FloatTensor).requires_grad_(True)
        # logits = logits.cpu().requires_grad_(True)
        # targets = targets.cpu().requires_grad_(True)
        # calculae dice for each class...
        max_loss = 0.
        class_ = targets.size()[1]
        threshold = 1/class_
        logits = (logits > threshold) #.float()

        # if weights are given use them...
        # max = torch.sum(self.weight) if self.weight is not None else None
        # weights = []
        for index in range(class_):

            score_ = dice_coeff(logits[:,index], targets[:, index])
            score_ *= self.weight[index] if self.weight is not None else score_
            max_loss += score_

        return max_loss/class_

        # if index==1:
        #     # CTV is combination of GTV & CTV...
        #     # fix weighting factor...
        #     # w_ = 1/((1/self.weight[index]) + (1/self.weight[index+1]))
        #     score_ *= w_ if self.weight is not None else score_
        #     # weights.append(w_)
        # else:
        # should calculate this equally for CTV/GTV...
        # weights.append(self.weight[index])

        # normalize by size of weights...
        # max_loss = max_loss/np.sum(np.array(weights))



class HardDiceLoss(nn.Module):
    def __init__(self, weight=None, format=True):
        super(HardDiceLoss, self).__init__()
        self.format = format
    def forward(self, logits, targets):
        # pred = torch.sigmoid(logits)
        if self.format is True:
            threshold = 0.5
            pred = (logits > threshold).float()
        else:
            pred=logits

        smooth = 1.
        dice = (smooth + (pred[pred==targets]).sum()*2.0) / (pred.sum() + targets.sum() + smooth)
        loss = 1. - dice
        losss = loss.type(torch.FloatTensor).to('cuda').requires_grad_(True)
        return loss

class AvgHausdorffLoss(nn.Module):
    def __init__(self):
        super(AvgHausdorffLoss, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.

        From https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/master/object-locator/losses.py

        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res

################################################################################
############# Version of TopKBCE as outlined in DEEPMINDS paper...##############

class BinaryTopKCELoss(nn.Module):

    '''
    Used when one pixel can belong to more than one class label.
    Top-K Binary Cross Entropy Loss Adapted from description in Deepminds Paper...
    Link: arXiv:1809.04430v1

    # TODO: ad a rotating loss component to this... would be an extra def to the class
    '''

    def __init__(self, top_k = .15):
        # , weight=None, reduction='none' # is this nesessary
        super(BinaryTopKCELoss, self).__init__()
        self.loss = BinaryCrossEntropyLoss2d(reduction='none')
        # nn.BCEWithLogitsLoss(reduction=reduction)
        self.top_k = top_k

    def forward(self, logits, targets):

        class_ = targets.size()[1]

        if self.top_k == 1:
            bce_loss = self.loss(logits, targets)
            assert bce_loss.size() == logits.size()
            return torch.mean(bce_loss)

        else:
            # For each channel take 10% of the max exited pixels from loss
            # can put K in a range and choose top 5/10/20% of the pixesl for
            # each class...
            valid_loss = []
            for c in range(class_):
                bce_loss = self.loss(logits[:,c], targets[:,c])
                # bce will be exported as a linear vector...
                values, idxs = torch.topk(bce_loss, int(self.top_k*len(bce_loss)))
                valid_loss.append(torch.mean(values))

            # previous implementation
            # for i, slice in enumerate(input):
            #     for j, channel in enumerate(slice):
            #         bce_loss = self.loss(channel, target[i,j])
            #         values, idxs = torch.topk(bce_loss, int(self.top_k*len(bce_loss))) # used to be channel, found error Nov 13/19
            #         valid_loss.append(torch.mean(values))

            return torch.mean(torch.stack(valid_loss)).requires_grad_(True)

        # More than one pixel can belong to more than one class
        # old implementation
        # bce_loss = self.loss(input, target)
        # print(f'\n Loss Shape is {bce_loss.size()}')
        # assert bce_loss.size() == input.size()
        # B,C,H,W => B,C,H*W
        # reshaped_loss = bce_loss.view(bce_loss.size(0),bce_loss.size(1),-1)
        # for every channe in the image, compute the loss

################################################################################
##################### New Hybrid Loss Dice + HU ################################

class NewLoss(nn.Module):
    def __init__(self, new=False, single_loss='dice'):
        super(NewLoss, self).__init__()
        self.new=new
        self.single_loss=single_loss
        self.dice = HardDiceLoss(format=False)
        self.hu = AvgHausdorffLoss()

    def forward(self, logits, targets):

        new = self.new
        single_loss = self.single_loss
        threshold = 0.5
        pred = (logits > threshold).float()
        smooth = 1.
        # check if logits in targets
        batch = targets.size()[0]
        class_ = targets.size()[1]
        loss = 0

        for c in range(class_):
            # batch imbalance
            total = targets[:,c]
            background = np.sum(total[total == 0])
            weight = np.sum(total[total==1])

        for i, slice in enumerate(pred):
            for j, channel in enumerate(slice):
                targ = targets[i,j,:,:]
                # if predictions inside targets...
                if new == True:
                    if targ[channel==1].sum() < targ.sum():
                        loss += self.dice(channel, targ)
                        # minimize DICE loss...
                    else:
                        loss += self.hu(channel, targ)
                        # minimize hausendorf distance...
                else:
                    if single_loss=='dice':
                        loss += self.dice(channel, targ)
                    elif single_loss == 'both':
                        loss += self.hu(channel, targ)
                        loss += self.dice(channel, targ)
                    else:
                        loss += self.hu(channel, targ)

        # normalize for batch
        loss = loss/batch*class_
        loss = loss.type(torch.FloatTensor).to('cuda').requires_grad_(True)

        return loss

############################################
############################################
############ Additional losses #############
############################################
############################################

class AnatFocalDLoss(nn.Module):
    def __init__(self, weights=1):
        super(AnatFocalDLoss, self).__init__()
        self.flagvec=weights

    def fdl_loss_wmask(self, y_pred, y_true):
        # if a class is present flagvec == 1
        # if self.flagvec == 1:
        #     self.flagvec = t.ones(y_pred.size()[1]).type_as(y_pred)
        alpha = 0.5
        beta  = 0.5
        ones = torch.ones_like(y_pred) #K.ones(K.shape(y_true))

        #     print(type(ones.data), type(y_true.data), type(y_pred.data), ones.size(), y_pred.size())
        p0 = y_pred      # proba that voxels are class i
        p1 = ones-y_pred # proba that voxels are not class i
        g0 = y_true.clone()
        g1 = ones-g0
        num = torch.sum(torch.sum(torch.sum(torch.sum(p0*g0*torch.pow(1-p0,2), 4),3),2),0) #(0,2,3,4)) #K.sum(p0*g0, (0,1,2,3))
        den = num + alpha*torch.sum(torch.sum(torch.sum(torch.sum(p0*g1,4),3),2),0) + beta*torch.sum(torch.sum(torch.sum(torch.sum(p1*g0,4),3),2),0) #(0,2,3,4))

        T = torch.sum(((num * self.flagvec)/(den+1e-6))) # * t.pow(1-num/(t.sum(t.sum(t.sum(t.sum(g0,4),3),2),0)+1e-5),2))
        #     Ncl = y_pred.size(1)*1.0
        #     print(Ncl, T)
        return 1.6 * (torch.sum(self.flagvec)- T) #Ncl-T

    def forward(self, logits, targets):
        # logits 5 D tensor with channels for classes
        # targets should also be a binary 5D float tensor...
        loss = self.fdl_loss_wmask(logits.cpu(), targets.cpu())

        return loss

# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class FocalLoss(_WeightedLoss):
    """
    PyTorch implementation of the Focal Loss.
    [1] "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        """
        Args:
            gamma (float): value of the exponent gamma in the definition of the Focal loss.
            weight (tensor): weights to apply to the voxels of each class. If None no weights are applied.
                This corresponds to the weights `\alpha` in [1].
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the batch size in the output,
                ``'sum'``: the output will be summed over the batch dim.
                Default: ``'mean'``.
        Example:
            .. code-block:: python
                import torch
                from monai.losses import FocalLoss
                pred = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
                grnd = torch.tensor([[0], [1], [0]], dtype=torch.int64)
                fl = FocalLoss()
                fl(pred, grnd)
        """
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma

    def forward(self, input, target):
        """
        Args:
            input: (tensor): the shape should be BCH[WD].
                where C is the number of classes.
            target: (tensor): the shape should be B1H[WD].
                The target that this loss expects should be a class index in the range
                [0, C-1] where C is the number of classes.
        """
        i = input
        t = target

        if i.ndim != t.ndim:
            raise ValueError(f"input and target must have the same number of dimensions, got {i.ndim} and {t.ndim}")

        if target.shape[1] != 1:
            raise ValueError(
                "target must have one channel, and should be a class index in the range [0, C-1] "
                + f"where C is the number of classes inferred from 'input': C={i.shape[1]}."
            )
        # Change the shape of input and target to
        # num_batch x num_class x num_voxels.
        if input.dim() > 2:
            i = i.view(i.size(0), i.size(1), -1)  # N,C,H,W => N,C,H*W
            t = t.view(t.size(0), t.size(1), -1)  # N,1,H,W => N,1,H*W
        else:  # Compatibility with classification.
            i = i.unsqueeze(2)  # N,C => N,C,1
            t = t.unsqueeze(2)  # N,1 => N,1,1

        # Compute the log proba (more stable numerically than softmax).
        logpt = F.log_softmax(i, dim=1)  # N,C,H*W
        # Keep only log proba values of the ground truth class for each voxel.
        logpt = logpt.gather(1, t.long())  # N,C,H*W => N,1,H*W
        logpt = torch.squeeze(logpt, dim=1)  # N,1,H*W => N,H*W

        # Get the proba
        pt = torch.exp(logpt)  # N,H*W

        if self.weight is not None:
            self.weight = self.weight.to(i)
            # Convert the weight to a map in which each voxel
            # has the weight associated with the ground-truth label
            # associated with this voxel in target.
            at = self.weight[None, :, None]  # C => 1,C,1
            at = at.expand((t.size(0), -1, t.size(2)))  # 1,C,1 => N,C,H*W
            at = at.gather(1, t.long())  # selection of the weights  => N,1,H*W
            at = torch.squeeze(at, dim=1)  # N,1,H*W => N,H*W
            # Multiply the log proba by their weights.
            logpt = logpt * at

        # Compute the loss mini-batch.
        weight = torch.pow(-pt + 1.0, self.gamma)
        loss = torch.mean(-weight * logpt, dim=1)  # N

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()

        raise ValueError(f"reduction={self.reduction} is invalid.")

############################################
############################################
