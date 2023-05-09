import os, torch, time, datetime, warnings, pickle, json, glob, nrrd, sys
from pathlib import Path
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from .prepare import GetSplits, PatientData, LoadPatientSlices, LoadPatientVolumes
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from .scheduler import Poly
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import lightning as pl
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch.nn.functional as F
# from sliding_window import sliding_window_inference
# from .sliding_window import sliding_window_inference as swi
# from monai.inferers import sliding_window_inference as swi
from .metrics import getMetrics, CombinedLoss, SoftDiceLoss, AnatFocalDLoss
from .loss import *
from .optimizers import *
from .models import *
from .transform import *
from .utils import swi

sys.path.append('/content/drive/My Drive/ptl-oar-segmentation/')

def cuda(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device)


class SegmentationModule(pl.LightningModule):
    def __init__(self, hparams, update_lr=None):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(SegmentationModule, self).__init__()
        self.save_hyperparameters(hparams) # 1.3+
        # self.hparams = hparams # < 1.3 able to do this in older versions...
        self.get_model_root_dir()
        # self.prepare_data()
        if self.hparams.oar_version == 1:
            self.__get_gtv_data()
        else:
            # please inputput the root directory of the repository
            self.root="./ptl-oar-segmentation/"
            self.__get_data()

        self.__build_model()
        self.__get_loss()
        if update_lr is not None:
            self.hparams.lr = update_lr

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):

        """
        Layout model
        :return:
        n_classes + 1 (because we need to include background)
        """
        
        classes = self.hparams.n_classes + 1
        if self.hparams.model == "DEEPNET":
            # num_classes + 1 that includes background...
            self.net = DeepUNet(num_classes=classes, sub_enc=self.hparams.sub_enc)

        elif self.hparams.model == "SIMPLE":
            self.net = simpleUNet(num_classes=classes)

        elif self.hparams.model == "UNET":
            self.net = UNet3D(
                num_classes=classes, scale=self.hparams.scale_factor,
                deformable=self.hparams.deform, project=self.hparams.project
            )

        elif self.hparams.model == "VNET":
            self.net = VNet3D(num_classes=classes)

        elif self.hparams.model == "WOLNET":
            self.net = WolnyUNet3D(num_classes=classes, f_maps=self.hparams.f_maps)

        elif self.hparams.model == "RESUNET":
            self.net = ResUNet3D( num_classes=classes, f_maps=self.hparams.f_maps)

        elif self.hparams.model == "MODIFIED3DUNET":

            self.net = Modified3DUNet(1, n_classes=classes)

        elif self.hparams.model == "TIRAMISU":
            # 3D version of tiramisu_network...
            self.net = FCDenseNet(
                in_channels=1,
                down_blocks=(2, 2, 2, 2, 3),
                up_blocks=(3, 2, 2, 2, 2),
                bottleneck_layers=2,
                growth_rate=12,
                out_chans_first_conv=16,
                n_classes=classes,
            )

        elif self.hparams.model == "ANATOMY":
            # AnatomyNet discussed in https://github.com/wentaozhu/AnatomyNet-for-anatomical-segmentation
            self.net = AnatomyNet3D(num_classes=classes)

        elif self.hparams.model == "PIPOFAN":
            self.net = PIPOFAN3D(num_classes=classes, factor=3)

        elif self.hparams.model == "HIGHRESNET":
            self.net = HighResNet3D(classes=classes)

        elif self.hparams.model == "UNET++":
            self.net = NestedUNet( num_classes=classes, factor=4, deep_supervision=True)
        elif self.hparams.model == "VGG163D":
            self.net = VGGUNet(num_classes=classes, factor=2)

        elif self.hparams.model == "UNET3+":
            self.net = UNet_3Plus(n_classes=classes, factor=2)

        elif self.hparams.model == "UNET3+DEEPSUP":
            self.net = UNet_3Plus_DeepSup(
                n_classes=classes, factor=8
            )
        elif self.hparams.model == "RSANET":
            self.net = RSANet(n_classes=classes)

        elif self.hparams.model == "HYPERDENSENET":
            self.net = HyperDenseNet(in_channels=1, num_classes=classes)

        elif self.hparams.model == "DENSEVOX":
            self.net = DenseVoxelNet(in_channels=1, num_classes=classes)

        elif self.hparams.model == "MEDNET":
            self.net = generate_resnet3d(in_channels=1, classes=classes, model_depth=10)

        elif self.hparams.model == "SKIPDNET":
            self.net = SkipDenseNet3D(growth_rate=16, num_init_features=self.hparams.f_maps, drop_rate=0.1, classes=classes)

    def __get_gtv_data(self):
        # for GTV ...
        fold = self.hparams.fold
        # path = '/cluster/projects/radiomics/Temp/joe/'
        path = '/cluster/home/jmarsill/SegmentHN/paper/'
        train_csv_path = str(self.root) + f"/train_fold_{fold}.csv"
        valid_csv_path = str(self.root) + f"/valid_fold_{fold}.csv"
        test_csv_path = str(self.root) + f"/test_fold_{fold}.csv"
        weights_path = str(self.root) + f"/weights_{fold}.npy"
        metrics_path = str(self.root) + f"/metrics_{fold}.npy"

        if os.path.isfile(weights_path) is True:
            self.train_data = pd.read_csv(train_csv_path, index_col=0)
            self.valid_data = pd.read_csv(valid_csv_path, index_col=0)
            self.test_data = pd.read_csv(test_csv_path, index_col=0)
            metrics = np.load(metrics_path)
            self.mean = metrics[0]
            self.std = metrics[1]
            # self.class_weights = np.array([1e-4, 1.0, 10.])#load(weights_path) # /10
            print(self.train_data.head())
        else:
            self.train_data = pd.read_csv(path+'train_gtv21.csv', index_col=0)
            self.valid_data = pd.read_csv(path+'val_gtv21.csv', index_col=0)
            self.test_data = pd.read_csv(path+'test_gtv21.csv', index_col=0)

        print('Loading GTV data.')
        #######################################################
        fold = self.hparams.fold
        # path = '/cluster/projects/radiomics/Temp/joe/'
        path = '/cluster/home/jmarsill/SegmentHN/paper/'
        train_csv_path = str(self.root) + f"/train_fold_{fold}.csv"
        valid_csv_path = str(self.root) + f"/valid_fold_{fold}.csv"
        test_csv_path = str(self.root) + f"/test_fold_{fold}.csv"
        weights_path = str(self.root) + f"/weights_{fold}.npy"
        metrics_path = str(self.root) + f"/metrics_{fold}.npy"
        ######################################################

        # Train editing...
        tr = list(self.train_data['0'])
        tr_ = ['0'+str(i) for i in tr if len(str(i)) < 7]
        tr_1 = ['0'+str(i) for i in tr_ if len(str(i)) < 7]
        tr_ = [str(i) for i in tr_ if len(str(i)) == 7]
        tr_ += [str(i) for i in tr if len(str(i)) == 7]
        tr_ += tr_1
        # random.shuffle(tr_)
        self.train_data['0'] = tr_

        # valid editing
        tr = list(self.valid_data['0'])
        tr_ = ['0'+str(i) for i in tr if len(str(i)) < 7]
        tr_1 = ['0'+str(i) for i in tr_ if len(str(i)) < 7]
        tr_ = [str(i) for i in tr_ if len(str(i)) == 7]
        tr_ += [str(i) for i in tr if len(str(i)) == 7]
        tr_ += tr_1
        # random.shuffle(tr_)
        self.valid_data['0'] = tr_

        # test editing...
        tr = list(self.test_data['0'])
        tr_ = ['0'+str(i) for i in tr if len(str(i)) < 7]
        tr_1 = ['0'+str(i) for i in tr_ if len(str(i)) < 7]
        tr_ = [str(i) for i in tr_ if len(str(i)) == 7]
        tr_ += [str(i) for i in tr if len(str(i)) == 7]
        tr_ += tr_1
        # random.shuffle(tr_)
        self.test_data['0'] = tr_

        # metrics...
        self.class_weights = [0.0001, .9999]
        self.mean = -410.6404651018592
        self.std = 221.25601368260544

        if os.path.isfile(weights_path) is False:
            self.train_data.to_csv(train_csv_path)
            self.valid_data.to_csv(valid_csv_path)
            self.test_data.to_csv(test_csv_path)
            np.save(metrics_path, np.array([self.mean, self.std]))
            np.save(weights_path, self.class_weights)

        print( f"Training Set Mean HU: {self.mean} Mean STD: {self.std} \n" ) if self.hparams.verbose else None
        print( f"Using this weight array to mitigate class imbalance {self.class_weights} \n") if self.hparams.verbose else None
        print(self.train_data.head(), f'There are {len(self.train_data)} training scans, \n{len(self.valid_data)} validation scans and \n{len(self.test_data)} scans for testing.')

    # ------------------
    # Extract Dataframes
    # ------------------
    
    def __get_data(self):
        
        self.class_weights = [0.1, 1.3, 5.5]
        self.Kfold = None
        self.val_loss = torch.tensor([0], dtype=torch.float)
        fold = self.hparams.fold
        pkl_name = self.hparams.pkl_name 
        dataset_path = f"{self.root}wolnet-sample/Kfold_WOLNET_2020_08_28_152828.pkl"
        
        # resume from previous crash
        print(f"\n Load folds from: {dataset_path}")
        self.Kfold = self.load_obj(dataset_path) 
        
        print('Loading OAR data.')
        # set mean/std to normalize data
        # these are the standard values for clipped image from -500 to 1000
        if self.hparams.clip_max < 300:
            self.mean = self.Kfold["means"][fold] # - 300.
            self.std =  self.Kfold["stds"][fold] # + 75
        else:
            self.mean = -407.4462155135238 # Kfold["means"][fold] # - 300.
            self.std = 226.03663728492648 # Kfold["stds"][fold] # + 75
        # should return a numpy array of class weights n_classes + 1
        self.class_weights = self.Kfold["weights"][fold]
        print( f"Training Set Mean HU: {self.mean} Mean STD: {self.std} \n") if self.hparams.verbose else None
        print( f"Using this weight array to mitigate class imbalance {self.class_weights} \n") if self.hparams.verbose else None

        # if paramater files from old training sessions exist - load them...
        param_path = str(self.root) + f"/tensoborad_logs_{fold}.json"

        print( "\n Number of Patient Scans in Training:", len(self.Kfold["train"][0]),
            "\n Patients for Validation:", len(self.Kfold["valid"][0]),
            "\n Patients for Testing", len(self.Kfold["test"])) if self.hparams.verbose else None
        
        self.train_data = pd.read_csv(f"{self.root}wolnet-sample/new_train_fold_{fold}.csv")
        self.valid_data = pd.read_csv(f"{self.root}wolnet-sample/new_valid_fold_{fold}.csv")
        self.test = pd.read_csv(f"{self.root}wolnet-sample/new_test_fold.csv")
        self.test_name = '_RADCURE'

    # ------------------
    # Assign Loss
    # ------------------
    def __get_loss(self):
        # self.class_weights this must be provided and calculated separately...
        # usually this will be the amount of voxels given for any OAR class...
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights) # *.01
            if self.hparams.volume_type=='targets':
                pass
            else:
                weight -= weight.min() - 1e-4
                weight /= weight.max() + 1e-4
                weight /= weight.sum() + 1e-4
                weight *= 100
                weight[0] = 1e-4

            self.class_weights = ( weight * self.hparams.scale_weights).float()
            print(f"Weights are now: {self.class_weights}")

        self.criterion = None
        self.criterion2 = None
        self.criterion3 = None
        self.criterion4 = None

        if self.hparams.loss == "FOCALDSC":
            # binary TOPK loss + DICE + HU
            # enables us to tackle class imbalance...
            # Metric from elektronn3
            loss = FocalLoss(weight=self.class_weights)
            dice_loss = SoftDiceLoss(weights=self.class_weights)
            self.criterion = loss
            self.criterion2 = dice_loss

        elif self.hparams.loss == "CATEGORICAL":
            loss = CrossEntropyLoss(weight=self.class_weights)
            self.criterion = loss

        elif self.hparams.loss == "FOCAL":
            # needs same structure/dim as DICE ... (AnatomyNet Loss...)
            loss = FocalLoss(weight=self.class_weights)
            self.criterion2 = loss

        elif self.hparams.loss == 'ANATOMY':
            loss = AnatFocalDLoss(weights=self.class_weights)
            self.criterion2 = loss

        elif self.hparams.loss == 'HDBDLOSS':
            soft_dice_kwargs = {'batch_dice':False, 'do_bg':True, 'smooth':1., 'square':False}
            loss =  SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)# DistBinaryDiceLoss()
            loss2 = CrossEntropyLoss(weight=self.class_weights) # HDDTBinaryLoss(onehot=False)
            self.criterion = loss
            self.criterion4 = loss2

        elif self.hparams.loss == "WDCTOPK":
            ce_kwargs = {'weight':self.class_weights}
            soft_dice_kwargs = {'batch_dice':False, 'do_bg':True, 'smooth':1., 'square':False, 'weight':self.class_weights}
            # loss = WeightedCrossEntropyLoss(weight=self.class_weights)
            loss = DC_and_topk_loss(soft_dice_kwargs, ce_kwargs)
            self.criterion = loss

        elif self.hparams.loss == "WFTTOPK":
            ce_kwargs = {'weight':self.class_weights}
            tversky_kwargs = {'batch_dice':False, 'do_bg':True, 'smooth':1., 'square':False}
            # can add weight class if necessary ...
            loss = FocalTversky_and_topk_loss(tversky_kwargs, ce_kwargs)
            self.criterion = loss

        elif self.hparams.loss == "COMBINED":
            loss = CrossEntropyLoss(weight=self.class_weights)
            dice_loss = SoftDiceLoss(weights=self.class_weights)
            self.criterion = loss
            self.criterion2 = dice_loss if dice_loss is not None else None

        elif self.hparams.loss == 'COMBINEDFOCAL':
            loss = CrossEntropyLoss(weight=self.class_weights)
            foc_loss = AnatFocalDLoss(weights=self.class_weights)
            dice_loss = SoftDiceLoss(weights=self.class_weights)
            self.criterion = loss
            self.criterion2 = foc_loss if foc_loss is not None else None
            self.criterion3 = dice_loss if dice_loss is not None else None

        else:
            warnings.warn("Using Standard DICE loss. One Hot encoded target required.")
            loss = SoftDiceLoss(weight=self.class_weights)
            self.criterion2 = loss

    # ---------------------
    # SAVE Figures : Plot to Tensorboard
    # ---------------------
    def save_metrics(self, batch_idx, loss, dices, hus, name="train"):

        if batch_idx % 5 == 0:
            # logging during training
            step = self.trainer.global_step
            # try:
            self.logger.experiment.add_scalar(f"{name}/loss", loss, step)
            self.logger.experiment.add_scalars(f"{name}/dices", dices, step)
            self.logger.experiment.add_scalars(f"{name}/hausd", hus, step)

    def export_figs(self, inputs, outputs, targets):

        # plt.switch_backend('agg')
        # assume sigmoid applied to outputs
        shape = outputs.shape  # size of outputs

        if shape[0] != 1:
            # choose random image in batch
            a = np.arange(1, shape[0])
            a = np.random.choice(a)
        else:
            # satisfies case when only one image in batch
            a = 0

        in_pic = inputs[a]
        out_pic = outputs[a]
        targ_pic = targets[a]

        # should be the same dimensions
        assert out_pic.shape == targ_pic.shape

        in_ = in_pic[self.hparams.window]

        if len(targ_pic.shape) == 3:
            # plt.imshow(np.sum(targ_pic, axis=0), cmap='jet', alpha=0.5)
            targ = np.sum(targ_pic, axis=0)
        else:
            # targets already in n_classes - 1 (0 background: 1 CTV, 2 GTV)
            # plt.imshow(targ_pic, cmap='jet', alpha=0.5)
            targ = targ_pic

        if len(out_pic.shape) == 3:
            # plt.imshow(np.sum(out_pic, axis=0), cmap='jet', alpha=0.5)
            out = np.sum(out_pic, axis=0)
        else:
            # targets already in n_classes - 1 (0 background: 1 CTV, 2 GTV)
            # plt.imshow(out_pic, cmap='jet', alpha=0.5)
            out = out_pic
        return in_, out, targ

    def save_figs(self, in_, out, targ, name="train", step=None):

        if step is None:
            step = self.trainer.global_step

        plt.switch_backend("agg")
        # plot & save ground truth targets
        fig1 = plt.figure(1)
        plt.axis("off")
        plt.imshow(in_, cmap="gray", interpolation="none")
        plt.imshow(targ, cmap="jet", alpha=0.5)
        # save targets
        self.logger.experiment.add_figure(f"{name}/ground_truth", fig1, step)
        plt.close()

        # plot and save predicated masks
        fig2 = plt.figure(2)
        plt.axis("off")
        plt.imshow(in_, cmap="gray", interpolation="none")
        plt.imshow(out, cmap="jet", alpha=0.5)
        # save model predictions
        self.logger.experiment.add_figure(f"{name}/prediction", fig2, step)
        plt.close()

    def forward(self, x):
        # when running deepnet moduel...
        # make sure output is unsqueezed...
        x = x.unsqueeze(1)
        return self.net(x)

    def metrics(self, outs, targ, outputs=None):
        # Metrics requiring one hot encoded targets, pass through sigmoid or softmax
        # convert to one hot encoded target...
        shape = targ.size()
        batch = shape[0]
        # calculate argmax...
        outputs = torch.argmax(outs, dim=1)
        if len(shape) == 4:
            # 3D OUTPUT ARRAY HAS TO BE 5D IF ONE HOT ENCODED
            # see what you do when you initialize a new tensor
            # https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html
            sh = (batch, self.hparams.n_classes + 1, shape[1], shape[2], shape[3])
            targets_dice = torch.zeros(sh, dtype=torch.float) # .type_as(outs)
            targets_out = torch.zeros(sh, dtype=torch.float) # .type_as(targ)

        for i in range(self.hparams.n_classes + 1):

            targets_dice[:, i][targ == i] = 1
            targets_out[:, i][outputs == i] = 1

        # calculate distance accuracy metrics
        losses = ['ANATOMY', 'COMBINEDFOCAL']

        if self.hparams.loss in losses:
            # assert outs without argmax...
            loss = (self.criterion2(outs, targets_dice) if self.criterion2 is not None else 0)
        else:
            loss = (self.criterion2(targets_out, targets_dice) if self.criterion2 is not None else 0)

        # if self.criterion3 is not None:
        loss += (self.criterion3(targets_out, targets_dice) if self.criterion3 is not None else 0)

        dices, hauss = getMetrics(targets_out.detach().numpy(), targets_dice.detach().numpy(), multi=True)

        #retrun all metrics including argmaxed outputs...
        return dices, hauss, loss, outputs

    # ---------------------
    # TRAINING
    # ---------------------
    def training_step(self, batch, batch_idx):

        print("AT TRAIN START")
        self.step_type = "train"
        inputs, targets = batch

        if inputs.shape != targets.shape:
            warnings.warn("Input Shape Not Same size as label...")
        if batch_idx == 0:
            print(inputs.max(), inputs.size())
            print(targets.max(), targets.size())

        outputs = self.forward(inputs)

        if type(outputs) == tuple:
            outputs = outputs[0]
        if self.hparams.crop_as != "3D":
            outputs = outputs.squeeze(2)

        loss = self.criterion(outputs, targets)
        outputs, targets = onehot(outputs, targets)

        # take this out during training...
        # hdfds = monmet.compute_hausdorff_distance(outputs, targets, percentile=95,
        #                                           include_background=True)
        dices = monmet.compute_meandice(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # get size of dice array,
        # fist dim should be that of batch...
        s = dices.size()
        if s[0]==1:
            dices = dices[0]
            # hdfds = hdfds[0]
        else:
            dices=dices.mean(dim=0)
            # hdfds=hdfds.mean(dim=0)

        for i, val in enumerate(dices):
            # for the NEW ptl...
            self.log(f'train_dice_{i}', dices[i], on_step=True, prog_bar=True, logger=True)
            # self.log(f'train_haus_{i}', hdfds[i], on_step=True, logger=True)
        return {'loss':loss}

    # ---------------------
    # Validation
    # ---------------------
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        self.step_type = "valid"
        inputs, targets = batch
        shape = inputs.size()

        if batch_idx == 0:
            print(inputs.max(), inputs.size())
            print(targets.max(), targets.size())

        if inputs.shape != targets.shape:
            warnings.warn("Input Shape Not Same size as label...")

        # change this to incorporate sliding window...
        # roi_size = (int(shape[1]), 192, 192) # (64, 192, 192)
        # sw_batch_size = 1 # second sliding window inference
        # outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward)
        # evaluation using sliding window inference only (really) required during testing.

        outputs = self.forward(inputs)
        if type(outputs) == tuple:
            outputs = outputs[0]
        if self.hparams.crop_as != "3D":
            outputs = outputs.squeeze(2)  # shouldn't this be conditional??
        loss = self.criterion(outputs,targets) # (self.criterion(outputs, targets.unsqueeze(1)).cpu() if self.criterion is not None else 0)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # apply soft/argmax to outputs...
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs , dim=1)

        ################
        # grab images to save during training...
        in_   = inputs.cpu().numpy()
        targ_ = targets.cpu().numpy()
        out_  = outputs.cpu().detach().numpy()
        in_, out_, targ_ = self.export_figs(in_, out_, targ_)
        ################
        # calculating metrics
        outputs, targets = onehot(outputs, targets, argmax=False)
        hdfds = monmet.compute_hausdorff_distance(outputs, targets, percentile=95,
                                                  include_background=True)
        dices = monmet.compute_meandice(outputs, targets)
        print(dices.size(), hdfds.size())
        print(dices,hdfds)
        output = OrderedDict(
            {
                "val_loss": loss,
                "input": torch.from_numpy(in_).type(torch.FloatTensor),
                "out": torch.from_numpy(out_).type(torch.FloatTensor),
                "targ": torch.from_numpy(targ_).type(torch.FloatTensor),
            }
        )

        s = dices.size()
        if s[0]==1:
            dices = dices[0]
            hdfds = hdfds[0]
        else:
            dices=dices.mean(dim=0)
            hdfds=hdfds.mean(dim=0)

        for i, val in enumerate(dices):
            self.log(f'val_dice_{i}', dices[i], on_step=True, prog_bar=True, logger=True)
            self.log(f'val_haus_{i}', hdfds[i], on_step=True, logger=True)
            output[f"val_dice_{i}"] = dices[i]
            output[f"val_haus_{i}"] = hdfds[i]

        return output

    def validation_epoch_end(self, outputs):

        # average validation loss
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(
            f"Average validation loss for Epoch: {self.trainer.current_epoch} is :",
            avg_loss.item(),
        )
        # print('Validation outputs: ', outputs)
        # save to progress bar to plot metrics
        self.val_loss = avg_loss
        tqdm_dict = {"val_loss": avg_loss}

        # save images (Should be the size of the batch?)
        in_ = torch.stack([x["input"] for x in outputs])
        in_ = in_.cpu().numpy()
        out_ = torch.stack([x["out"] for x in outputs])
        out_ = out_.cpu().numpy()
        targ_ = torch.stack([x["targ"] for x in outputs])
        targ_ = targ_.cpu().numpy()

        shape = out_.shape  # size of outputs

        if shape[0] != 1:
            # decided to only choose one image from one GPU
            a = np.arange(0, shape[0])
            a = np.random.choice(a)
        else:
            a = 0

        self.save_figs(in_[a], out_[a], targ_[a], name="val")

        mean_dices = []
        mean_hauss = []
        step = self.trainer.global_step

        for i in range(self.hparams.n_classes + 1):
            mean_dices.append(torch.stack([x[f"val_dice_{i}"] for x in outputs]))
            mean_hauss.append(torch.stack([x[f"val_haus_{i}"] for x in outputs]))
        for i, dice in enumerate(mean_dices):
            tqdm_dict[f"val_dice_{i}"] = dice.mean()
            # .cpu().numpy()
        log_dic = tqdm_dict.copy()
        for i, haus in enumerate(mean_hauss):
            log_dic[f"val_haus_{i}"] = haus.mean()  # .cpu().numpy()

        dices = torch.stack(mean_dices).cpu().numpy()
        hauss = torch.stack(mean_hauss).cpu().numpy()

        if self.hparams.backend != "ddp":
            shape = dices[1].shape
            #used to be dices[1] for 'dp'
            # print('dices', dices)
            print('shape of dices tensor:', shape)
            factor = shape[0] * shape[1]
        else:
            print('Shape of output dice tensor:', dices.shape)

        # changed in new version to support multiple and/or binary...
        if self.hparams.backend != "ddp":
            dice_data = [dice.reshape(1, factor)[0] for i, dice in enumerate(dices)]
            hu_data = [hu.reshape(1, factor)[0] for i, hu in enumerate(hauss)]
        else:
            dice_data = [dice for i, dice in enumerate(dices)]
            hu_data = [haus for i, haus in enumerate(hauss)]

        rois = ["GTV", "BRAIN","BSTEM","SPCOR","ESOPH","LARYNX","MAND",
            "POSTCRI","LPAR","RPAR","LACOU","RACOU","LLAC","RLAC","RRETRO",
            "LRETRO","RPLEX","LPLEX","LLENS","RLENS","LEYE","REYE","LOPTIC",
            "ROPTIC","LSMAN","RSMAN","CHIASM","LIPS","OCAV","IPCM","SPCM",
            "MPCM"]

        if self.hparams.oar_version == 16:
            idxs = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]
        elif self.hparams.oar_version == 1:
            idxs = [1]
        elif self.hparams.oar_version == 19:
            idxs = [3,4,5,6,7,9,10,11,12,17,18,19,20,21,22,23,24,27,28]

        chosen = [rois[i - 1] for i in idxs]

        plt.switch_backend("agg")
        fig1 = plt.figure()
        if self.hparams.oar_version == 1:
            plt.title("VAL DICE Overlap for OAR Segmentation in HNSCC")
        else:
            plt.title("VAL DICE Overlap for GTV Segmentation in HNSCC")
        plt.boxplot(dice_data[1:])
        plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
        plt.ylabel("DSC")
        plt.xlabel("Target Volume")
        self.logger.experiment.add_figure(f"val/dice_plot", fig1, step)

        # for now we need the last 3 classes excluding background
        fig2 = plt.figure()
        if self.hparams.oar_version == 1:
            plt.title("Valid HU for GTV Segmentation in HNSCC (mm)")
        else:
            plt.title("Valid HU for OAR Segmentation in HNSCC (mm)")
        plt.boxplot(hu_data[1:])
        plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
        plt.ylabel("d(mm)")
        plt.xlabel("Target Volume")
        self.logger.experiment.add_figure(f"val/hauss_plot", fig2, step)
        plt.close()

        assert len(mean_dices) == self.hparams.n_classes + 1

        for i, dice in enumerate(mean_dices):
            print(
                f"\n Max Val Dice for class {i} is {dice.mean()}"
            ) if self.hparams.verbose else None
            print(
                f"\n Max Val HausDist for class {i} is {mean_hauss[i].mean()}"
            ) if self.hparams.verbose else None

        print(f"BOOYA - End of Validation for {self.trainer.current_epoch}.")

        return OrderedDict( {"val_loss": avg_loss, "progress_bar": tqdm_dict, "log": log_dic})

    def test_step(self, batch, batch_idx):
         """
         Lightning calls this inside the validation loop
         :param batch:
         :return:
         """
         self.step_type = "test"
         # inpoot == regular (unwindowed) image...
         inputs, targets = batch
         if batch_idx == 0:
             print(inputs.max())
             print(targets.max())
         if inputs.shape != targets.shape:
             warnings.warn("Input Shape Not Same size as label...")
         # change this to incorporate sliding window...
         # get uncropped image...

         to_crop = RandomCrop3D(
                   window=self.hparams.window,
                   mode="test",
                   factor=292,#self.hparams.crop_factor,
                   crop_as=self.hparams.crop_as)

         #####################
         #####################
         # only for dataset2_3
         warnings.warn(f'SIZE OF {inputs.size()}')
         # inputs = inputs[:,:,0:292,104:396] # center = [0, 104]
         # inputs = inputs.permute(0,3,1,2)
         # targets = targets.permute(0,3,1,2)
         #####################
         #####################

         in_ = inputs.cpu().numpy()
         og_shape = inputs.size()
         ###########################
         # pad 3rd to last dim if below 112 with MIN value of image
         a, diff = (None, None)
         if og_shape[1]<112:
             difference = 112 - og_shape[1]
             a = difference//2
             diff = difference-a
             pad_ = (0,0,0,0,a,diff)
             warnings.warn(f'Padding {inputs.size()} to 112')
             inputs = F.pad(inputs, pad_, "constant", inputs.min())
             warnings.warn(f'NEW size is {inputs.size()},')
             targets=inputs.clone()
             # og_shape[1] = 112
         ##########################
         img, targ, center = to_crop(inputs,targets,in_)
         # img_, targ_ = transform(img, targ)
         roi_size = (112, 176, 176)
         # (128, 192, 192) # (during training) (self.hparams.window*2, self.hparams.crop_factor, self.hparams.crop_factor)
         sw_batch_size = 1
         # self.hparams.batch_size
         batch = []
         centers = []
         shape = img.size()
         ##########################
         ##########################
         # assumes first and last eight of image are fluff
         if 180<=shape[1]:
             cropz = (shape[1]//12, shape[1]-shape[1]//12)
         elif 165 <= shape[1] < 180:
             diff = 180 - shape[1]
             cropz = (shape[1]//13, shape[1]//13+152-diff)
         else:
             cropz = (0, shape[1])

         img = img[:,cropz[0]:cropz[1]]
         shape = img.size()
         warnings.warn(f'First crop size is {img.size()},')
         ##########################
         ##########################

         # this is the infernece using built in MONAI...
         # to_crop = RandomCrop3D(
         #           window=self.hparams.window,
         #           mode="test",
         #           factor=200,#self.hparams.crop_factor,
         #           crop_as=self.hparams.crop_as)
         # outputs = swi(img, roi_size, sw_batch_size, self.forward, overlap=0.8, mode='gaussian') # from monai 0.4.0
         # hello is it me you're looking for?
         # img = img[:,cropz[0]:cropz[1]]
         outputs = swi(img, self.forward, 20)
         warnings.warn(f'Hello size is {outputs.size()},')
         # because we're passing the same image through, mean the outputs...
         # outputs = torch.mean(outputs, 0)
         if type(outputs) == tuple:
             outputs = outputs[0]
         if self.hparams.crop_as != "3D":
             outputs = outputs.squeeze(2)  # shouldn't this be conditional??
         # targ = targets.clone() #.cpu()
         warnings.warn(f'Hello size is {outputs.size()}')
         out = outputs.clone() #.cpu()
         outs = torch.softmax(out, dim=1)
         warnings.warn(f'Hello size is {outs.size()} AFTER SOFTMAX')
         # sum predictions after softmax BECAUSE SAME IMAGE...
         outs = torch.mean(outs, dim=0)
         outs_raw = outs.cpu().numpy()
         warnings.warn(f'Hello size is {outs.size()} AFTER SOFTMAX')
         outs = torch.argmax(outs, dim=0)
         inp = inputs[0]
         warnings.warn(f'OUTPUT size is {outs.size()} with inputs {inp.size()}')
         # assert outs.size() == inp.size()
         out_full = torch.zeros(inp.size())
         warnings.warn(f'Hello size is {inp.size()}')
         out_full[cropz[0]:cropz[1], center[0]:center[0]+292,center[1]:center[1]+292] = outs

         # FOR REGULAR RADCURE PATIENTS
         # path = f'/cluster/projects/radiomics/EXTERNAL/OAR-TESTING/FOLD_{self.hparams.fold}'
         # targ_fold = '/cluster/projects/radiomics/EXTERNAL/OAR-TESTING/RAW/'

         #############
         # For External Datasets
         # save outputs ...
         # path = self.hparams.root + '/RADCURE-FULL'# f'/FINAL_TEST_PDDCA-3' # '/Test_STRUCTSEG'
         # path = '/cluster/projects/radiomics/EXTERNAL/DeepMind/AI_'
         # path = '/cluster/projects/radiomics/EXTERNAL/dataset2_3/AI'
         # path = '/cluster/projects/radiomics/EXTERNAL/MASTRO/AI_2'
         # path = '/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_HNSCC-3DCT-RT2'
         # path = '/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_RADIOMICS_HN1'
         # path = "/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_TCIA_HNSCC"
         # path = "/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_STRUCTSEG_19"
         # path = '/cluster/projects/radiomics/EXTERNAL/PDDCA/AI_'
         # path = '/cluster/projects/radiomics/EXTERNAL/OAR-TESTING/AI_PDDCA_2'
         # path = '/cluster/projects/radiomics/EXTERNAL/STRUCTSEG19/HaN_OAR/AI_'
         ###############
        
         path = f"{self.root}wolnet-sample"
         path += f'/FOLD_{self.hparams.fold}'
         targ_fold = path + '/RAW/'
         idx=0
         os.makedirs(path, exist_ok=True)
         os.makedirs(targ_fold, exist_ok=True)
         targ_path = targ_fold + f'targ_{batch_idx+idx}_FULL.nrrd'
         img_path = targ_fold + f'input_{batch_idx+idx}_FULL.nrrd'

         if os.path.isfile(targ_path) is True:
             pass
         else:
             in_ = inp.cpu().numpy()
             targ_ = targets.cpu().numpy()
             nrrd.write(img_path, in_)
             nrrd.write(targ_path, targ_[0].astype('uint8'), compression_level=9)

         # save FULL outputs...
         outs_ = out_full.cpu().numpy()
         warnings.warn(f'Max pred is {out_full.max()}')
         nrrd.write(f"{path}/outs_{batch_idx+idx}_RAW.nrrd", outs_raw)
         nrrd.write(f"{path}/outs_{batch_idx+idx}_FULL.nrrd", outs_.astype('uint8'), compression_level=9)
         np.save( f"{path}/center_{batch_idx+idx}.npy", np.array([cropz[0], cropz[1], center[0], center[1]]))

        #####################################
        # USE TO AVERAGE PREDICTIONS FROM ENSEMBLE
        # import torch, os, glob, nrrd
        # import numpy as np
        # os.mkdir('FINAL_')
        # folders=glob.glob('./FOLD_*')
        # folders.sort()
        # imgs = glob.glob(folders[0]+'/*_RAW*')
        # for b in range(len(imgs)):
        #     im = None
        #     for i, fold in enumerate(folders):
        #         # img_ = torch.tensor(np.load(fold+f'/outs_{b}_RAW.npy'))
        #         img_=torch.tensor(nrrd.read(fold+f'/outs_{b}_RAW.nrrd')[0])
        #         if im is None:
        #             im = img_
        #         else:
        #             im = torch.stack([im, img_])
        #             im = torch.mean(im, dim=0)
        #         print(f'LOADED {i}')
        #     print(im.size())
        #     im = torch.argmax(im, dim=0)
        #     print(im.size()) # correlate to original input size
        #     center = np.load(fold+f'/center_{b}.npy')
        #     # orig = np.load('./RAW/'+f'/input_{b}_FULL.npy')
        #     orig = nrrd.read('./RAW/'+f'/input_{b}_FULL.nrrd')[0]
        #     orig = torch.zeros(orig.shape)
        #     orig[center[0]:center[1], center[2]:center[2]+292,center[3]:center[3]+292] = im
        #     orig = orig.cpu().numpy() # np.save(f'./FINAL_/outs_{b}_FULL.npy', orig)
        #     nrrd.write(f'./FINAL_/outs_{b}_FULL.nrrd', orig)
        #     print(f'Done {b}')
        #######################################

    def configure_optimizers(self):

        ada = ['ADABOUND', 'AMSBOUND']

        # for retraining wolnet...
        self.hparams.lr = 0.001
        # self.hparams.decay_after = 12
        # self.hparams.gamma = 0.25

        if self.hparams.optim == "ADAM":
            # only if loading weights from presaved model...
            # self.hparams.lr = 0.001
            init_optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.decay)

            warnings.warn("Using ADAM as default optimizer.")

        elif self.hparams.optim == "RADAM":
            # the new RADAM optimizer
            # as defined in https://arxiv.org/pdf/1908.03265.pdf
            init_optimizer = RAdam(self.net.parameters(),
                                   lr=self.hparams.lr,)
                                   # weight_decay=self.hparams.decay)

        elif self.hparams.optim == "RMSPROP":
            init_optimizer = torch.optim.RMSprop(self.net.parameters(),
                                                 lr=self.hparams.lr)
        elif self.hparams.optim in ada:

            if self.hparams.optim == 'AMSBOUND':
                ams = True
            else:
                ams = False

            init_optimizer = AdaBoundW(self.net.parameters(),
                                      lr=self.hparams.lr, final_lr=0.001,
                                      weight_decay=self.hparams.decay,
                                      amsbound=ams)
        else:

            init_optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.decay,
                nesterov=True,
            )
            warnings.warn("Using SGD as default optimizer.")

        if self.hparams.scheduler is True:

            if self.hparams.scheduler_type == 'plateau':
                scheduler = ReduceLROnPlateau(init_optimizer, factor=self.hparams.gamma,
                                              patience=self.hparams.decay_after,
                                              threshold=0.0001)
            else:
                scheduler = StepLR(
                    init_optimizer,
                    step_size=self.hparams.decay_after,
                    gamma=self.hparams.gamma,
                )

            return [init_optimizer], [scheduler]

        else:
            return [init_optimizer]

    # function hook in LightningModule
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure,
    #                    on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #   # `optimizer is a ``LightningOptimizer`` wrapping the optimizer.
    #   # To access it, do as follow:
    #   # optimizer = optimizer.optimizer
    #   for pg in optimizer.param_groups:
    #       self.lr = pg["lr"]
    #   self.log("lr", self.lr, on_epoch=True, prog_bar=True, logger=True)
    #   # run step. However, it won't work on TPU, AMP, etc...
    #   optimizer.step(closure=closure)

    def get_dataloader( self, df, mode="valid", transform=None, resample=None,
                        shuffle=False, transform2=None, batch_size=None):

        # list of 3D models
        models = [
            "UNET",
            "VGG163D",
            "WOLNET",
            "RESUNET",
            "VNET",
            "MODIFIED3DUNET",
            "TIRAMISU",
            "TERANUS",
            "ELEKTRONUNET",
            "ANATOMY",
            "PIPOFAN",
            "DEEPLABV3",
            "HIGHRESNET",
            "UNET3+",
            "UNET3+DEEPSUP",
            "UNET++",
            "RSANET",
            "HYPERDENSENET",
            "DENSEVOX",
            "MEDNET",
            "SKIPDNET"
        ]

        if self.hparams.model not in models:

            dataset = LoadPatientSlices(
                df=df,
                transform=transform,
                window=self.hparams.window,
                root=self.root,
                resample=resample,
                factor=self.hparams.crop_factor,
            )

            warnings.warn(
                f"Loading Dataset for 3D slices. Using Model {self.hparams.model}."
            )

        else:

            img_path='/cluster/projects/radiomics/Temp/joe/OAR0720/img'
            mask_path='/cluster/projects/radiomics/Temp/joe/OAR0720/masks'

            # default volume type is targets...
            # nothing should change in the pipeline
            dataset = LoadPatientVolumes(
                df=df,
                transform=transform,
                transform2=transform2,
                window=self.hparams.window, # ,
                root=self.root,
                mode=mode,
                to_filter=self.hparams.filter,
                spacing=self.hparams.spacing,
                volume_type=self.hparams.volume_type,
                oar_version=self.hparams.oar_version,
                mask_path=mask_path, #self.hparams.mask_path,
                img_path=img_path, #self.hparams.img_path,
                resample=resample,
                external=True, # True, would be better to define dataset ie. PDDCA OR STRUCTSEG
            )

        train_sampler = None

        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        # best practices to turn shuffling off during validation...
        validate = ['valid', 'test']

        if mode in validate:
            shuffle=False
        else:
            shuffle = self.hparams.shuffle_data

        return DataLoader(
            dataset=dataset,
            num_workers=self.hparams.workers,
            batch_size=batch_size,
            pin_memory=True,  # comment this out iff running on CPU
            # sampler=train_sampler,
            shuffle=shuffle,
            drop_last=True,
        )

    def train_dataloader(self):

        transform = Compose(
            [
                HistogramClipping(
                    min_hu=self.hparams.clip_min, max_hu=self.hparams.clip_max
                ),
                RandomFlip3D(), # left and right should be distinguished...
                RandomRotation3D(p=self.hparams.aug_prob/1.5),
                ElasticTransform3D(p=self.hparams.aug_prob/1.5),
                RandomZoom3D(p=self.hparams.aug_prob/1.5),
                RandomCrop3D(window=self.hparams.window,
                             mode="train",
                             factor=self.hparams.crop_factor,
                             crop_as=self.hparams.crop_as,
                            ),
                NormBabe(mean=self.mean, std=self.std, type=self.hparams.norm),
            ]
        )

        # add transform to dataloader
        return self.get_dataloader(
            df=self.train_data,
            mode="train",
            transform=transform,
            resample=False,
            batch_size=self.hparams.batch_size,
        )

    # @pl.data_loader
    def val_dataloader(self):
        # imported from transform.py
        transform = Compose(
            [
                HistogramClipping(
                    min_hu=self.hparams.clip_min, max_hu=self.hparams.clip_max
                ),
                RandomCrop3D(
                    window=self.hparams.window,
                    mode="valid",
                    factor=self.hparams.crop_factor,
                    crop_as=self.hparams.crop_as,
                ),
                NormBabe(mean=self.mean, std=self.std, type=self.hparams.norm),
            ]
        )

        return self.get_dataloader(
            df=self.valid_data,
            mode="valid",
            transform=transform,  # should be default
            resample=False,
            batch_size=self.hparams.batch_size,
            # first model was trained and will be validated with batch size of 1...
        )

    # @pl.data_loader
    def test_dataloader(self):
        # imported from transform.py
        # before histogram_clipping wasn't included as a transformation
        # that could've messed things up
        transform = Compose(
            [
                HistogramClipping(
                    min_hu=self.hparams.clip_min, max_hu=self.hparams.clip_max
                ),
                # RandomCrop3D(
                #     window=self.hparams.window,
                #     mode="valid",
                #     factor=200, # 232, # 192,
                #     crop_as=self.hparams.crop_as,
                # ),
                NormBabe(mean=self.mean, std=self.std, type=self.hparams.norm),
            ]
        )

        transform2 = Compose(
            [
                RandomCrop3D(
                    window=self.hparams.window,
                    mode="valid",
                    factor=200, # 232, #192,
                    crop_as=self.hparams.crop_as,
                    )
            ]
        )


        # # PDDCA ...
        # test = pd.read_csv('/cluster/projects/radiomics/Temp/models/WOLNET_2020_08_28_152828/test_PDDCA.csv')
        # print(f'New data..., {test.head()}')
        # test = pd.read_csv('/cluster/projects/radiomics/Temp/models/WOLNET_2020_07_15_152828/struct_hanoar.csv', index_col=0)
        # print(f'New data..., {test.head()}')
        # print(test.iloc[0])
        # print(test.iloc[0][0])
        # # ###########################
        # # # comment this out if just want to use COM ONLY...
        # test = pd.concat([test, test, test])
        # vals = []
        # count = 0
        # for i in range(len(test)):
        #     if i%50 == 0:
        #         count += 1
        #     vals.append(count)
        #
        # test['version'] = vals
        # # ###########################

        # Deepmind...
        self.hparams.external = True
        # self.test = pd.read_csv('/cluster/home/jmarsill/deep_.csv', index_col=0)
        # save directory /cluster/projects/radiomics/EXTERNAL/DeepMind/AI
        # UANET
        # self.test = pd.read_csv('/cluster/home/jmarsill/dataset23.csv', index_col=0)
        # save directory /cluster/projects/radiomics/EXTERNAL/dataset2_3/AI
        # MASTRO - RADIOMICS HN1
        # self.test = pd.read_csv('/cluster/home/jmarsill/mastro_.csv', index_col=0)
        # self.test = self.test[264:]
        # self.test = pd.read_csv('/cluster/home/jmarsill/head_neck_radiomics_hn1.csv', index_col=0)
        # self.test=pd.read_csv('/cluster/home/jmarsill/tcia_hnscc.csv', index_col=0)
        # self.test = self.test[946:]
        # self.test=pd.read_csv('/cluster/home/jmarsill/dataset23.csv', index_col=0)
        # self.test=pd.read_csv('/cluster/home/jmarsill/structseg.csv', index_col=0)
        # save to /cluster/projects/radiomics/EXTERNAL/MASTRO/AI
        # MANIFEST
        # old .csv
        # self.test = pd.read_csv('/cluster/home/jmarsill/manifest_updated_ref.csv', index_col=0)
        # new .csv
        # self.test = self.test[94:]
        # /cluster/projects/radiomics/EXTERNAL/manifest-1549495779734/AI
        # QUEBEC
        # self.test = pd.read_csv('/cluster/home/jmarsill/mastro.csv', index_col=0)
        # /cluster/projects/radiomics/EXTERNAL/Head-Neck-PET-CT/AI
        # PDDCA
        # self.test = pd.read_csv('/cluster/home/jmarsill/pddca_.csv', index_col=0)
        # save dir /cluster/projects/radiomics/EXTERNAL/PDDCA/AI
        # HaN OAR
        # self.test = pd.read_csv('/cluster/home/jmarsill/structseg_.csv', index_col=0)
        # run because need batch size of 2 .ie need to reload same image twice...
        # FOR
        # self.test = pd.read_csv('/cluster/projects/radiomics/Temp/models/WOLNET_2020_08_28_152828/test_fold_0_edited.csv', index_col=0)
        # FOR GTV TESTING...
        # self.test = pd.read_csv('/cluster/projects/radiomics/Temp/models/WOLNET_2021_02_22_200100/test_fold_1_edited.csv', index_col=0)
        # do a we bit of modifications...
        # test editing...
        # tr = list(self.test['0'])
        # tr_ = ['0'+str(i) for i in tr if len(str(i)) < 7]
        # tr_1 = ['0'+str(i) for i in tr_ if len(str(i)) < 7]
        # tr_ = [str(i) for i in tr_ if len(str(i)) == 7]
        # tr_ += [str(i) for i in tr if len(str(i)) == 7]
        # tr_ += tr_1
        # # random.shuffle(tr_)
        # self.test['0'] = tr_
        # print(self.test.head())
        # self.test = 
        return self.get_dataloader(
            df=self.test,# test, self.test[self.test['version']==1]
            mode="test",
            transform=transform, # transform,  # should be default
            transform2=None, # transform2,
            resample=self.hparams.resample,
            batch_size=self.hparams.batch_size,
        )

    def on_save_checkpoint(self, checkpoint):
        # 99% of use cases you don't need to implement this method
        checkpoint["Kfold"] = self.Kfold
        checkpoint["train_data"] = self.train_data
        checkpoint["valid_data"] = self.valid_data
        checkpoint["test_data"] = self.test_data
        checkpoint["mean"] = self.mean
        checkpoint["std"] = self.std

    def on_load_checkpoint(self, checkpoint):
        # 99% of the time you don't need to implement this method
        if self.hparams.testing is False:
            self.Kfold = checkpoint["Kfold"]
            self.train_data = checkpoint["train_data"]
            self.valid_data = checkpoint["valid_data"]
            # self.test_data = checkpoint["test_data"]
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]

    @staticmethod
    def load_obj(name):
        with open(name, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_obj(obj, name):
        with open(name, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def get_model_root_dir(self):

        self.ts = time.time()
        self.date = datetime.datetime.fromtimestamp(self.ts).strftime("%Y_%m_%d_%H%M%S")
        self.model_path = self.hparams.model_path

        model_names = [
            "DEEPNET",
            "SIMPLE",
            "UNET",
            "WOLNET",
            "RESUNET",
            "ATTENTION3D",
            "VNET",
            "MODIFIED3DUNET",
            "TIRAMISU",
            "TERANUS",
            "ELEKTRONUNET",
            "ANATOMY",
            "DEEPLABV3",
            "PIPOFAN",
            "HIGHRESNET",
            "UNET3+",
            "UNET3+DEEPSUP",
            "UNET++",
            "VGG163D",
            "RSANET",
            "HYPERDENSENET",
            "DENSEVOX",
            "MEDNET",
            "SKIPDNET"
        ]

        if self.hparams.model in model_names:
            if self.hparams.model_name is None:
                self.model_name = f"{self.hparams.model}_{self.date}"
                self.hparams.model_name = self.model_name
            else:
                self.model_name = self.hparams.model_name

        self.hparams.root = str(self.model_path + "/" + self.model_name)
        self.root = Path(self.hparams.root)
        self.root.mkdir(exist_ok=True, parents=True)

#################
### RANDOM STUFF
#################

            # Taken from get_data for OAR
            # extract dataframe to be used for training/validation...
            # only set true if actually testing...
            # testing = self.hparams.testing  # True  #  # originally False
            # if self.hparams.site is not None:
            #     site = self.hparams.site
            #     if testing is True:
            #         site = "ALL"  # or specific subsite
            # else:
            #     site = "Oropharynx"
            #
            # print("Resampled data? ", self.hparams.filter)
            # if testing is True:
            #     dataset_path = (str(self.root) + "/" + f"Kfold_{self.model_name}_{site}" + ".pkl")
            #     print(dataset_path)
            # else:

        # ideally self.hparams.external_test
        # self.external_test = True
        #
        # if self.external_test is True:
        #     # do stuff here...
        #     # end format we need pandas dataframe...
        #     self.test_name = '_PDDCA'
        #     external_test_csv_path = str(self.root) + f"/test{self.test_name}.csv"
        #     # get images that are good...
        #     bad = self.load_obj('/cluster/home/jmarsill/SegmentHN/data/badpddca.pkl')
        #     bad_vals = list(bad.values())
        #     bad_vals = [b.split('/')[-1] for b in bad_vals]
        #
        #     masks =  glob.glob('/cluster/projects/radiomics/EXTERNAL/dataset2_3/raw_masks/*.nrrd')
        #     masks = [msk for msk in masks if msk.split('/')[-1].split('.')[0] not in bad_vals]
        #     patients = [pat.split('/')[-1].split('.')[0] for pat in masks]
        #     imgs = [f'/cluster/projects/radiomics/EXTERNAL/dataset2_3/raw/{pat}/img.nrrd' for pat in patients if pat not in bad_vals]
        #
        #     print('PDDCA', imgs, len(imgs))
        #     # get masks
        #     masks = [ glob.glob(f'/cluster/projects/radiomics/EXTERNAL/dataset2_3/preprocessed/{p}*').sort() for p in pats ]
        #
        #     # masks =  glob.glob('/cluster/projects/radiomics/EXTERNAL/dataset2_3/raw_masks/*')
        #     # patients = [pat.split('/')[-1].split('.')[0] for pat in masks]
        #     # imgs = [f'/cluster/projects/radiomics/EXTERNAL/dataset2_3/raw/{pat}/img.nrrd' for pat in patients]
        #     # define external dataset to use...
        #     # iterate through each mask, if mask doesn't exist we need to make it...
        #
        #     dic = {}
        #     for i, msk in enumerate(masks):
        #         dic[i] = [imgs[i], msk, patients[i]]
        #         if i == 0:
        #             print(dic[i])
        #
        #     self.test_data = pd.DataFrame.from_dict(dic, orient='index')
        #     print(self.test_data.head())
        #     print(self.test_data.iloc[0][1])
        #     self.test_data.to_csv(external_test_csv_path)
        #
        # else:
        #     self.test_data = test_fold.dataset()
        #     self.test_name = '_RADCURE'

         ##################

         # else:
         #     # original loss for segmentation
         #     loss = self.criterion(outputs, targets)
         #     dices, hauss = getMetrics(outputs, targets)

         # grab images to record...
         # inpoot = inpoot#.cpu().numpy()
         # targ = targ.cpu().numpy()
         # outs = outs.cpu().detach().numpy()
         # in_, out, targ = self.export_figs(inpoot, outs, targ)

         # take mean dice of batch
         # mean_dices = np.mean(np.array(dices), axis=0)
         # dice_dic = {f"test_dice_{i}": dice for i, dice in enumerate(dices)}
         # take mean hausdorff distance of batch
         # hu_dic = {f"test_haus_{i}": hu for i, hu in enumerate(hauss)}

         # loss = loss.unsqueeze(0)
         #
         # output = OrderedDict(
         #     {
         #         "test_loss": loss,
         #     }
         # )
         #
         # for i, dice in enumerate(dices):
         #     output[f"test_dice_{i}"] = torch.tensor(dice)
         #     output[f"test_haus_{i}"] = torch.tensor(hauss[i])
         #
         # return output

    # def test_epoch_end(self, outputs):
    #
    #     # save actual images...
    #     # version = "OAR072220"
    #     path = self.hparams.root + 'FINAL_TEST_STRUCT21' #f'/FINAL_TEST_WOL022221_4' # '/Test_STRUCTSEG'
    #     # set directory to save test images...
    #     # os.makedirs(f"/cluster/projects/radiomics/Temp/TEST_{version}/", exist_ok=True)
    #     os.makedirs(path, exist_ok=True)
        # # average validation loss
        # avg_loss = torch.stack([x["test_loss"] for x in outputs])
        # np.save(
        #     f"{path}/losses_{self.trainer.proc_rank}.npy",
        #     avg_loss.cpu().numpy(),
        # )
        # avg_loss = avg_loss.mean()
        # print(
        #     f"Average test loss for Epoch: {self.trainer.current_epoch} is :",
        #     avg_loss.item(),
        # )
        # step = self.trainer.global_step
        # # save to progress bar to plot metrics
        # self.test_loss = avg_loss
        #
        # tqdm_dict = {"test_loss": avg_loss}
        #
        # mean_dices = []
        # mean_hauss = []
        #
        # for i in range(self.hparams.n_classes + 1):
        #     mean_dices.append(torch.stack([x[f"test_dice_{i}"] for x in outputs]))
        #     mean_hauss.append(torch.stack([x[f"test_haus_{i}"] for x in outputs]))
        #
        # dices = torch.stack(mean_dices).cpu().numpy()
        # hauss = torch.stack(mean_hauss).cpu().numpy()
        #
        # np.save(
        #     f"{path}/dices_{self.trainer.proc_rank}.npy",
        #     dices,
        # )
        # np.save(
        #     f"{path}/hauss_{self.trainer.proc_rank}.npy",
        #     hauss,
        # )
        #
        # assert len(mean_dices) == self.hparams.n_classes + 1
        #
        # for i, dice in enumerate(mean_dices):
        #     print(
        #         f"\n Max Test Dice for class {i} is {dice.mean()}"
        #     ) if self.hparams.verbose else None
        #     print(
        #         f"\n Max Test HausDist for class {i} is {mean_hauss[i].mean()}"
        #     ) if self.hparams.verbose else None
        #     tqdm_dict[f"test_dice_{i}"] = dice.mean()
        #
        # log_dic = tqdm_dict.copy()
        # for i, haus in enumerate(mean_hauss):
        #     log_dic[f"test_haus_{i}"] = haus.mean()  # .cpu().numpy()
        #
        # # save images (Should be the size of the batch?)
        # # in_ = torch.stack([x["input"] for x in outputs])
        # # in_ = in_.cpu().numpy()
        # # out = torch.stack([x["out"] for x in outputs])
        # # out = out.cpu().numpy()
        # # targ = torch.stack([x["targ"] for x in outputs])
        # # targ = targ.cpu().numpy()
        # #
        # # np.save(
        # #     f"{path}/in_{self.trainer.proc_rank}.npy",
        # #     in_,
        # # )
        # # np.save(
        # #     f"{path}/out_{self.trainer.proc_rank}.npy",
        # #     out,
        # # )
        # # np.save(
        # #     f"{path}/targ_{self.trainer.proc_rank}.npy",
        # #     targ,
        # # )
        #
        # backend = ['ddp', 'ddp_spawn']
        #
        # if self.hparams.backend not in backend:
        #     shape = dices[1].shape
        #     #used to be dices[1] for 'dp'
        #     # print('dices', dices)
        #     print('shape of dices tensor:', shape)
        #     factor = shape[0] * shape[1]
        # else:
        #     print('Shape of output dice tensor:', dices.shape)
        #
        # # changed in new version to support multiple and/or binary...
        # if self.hparams.backend not in backend:
        #     dice_data = [dice.reshape(1, factor)[0] for i, dice in enumerate(dices)]
        #     hu_data = [hu.reshape(1, factor)[0] for i, hu in enumerate(hauss)]
        # else:
        #     dice_data = [dice for i, dice in enumerate(dices)]
        #     hu_data = [haus for i, haus in enumerate(hauss)]
        #
        # rois = ["GTV","BRAIN","BSTEM","SPCOR","ESOPH","LARYNX","MAND",
        #     "POSTCRI","LPAR","RPAR","LACOU","RACOU","LLAC","RLAC","RRETRO",
        #     "LRETRO","RPLEX","LPLEX","LLENS","RLENS","LEYE","REYE","LOPTIC",
        #     "ROPTIC","LSMAN","RSMAN","CHIASM","LIPS","OCAV","IPCM","SPCM",
        #     "MPCM"]
        #
        # if self.hparams.oar_version == 16:
        #     idxs = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]
        # elif self.hparams.oar_version == 1:
        #     idxs = [1]
        # elif self.hparams.oar_version == 19:
        #     idxs = [3,4,5,6,7,9,10,11,12,17,18,19,20,21,22,23,24,27,28]
        #
        # chosen = [rois[i - 1] for i in idxs]
        #
        # plt.switch_backend("agg")
        # fig1 = plt.figure()
        # if self.hparams.oar_version == 1:
        #     plt.title("Test DICE Overlap for GTV Segmentation in HNSCC")
        # else:
        #     plt.title("Test DICE Overlap for OAR Segmentation in HNSCC")
        # plt.boxplot(dice_data[1:])
        # plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
        # plt.ylabel("DSC")
        # plt.xlabel("Target Volume")
        # self.logger.experiment.add_figure(f"test/dice_plot", fig1, step)
        #
        # # for now we need the last 3 classes excluding background
        # fig2 = plt.figure()
        # if self.hparams.oar_version == 1:
        #     plt.title("Test HD for GTV Segmentation in HNSCC (mm)")
        # else:
        #     plt.title("Test HD for OAR Segmentation in HNSCC (mm)")
        # plt.boxplot(hu_data[1:])
        # plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
        # plt.ylabel("d(mm)")
        # plt.xlabel("Target Volume")
        # self.logger.experiment.add_figure(f"test/hauss_plot", fig2, step)
        # plt.close()
        #
        # assert len(mean_dices) == self.hparams.n_classes + 1
        #
        # for i, dice in enumerate(mean_dices):
        #     print(
        #         f"\n Max Val Dice for class {i} is {dice.mean()}"
        #     ) if self.hparams.verbose else None
        #     print(
        #         f"\n Max Val HausDist for class {i} is {mean_hauss[i].mean()}"
        #     ) if self.hparams.verbose else None
        #
        # print(f"BOOYA - End of Test for {self.trainer.current_epoch}.")
        #
        # return OrderedDict(
        #     {"loss": avg_loss, "progress_bar": tqdm_dict, "log": log_dic}
        # )

###############

        # if self.hparams.backend != "ddp":
        #     shape = dices[1].shape
        #     #used to be dices[1] for 'dp'
        #     # print('dices', dices)
        #     print('shape of dices tensor:', shape)
        #     factor = shape[0] * shape[1]
        # else:
        #     print('Shape of output dice tensor:', dices.shape)
        #
        # # changed in new version to support multiple and/or binary...
        # if self.hparams.backend != "ddp":
        #     dice_data = [dice.reshape(1, factor)[0] for i, dice in enumerate(dices)]
        #     hu_data = [hu.reshape(1, factor)[0] for i, hu in enumerate(hauss)]
        # else:
        #     dice_data = [dice for i, dice in enumerate(dices)]
        #     hu_data = [haus for i, haus in enumerate(hauss)]
        #
        # rois = ["GTV","BRAIN","BSTEM","SPCOR","ESOPH","LARYNX","MAND",
        #     "POSTCRI","LPAR","RPAR","LACOU","RACOU","LLAC","RLAC","RRETRO",
        #     "LRETRO","RPLEX","LPLEX","LLENS","RLENS","LEYE","REYE","LOPTIC",
        #     "ROPTIC","LSMAN","RSMAN","CHIASM","LIPS","OCAV","IPCM","SPCM",
        #     "MPCM"]
        #
        # if self.hparams.oar_version == 16:
        #     idxs = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]
        # elif self.hparams.oar_version == 1:
        #     idxs = [1]
        # elif self.hparams.oar_version == 19:
        #     idxs = [3,4,5,6,7,9,10,11,12,17,18,19,20,21,22,23,24,27,28]
        #
        # chosen = [rois[i - 1] for i in idxs]
        #
        # plt.switch_backend("agg")
        # fig1 = plt.figure()
        # if self.hparams.oar_version == 1:
        #     plt.title("VAL DICE Overlap for OAR Segmentation in HNSCC")
        # else:
        #     plt.title("VAL DICE Overlap for GTV Segmentation in HNSCC")
        # plt.boxplot(dice_data[1:])
        # plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
        # plt.ylabel("DSC")
        # plt.xlabel("Target Volume")
        # self.logger.experiment.add_figure(f"val/dice_plot", fig1, step)
        #
        # # for now we need the last 3 classes excluding background
        # fig2 = plt.figure()
        # if self.hparams.oar_version == 1:
        #     plt.title("Valid HU for GTV Segmentation in HNSCC (mm)")
        # else:
        #     plt.title("Valid HU for OAR Segmentation in HNSCC (mm)")
        # plt.boxplot(hu_data[1:])
        # plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
        # plt.ylabel("d(mm)")
        # plt.xlabel("Target Volume")
        # self.logger.experiment.add_figure(f"val/hauss_plot", fig2, step)
        # plt.close()
        #
        # assert len(mean_dices) == self.hparams.n_classes + 1
        #
        # for i, dice in enumerate(mean_dices):
        #     print(
        #         f"\n Max Val Dice for class {i} is {dice.mean()}"
        #     ) if self.hparams.verbose else None
        #     print(
        #         f"\n Max Val HausDist for class {i} is {mean_hauss[i].mean()}"
        #     ) if self.hparams.verbose else None
        #
        # print(f"BOOYA - End of Validation for {self.trainer.current_epoch}.")
        #
        # return OrderedDict(
        #     {"loss": avg_loss, "progress_bar": tqdm_dict, "log": log_dic}
        # )

    # output = OrderedDict({'loss': avg_loss})
    # for i, dice in enumerate(mean_dices):
    #     output[f'val_dice{i}'] = dice
    #     output[f'val_haus{i}'] = mean_hauss[i]
    # self.val_step[self.trainer.proc_rank] += 1
    # self.validation_dice[self.trainer.proc_rank] += torch.stack(mean_dices)
    # self.validation_haus[self.trainer.proc_rank] += torch.stack(mean_hauss)
    # print(f'AT VAL END - saving figs for {self.trainer.proc_rank}.')
    # self.save_figs(in_[a], out_[a], targ_[a], name=f'val')
    # return output
    # else:
    #     # this is the right way for dp, 2.5 implementation...
    #     # activate when 3D is set to false...
    #     # average validation dice
    #     avg_dice = [np.array(list(x['valid_dices'].values())) for x in outputs]
    #     mean_dices = np.mean(np.stack(avg_dice), axis=0)
    #     # average validation hausendorf distance...
    #     avg_hauss = [np.array(list(x['valid_hausdorff'].values())) for x in outputs]
    #     mean_hauss = np.mean(np.stack(avg_hauss), axis=0)
    # print result to output files...
    # def on_epoch_end(self):
    #
    #     if self.hparams.backend == 'ddp':
    #
    #         # save training metrics...
    #         if torch.mean(self.tr_step) == 0:
    #             step_ = 1.
    #         else:
    #             step_ = self.tr_step
    #
    #         # saving validation metrics...
    #         if torch.mean(self.val_step) == 0:
    #             vstep_ = 1.
    #         else:
    #             vstep_ = self.val_step
    #
    #         # average training metrics on the gpu dimension...
    #         avg_train_dice = torch.mean(self.training_dice / step_, dim=0).detach()
    #         avg_train_haus = torch.mean(self.training_haus / step_, dim=0).detach()
    #         print('SAVING AVG TRAIN METRICS:')
    #         assert len(avg_train_haus) == self.hparams.n_classes + 1 # self.hparams.gpus
    #
    #         # average validation metrics on gpu dimension
    #         avg_val_dice = torch.mean(self.validation_dice / vstep_, dim=0).detach()
    #         avg_val_haus = torch.mean(self.validation_haus / vstep_, dim=0).detach()
    #
    #         # log average training metrics at the epoch level...
    #         # cleaner to plot everything on the same graph...
    #         traind = {}
    #         trainh = {}
    #         vald = {}
    #         valh = {}
    #
    #         print('BABY')
    #
    #         for i, d in enumerate(avg_train_dice):
    #             traind[f'train_dice_{i}'] = d
    #             trainh[f'train_haus_{i}'] = avg_train_haus[i]
    #             vald[f'val_dice_{i}'] = avg_val_dice[i]
    #             valh[f'val_haus_{i}'] = avg_val_haus[i]
    #
    #         self.logger.experiment.add_scalars('train', traind, self.trainer.current_epoch)
    #         self.logger.experiment.add_scalars('train', trainh, self.trainer.current_epoch)
    #         self.logger.experiment.add_scalars('val', vald, self.trainer.current_epoch)
    #         self.logger.experiment.add_scalars('val', valh, self.trainer.current_epoch)

    # def __get_data(self):
    #     self.class_weights = [0.1, 1.3, 5.5]
    #     self.Kfold = None
    #     self.val_loss = torch.tensor([0], dtype=torch.float)
    #     fold = self.hparams.fold
    #     pkl_name = self.hparams.pkl_name
    #     # if self.hparams.oar_version == 1:
    #     dataset_path = str(self.root) + "/" + f"Kfold_{self.model_name}" + ".pkl"
    #     if os.path.isfile(dataset_path):
    #         # resume from previous crash
    #         print(f"\n Load folds from: {dataset_path}")
    #         self.Kfold = self.load_obj(dataset_path)
    #     else:
    #         # can put in .csv of folders to be used in training
    #         # make dictionary from Data splits...
    #         patient_splits = GetSplits(
    #             args=self.hparams,
    #             mask_path=self.hparams.mask_path,
    #             metrics_name=self.hparams.metrics_name,
    #             mrn_path=self.hparams.mrn_csv_path,
    #             volume_type=self.hparams.volume_type,
    #             site='ALL',
    #             mode=self.hparams.split_mode,
    #             test_split=self.hparams.tt_split,
    #             to_filter=self.hparams.filter,
    #             height=self.hparams.window * 2,
    #             width=self.hparams.crop_factor,
    #             classes=self.hparams.n_classes + 1,
    #             dir_path=self.hparams.dir_path,
    #         )

    #         print(
    #             f"\n There are {len(patient_splits)} training folds."
    #         ) if self.hparams.verbose else None
    #         print(f"Saving folds at {dataset_path}.")
    #         self.Kfold = {}
    #         # ensure given as string ... (for effective parsing)
    #         self.Kfold["train"] = patient_splits.train
    #         self.Kfold["valid"] = patient_splits.valid
    #         self.Kfold["test"] = patient_splits.test
    #         self.Kfold["means"] = patient_splits.means
    #         self.Kfold["stds"] = patient_splits.stds
    #         self.Kfold["weights"] = patient_splits.weights
    #         # save kfolds
    #         self.save_obj(self.Kfold, dataset_path)
    #         testing=True
    #         if testing:
    #             self.root.joinpath(str(self.root) + f"/folds_{site}.json").write_text(json.dumps(Kfold))
    #         else:
    #             self.root.joinpath(str(self.root) + f"/folds.json").write_text(json.dumps(Kfold))

    #     print('Loading OAR data.')
    #     # set mean/std to normalize data
    #     # these are the standard values for clipped image from -500 to 1000
    #     if self.hparams.clip_max < 300:
    #         self.mean = self.Kfold["means"][fold] # - 300.
    #         self.std =  self.Kfold["stds"][fold] # + 75
    #     else:
    #         self.mean = -407.4462155135238 # Kfold["means"][fold] # - 300.
    #         self.std = 226.03663728492648 # Kfold["stds"][fold] # + 75
    #     # should return a numpy array of class weights n_classes + 1
    #     self.class_weights = self.Kfold["weights"][fold]
    #     print( f"Training Set Mean HU: {self.mean} Mean STD: {self.std} \n") if self.hparams.verbose else None
    #     print( f"Using this weight array to mitigate class imbalance {self.class_weights} \n") if self.hparams.verbose else None

    #     # if paramater files from old training sessions exist - load them...
    #     param_path = str(self.root) + f"/tensoborad_logs_{fold}.json"

    #     print( "\n Number of Patient Scans in Training:", len(self.Kfold["train"][0]),
    #            "\n Patients for Validation:", len(self.Kfold["valid"][0]),
    #            "\n Patients for Testing", len(self.Kfold["test"])) if self.hparams.verbose else None

    #     # if self.testing is True:
    #     #     train_csv_path = str(self.root) + f"/train_fold_{fold}_{site}.csv"
    #     #     valid_csv_path = str(self.root) + f"/valid_fold_{fold}_{site}.csv"
    #     #     test_csv_path = str(self.root) + f"/test_fold_{fold}_{site}.csv"
    #     # else:
    #     train_csv_path = str(self.root) + f"/train_fold_{fold}.csv"
    #     valid_csv_path = str(self.root) + f"/valid_fold_{fold}.csv"
    #     test_csv_path = str(self.root) + f"/test_fold_{fold}.csv"

    #     train_fold = PatientData(file_name=pkl_name, nfold=fold, folds=self.Kfold["train"],
    #                              dir_path=self.hparams.dir_path)
    #     valid_fold = PatientData(file_name=pkl_name, nfold=fold, folds=self.Kfold["valid"],
    #                              dir_path=self.hparams.dir_path)
    #     test_fold = PatientData( file_name=pkl_name, nfold=fold, folds=self.Kfold["test"],
    #                              mode="single", dir_path=self.hparams.dir_path)

    #     # get mean & standard deviation for the dataset...
    #     print(f"Saving folds to {self.root}.") if self.hparams.verbose else None
    #     # save dataframes of CSV's to .csv
    #     self.train_data = train_fold.dataset()
    #     self.valid_data = valid_fold.dataset()
    #     self.test_data = test_fold.dataset()
    #     self.test = self.test_data.copy()
    #     print(train_fold.dataset())
    #     ###########################
    #     # comment this out if just want to use COM ONLY...
    #     # self.test = pd.concat([self.test, self.test, self.test])
    #     # vals = []
    #     # count = 0
    #     # for i in range(len(self.test)):
    #     #     if i%59 == 0:
    #     #         count += 1
    #     #     vals.append(count)
    #     #
    #     # self.test['version'] = vals
    #     ###########################
    #     self.test_name = '_RADCURE'
    #     # self.Kfold = Kfold
    #     # save each split acordingly...
    #     if os.path.isfile(train_csv_path) is False:
    #         self.train_data.to_csv(train_csv_path)
    #         self.valid_data.to_csv(valid_csv_path)
    #         self.test.to_csv(test_csv_path)
    #     print(self.train_data.iloc[0])
    #     print(self.test.head())
    #     print(self.test.iloc[0])
