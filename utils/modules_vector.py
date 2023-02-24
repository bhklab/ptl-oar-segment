import os, torch, time, datetime, warnings, pickle, json, glob, nrrd
from pathlib import Path
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch.nn.functional as F
# monai slifing window inference...
# from sliding_window import sliding_window_inference
# from .sliding_window import sliding_window_inference as swi
# from monai.inferers import sliding_window_inference as swi
from .scheduler import Poly
from .prepare_vector import *
from .metrics import getMetrics, CombinedLoss, SoftDiceLoss, AnatFocalDLoss
from .loss import *
from .optimizers import *
from .models import *
from .transform import *
from .utils import *

def cuda(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device)

def getJson(path):
    with open(path, 'r') as myfile:
        data=myfile.read()
    obj = json.loads(data)
    return obj

class SegmentationModule(pl.LightningModule):
    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        super().__init__()
        # init superclass
        # super(SegmentationModule, self).__init__()
        self.save_hyperparameters(hparams) # 1.3+ self.hparams
        self.get_model_root_dir()
        self.__build_model()
        self.__getCutomOrder()

    def setup(self, stage=None):

        '''
        Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).
        Setup expects a ‘stage’ arg which is used to separate logic for ‘fit’ and ‘test’.
        If you don’t mind loading all your datasets at once, you can set up a condition to allow for both ‘fit’ related setup and ‘test’ related setup to run whenever None is passed to stage.
        Note this runs across all GPUs and it is safe to make state assignments here.
        '''
        # should load in a configuration.json contaning information used to
        # pre-process images before training this should be of the following format
        # if you are making an ensemble with multiple folds, please nest it by fold number
        # ie. for fold 0 {0: {dataset_params_for_fold_1} }
        # {"weights": np.array() > type:len(n_classes) > specifically required to mitigate class imbalance in WeightedTopKCrossEntropy Loss},
        # "dataset_mean": np.float(), "dataset_std": np.float(), specifically used for Z-score normalization of images...
        # "clip_max": 1000 (Recommended), "clip_min": -500 (Recommended)}
        # NOTE: if windowing will be applied to images, mean/std of dataset must reflect the windowed version of the image
        # for testing the following can be used... this can be calculated on the fly ...
        # the voxel counts for each class are given in the meta header of each .nrrd in the structure folder...
        # the un-windowed meanHU and stdHU are saved in the meta header for each image...
        # for clipped image(s) from -500 to 1000; expect mean/std values to
        # fall within the following ranges... -390 < meanHU < -420; 205 < stdHU < 245

        path_ = self.hparams.root + f"/config_{self.tag}.json"
        # this excludes all the data with three contours in their files or less...
        exclude_ = ['RADCURE-2358', 'RADCURE-1645', 'RADCURE-0472', 'RADCURE-1870', 'RADCURE-0431', 'RADCURE-1461',
                    'RADCURE-1715', 'RADCURE-1856', 'RADCURE-2560', 'RADCURE-2620', 'RADCURE-1540', 'RADCURE-0314',
                    'RADCURE-0703', 'RADCURE-1295', 'RADCURE-1786', 'RADCURE-0526', 'RADCURE-1779', 'RADCURE-1570',
                    'RADCURE-1796', 'RADCURE-1707', 'RADCURE-1558', 'RADCURE-1314', 'RADCURE-0609', 'RADCURE-1774',
                    'RADCURE-1003', 'RADCURE-1617', 'RADCURE-0570', 'RADCURE-1787', 'RADCURE-0571', 'RADCURE-1957',
                    'RADCURE-4026', 'RADCURE-1616', 'RADCURE-0926', 'RADCURE-1635', 'RADCURE-0547', 'RADCURE-1878',
                    'RADCURE-1337', 'RADCURE-0701', 'RADCURE-1923', 'RADCURE-2293', 'RADCURE-2121', 'RADCURE-1895',
                    'RADCURE-2102', 'RADCURE-1301', 'RADCURE-1320', 'RADCURE-1966', 'RADCURE-0724', 'RADCURE-1700',
                    'RADCURE-2033', 'RADCURE-0684', 'RADCURE-2298', 'RADCURE-1720', 'RADCURE-1938', 'RADCURE-0999',
                    'RADCURE-1765', 'RADCURE-0987', 'RADCURE-1513', 'RADCURE-1733', 'RADCURE-0748', 'RADCURE-1368',
                    'RADCURE-0317', 'RADCURE-0415', 'RADCURE-1987', 'RADCURE-1282', 'RADCURE-0667', 'RADCURE-0567',
                    'RADCURE-1438', 'RADCURE-2205', 'RADCURE-0406', 'RADCURE-0499', 'RADCURE-1252', 'RADCURE-1889',
                    'RADCURE-1270', 'RADCURE-0953', 'RADCURE-1030', 'RADCURE-0897', 'RADCURE-3099', 'RADCURE-1582',
                    'RADCURE-0912', 'RADCURE-2503', 'RADCURE-0079', "RADCURE-0543", "RADCURE-1856", "RADCURE-2503", 
                    "RADCURE-0079", "RADCURE-2205", "RADCURE-1582", "RADCURE-2986", "RADCURE-2370"]
        try:
            # if os.path.isfile(self.hparams.is_config) is True:
            # ideally this should be a .json file in the format of self.data_config
            # produced by __getDataHparam() below...
            config = getJson(path_) # [self.hparams.fold]
            self.train_data = pd.DataFrame.from_dict({"NEWID":config["train_data"]})
            self.train_data = self.train_data[~self.train_data["NEWID"].isin(exclude_)]
            # self.train_data = self.train_data[10:]
            self.valid_data = pd.DataFrame.from_dict({"NEWID":config["valid_data"]})
            self.test_data =  pd.DataFrame.from_dict({"NEWID":config["test_data"]})
            # else:
            #     warnings.warn(".json file does not exist.")
            #     # configurations should always be based on the training dataset for each fold...
            #     config = self.__getDataHparam(self.train_data)
            #     with open(self.hparams.config_path, "w") as f:
            #         json.dump(config, f)
        except Exception:
            warnings.warn("Path to .json file cannot be read. Creating config...")
            if self.hparams.oar_version != "single":
                fold = self.hparams.fold
                if self.hparams.home_path[-1] != "/":
                    self.hparams.home_path += "/"
                # load dataset paths...
                train_csv_path = str(self.hparams.home_path) + "wolnet-sample/h4htrain2022.csv"#vector_train.csv" # f"wolnet-sample/new_train_fold_{fold}.csv"
                valid_csv_path = str(self.hparams.home_path) + "wolnet-sample/h4hvalid2022.csv"#vector_test.csv" # f"wolnet-sample/new_valid_fold_{fold}.csv"
                test_csv_path = str(self.hparams.home_path)  + "wolnet-sample/vector_test.csv" #f"wolnet-sample/new_test_fold.csv"
                # load corresponding .csv(s) for training fold...
                assert os.path.isfile(train_csv_path) is True
                self.train_data = pd.read_csv(train_csv_path)
                self.valid_data = pd.read_csv(valid_csv_path)
                self.test_data  = pd.read_csv(test_csv_path)
            else:
                
                data_ = getROIOrder(tag=self.tag, inverse=True)
                oars = list(data_.values())
                
                if self.tag != "NECKLEVEL":
                    data = pd.read_csv(f"{self.hparams.home_path}radcure_oar_summary.csv", index_col=0)
                    oar_data = data[data["ROI"].isin(oars)]
                    # exclude_ = ["RADCURE-0543", "RADCURE-3154", "RADCURE-0768"]
                    vals_ = list(oar_data["NEWID"].unique())
                    #################
                    
                # H$H specific...
                current = glob.glob(self.hparams.data_path+"*")
                # current = glob.glob("/cluster/projects/radiomics/Temp/joe/RADCURE_VECTOR_UPDATE/*")
                current = [c.split("/")[-1] for c in current]
                
                if self.tag != "NECKLEVEL":
                    current = [c for c in current if c in vals_]
                
                #################
                oar_data = pd.DataFrame.from_dict({"NEWID":current})
                oar_data = oar_data[~oar_data["NEWID"].isin(exclude_)]
                self.train_data = oar_data[:int(len(oar_data)*.9)]
                self.valid_data = oar_data[int(len(oar_data)*.9):]
                # select random test div for sitsagiigles
                test_csv_path = str(self.hparams.home_path)  + "wolnet-sample/vector_test.csv"
                self.test_data  = pd.read_csv(test_csv_path)
                warnings.warn(f"Creating model with {len(oars)} oars which are {oars}.")

            config = self.__getDataHparam(self.train_data)
            config["train_data"] = list(self.train_data["NEWID"])
            config["valid_data"] = list(self.valid_data["NEWID"])
            config["test_data"] =  list(self.test_data["NEWID"])
            with open(path_, "w") as f:
                json.dump(config,f)

        # vocel info for dataset by OAR
        self.voxel_info = config["VOXINFO"]
        self.config = config["IMGINFO"]
        print(config)
        # other values can be loaded in here as well...
        # ideally the data_config would be saved
        # Kfold["means"][fold] # - 300.
        self.mean = -407.4462155135238
        self.std = 226.03663728492648
        # self.mean = self.config["meanHU"]
        # self.std = self.config["stdHU"]
        # setup custom_order, loaded in with utils.py...
        self.config["roi_order"] = self.custom_order
        self.config["order_dic"] = getROIOrder(tag=self.tag, inverse=True)
        self.oars = list(self.config["order_dic"].values())
        self.config["data_path"] = self.hparams.data_path
        self.config["oar_order"] = self.oars
        warnings.warn(f"OARs chosen are {self.oars}")
        self.eval_data = None
        self.__get_loss()

    @staticmethod
    def load_obj(name):
        with open(name, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_obj(obj, name):
        with open(name, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def forward(self, x):
        # when running deepnet moduel...
        # make sure output is unsqueezed...
        x = x.unsqueeze(1)
        return self.net(x)

    def __getCutomOrder(self):

        self.tag = self.hparams.tag
        if self.tag == "NECK":
            # includes GTV...
            self.custom_order = [3,1,2]
        elif self.tag == "BRACP":
            self.custom_order = [28,29]
        elif self.tag == "NECKMUS":
            self.custom_order = [34,32,33]
        elif self.tag == "TOPHEAD":
            self.custom_order = [13,8,14,15,16,11,12]
        elif self.tag == "GLANDS":
            self.custom_order = [9,10,19,20,23,24,25,26,27,17,18,21,22]
        elif self.tag == "MAJOR":
            self.custom_order = [13,8,14,15,16,11,12,23,24,25,26,27,29,28,9,10,17,18,21,22,19,20,4,3,1,2]
        elif self.tag == "LACRIM":
            self.custom_order = [9,10]
        elif self.tag == "PAROTI":
            self.custom_order = [17,18]
        elif self.tag == "SUBMAN":
            self.custom_order = [21,22]
        elif self.tag == "COCHLEA":
            self.custom_order = [23,24]
        elif self.tag == "LARYNX":
            self.custom_order = [4,5,6,7]
        elif self.tag == "BRAIN":
            self.custom_order = [4,30,31]
        elif self.tag == "SPINE":
            self.custom_order = [29,28,30,7,31,4,25,26,27,4,5,6,19,20]
        elif self.tag == "MIDHEAD":
            self.custom_order = [23,24,17,18,19,20,21,22,9,10,4,5,6]
        elif self.tag == "OTHER":
            self.custom_order = [19,20,21,22,23,24,25,26,27,28,29,30,31,17,18,4]
        elif self.tag == "NECKLEVEL":
            self.custom_order = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        else:
            self.custom_order=custom_order
            warnings.warn("Tag not specified...using general ordering.")
            # will load in custom_order in utils.py...
            pass

    # ---------------------
    # TRAINING
    # ---------------------
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        """
        print("AT TRAIN START")
        self.step_type = "train"
        inputs, targets, counts = batch
        if inputs.shape != targets.shape:
            warnings.warn("Input Shape Not Same size as label...")
        if batch_idx == 0:
            print(inputs.max(), inputs.size())
            print(targets.max(), targets.size())
        outputs = self.forward(inputs) # WOLNET
        if type(outputs) == tuple:
            outputs = outputs[0]
        loss = self.criterion(outputs, targets, counts)
        # if loss is nan...
        # if torch.isnan(loss)[0] is True:
        nan_val = 10 + len(self.custom_order)
        loss = torch.nan_to_num(loss, nan=nan_val, posinf=nan_val)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss':loss}
        # calculate and log dices...
        ###############################
        # max_ = targets.max()
        # outputs, targets = onehot(outputs, targets)
        # # calculate dice for logging...
        # # can add other metrics here...
        # dices = monmet.compute_meandice(outputs, targets)
        # # compute_hausdorff_distance # compute_average_surface_distance
        # # get size of dice array,
        # # fist dim should be that of batch...
        # s = dices.size()
        # if s[0]==1:
        #     dices = dices[0]
        #     # hdfds = hdfds[0]
        #     # asds = asds[0]
        # else:
        #     dices=dices.mean(dim=0)
        #     # hdfds=hdfds.mean(dim=0)
        #     # asds = asds.mean(dim=0)
        #
        # # use counts to filter out which metrics to log for set OAR...
        # counts = counts[0].cpu().numpy()
        # bool_counts = (counts == 1)
        # counts_ = np.where(bool_counts)[0]
        # print(counts, bool_counts, counts_)
        # # counts_ = list(counts_.astype(bool))
        # # counts_2 = counts_[:max_]
        # try:
        #     dices_ = dices[counts_]
        # except Exception:
        #     try:
        #         counts_ = counts_[:len(counts_)-1]
        #         dices_ = dices[counts_]
        #         print(dices_)
        #     except Exception:
        #         dices_ = dices[0]
        #         counts_ = counts_[0]
        #
        # # hdfds_ = hdfds[counts_]
        # # print(hdfds_)
        # # asds_ = asds[counts_]
        # # print(asds_)
        # if "BACK" not in self.oars:
        #     self.oars = ["BACK"] + self.oars
        # oars_ = np.array(self.oars)[bool_counts]
        # # Let's log it shall we...
        # # try:
        # for i, val in enumerate(dices_):
        #     # if counts[0][i] == 1:
        #     self.log(f'train_dice_{oars_[i]}', val, on_step=True, prog_bar=True, logger=True)
        #         # be sure to log 95%HD if uncommented above
        #         # self.log(f'train_haus_{i}', hdfds[i], on_step=True, logger=True)
        # except Exception:
        #     self.log(f'train_dice_{oars_[0]}', dices_, on_step=True, prog_bar=True, logger=True)
        ########################################

    # ---------------------
    # Run Validation Step, Runs after Trainning Epoch converges
    # This can be modulated in Trainer() when running train.py
    # ---------------------
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        """
        self.step_type = "valid"
        inputs, targets, counts = batch
        shape = inputs.size()
        if batch_idx == 0:
            print(inputs.max(), inputs.size())
            print(targets.max(), targets.size())
        if inputs.shape != targets.shape:
            warnings.warn("Input Shape Not Same size as label...")
        # calculate loss...
        outputs = self.forward(inputs)
        if type(outputs) == tuple:
            outputs = outputs[0]
        loss = self.criterion(outputs, targets, counts)
        # (self.criterion(outputs, targets.unsqueeze(1)).cpu() if self.criterion is not None else 0)
        nan_val = 10 + len(self.custom_order)
        loss = torch.nan_to_num(loss, nan=nan_val, posinf=nan_val)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # calculating evaluation metrics metrics
        # apply soft/argmax to outputs...
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs , dim=1)
        max_ = targets.max()
        outputs, targets = onehot(outputs, targets, argmax=False)
        dices = monmet.compute_meandice(outputs, targets)
        print(dices.size(),dices)
        s = dices.size()
        if s[0]==1:
            dices = dices[0]
            # hdfds = hdfds[0]
            # asds = asds[0]
        else:
            dices=dices.mean(dim=0)
            # hdfds=hdfds.mean(dim=0)
            # asds = asds.mean(dim=0)
        # use counts to filter out which metrics to log for set OAR...
        counts = counts[0].cpu().numpy()
        bool_counts = (counts == 1)
        counts_ = np.where(bool_counts)[0]
        print(counts, bool_counts, counts_)
        try:
            try:
                dices_ = dices[counts_]
            except Exception:
                try:
                    counts_ = counts_[:len(counts_)-1]
                    dices_ = dices[counts_]
                    bool_counts = bool_counts[:len(counts_)-1]
                except Exception:
                    try:
                        counts_ = counts_[:len(counts_)-2]
                        dices_ = dices[counts_]
                        bool_counts = bool_counts[:len(counts_)-2]
                    except Exception:
                        counts_ = counts_[:len(counts_)-3]
                        dices_ = dices[counts_]
                        bool_counts = bool_counts[:len(counts_)-3]

            print(dices_)
            hdfds_ = hdfds[counts_]
            print(hdfds_)
            asds_ = asds[counts_]
            print(asds_)
            if "BACK" not in self.oars:
                self.oars = ["BACK"] + self.oars
            oars_ = np.array(self.oars)[bool_counts]
            print(oars_, bool_counts)
            for i, val in enumerate(dices_):
                self.log(f'val_dice_{oars_[i]}', val, on_step=True, prog_bar=True, logger=True)
                # logging individual evaluation metrics...
                # if you have the order of OAR(s) - given that they're variable, i can be replaced with ROI name...
                # if counts[i] == 1:
                # self.log(f'val_haus_{oars_[i]}', hdfds_[i], on_step=True, logger=True)
                # self.log(f"val_asds_{oars_[i]}", asds_[i], on_step=True, logger=True)
                # logging evaluation metrics ...
                # self.log(f"val_EVAL_{oars_[i]}", val*100/(hdfds_[i]+asds_[i]), on_step=True, logger=True)
        except Exception:
            warnings.warn(f"Couldn't log metrics for validation step {batch_idx}.")

        ###### EXTRA STUFF ##########
        # Can change this to incorporate sliding window...
        # roi_size = (int(shape[1]), 192, 192) # (64, 192, 192)
        # sw_batch_size = 1 # second sliding window inference
        # outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward)
        # evaluation using sliding window inference only (really) required during testing.
        # if batch_idx%2==0:
        ################
        # grab images to save during training...
        # in_   = inputs.cpu().numpy()
        # targ_ = targets.cpu().numpy()
        # out_  = outputs.cpu().detach().numpy()
        # in_, out_, targ_ = self.export_figs(in_, out_, targ_)
        ################

        # hdfds = monmet.compute_hausdorff_distance(outputs, targets, percentile=95,
        #                                            include_background=True)
        # asds = monmet.compute_average_surface_distance(outputs,targets, include_background=True)
        # only use if you'd like to plot example of outputs...
        # Note: Best way to do that is to define validation_epoch_end
        # output = OrderedDict( { "val_loss": loss,
        #         "input": torch.from_numpy(in_).type(torch.FloatTensor),
        #         "out": torch.from_numpy(out_).type(torch.FloatTensor),
        #         "targ": torch.from_numpy(targ_).type(torch.FloatTensor),})
        # counts_ = list(counts_.astype(bool))
        # counts_2 = counts_[:max_]
        #################################
        #################################

    def CalcEvaluationMetric(self, outputs, targs, batch_idx, total_time):

        self.patient = str(self.data.iloc[batch_idx][0])
        roi_order = self.config["roi_order"]
        # only do this if targets not loaded in with data
        # however, targets will be loaded in given sample dataloader was used...
        # structure_path = self.config["data_path"] + f'/{self.patient}/structures/'
        # structures = glob.glob(f'{structure_path}/*')
        oars = []
        dice = []
        haus = []
        asds = []
        eval = []
        pats = []
        p_idx = []
        time = []
        for j, c in enumerate(roi_order):
            oar = roi_order[j]
            try:
                # ideally this should be done outside this function...
                # some OARs will not be included in the targets...
                # allows us to save only OARs that we have ground truth information for.
                targ = targs.copy()
                targ = targ.cpu().numpy()
                targ[targ!=j+1] = 0
                if len(targ[targ==j+1]) == 0:
                    pass
                else:
                    targ[targ==j+1] = 1
                    outs = outputs[0,j+1]
                    ###############################
                    dc = met.compute_meandice(outs.unsqueeze(0).unsqueeze(0), targ.unsqueeze(0).unsqueeze(0))
                    h  = met.compute_hausdorff_distance(outs.unsqueeze(0).unsqueeze(0), targ.unsqueeze(0).unsqueeze(0), percentile=95, include_background=False)
                    s  = met.compute_average_surface_distance(outs.unsqueeze(0).unsqueeze(0), targ.unsqueeze(0).unsqueeze(0), include_background=False)
                    # print(self.patient, c, dc, h)
                    # save metrics...
                    oars.append(oar)
                    dice.append(dc[0][0].item())
                    haus.append(h[0][0].item())
                    asds.append(s[0][0].item())
                    eval.append(dc[0][0].item()/(h[0][0].item()+s[0][0].item()))
                    pats.append(folder)
                    p_idx.append(batch_idx)
                    time.append(total_time)

            except Exception as e:
                # print(e)
                pass

        data = {"ID":p_idx,"PATIENT":pats,"OAR":oars, "DICE":dice, "95HD":haus, "ASD":asds, "EVAL":eval}
        if self.eval_data is None:
            self.eval_data = pd.DataFrame.from_dict({data})
        else:
            # this being tun on different machines, you have to account for
            # varrying devices...
            self.eval_data = pd.concat([pd.DataFrame.from_dict({data}), self.eval_data])

        # save data after each iteration...
        # final model score will be mean across all OARs...
        self.eval_data.to_csv(f"{str(self.root)}/{self.model_name}_test.csv")

    def CropEvalImage(self,inputs):
        ###########################
        # IMAGE CROPPING/PADDING if required
        ###########################
        to_crop = RandomCrop3D( window=self.hparams.window, mode="test",
                                 factor=292,#self.hparams.crop_factor,
                                 crop_as=self.hparams.crop_as)
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

        img, targ, center = to_crop(inputs,targets,in_)
        # varry's depending on imgsize used to train the model...
        roi_size = (112, 176, 176)
        shape = img.size()
        # assumes first and last eight of image are fluff
        if 180<=shape[1]:
            cropz = (shape[1]//12, shape[1]-shape[1]//12)
        elif 165 <= shape[1] < 180:
            diff = 180 - shape[1]
            cropz = (shape[1]//13, shape[1]//13+152-diff)
        else:
            cropz = (0, shape[1])

        img = img[:,cropz[0]:cropz[1]]

        return img, targ, center, cropz


    def test_step(self, batch, batch_idx):
         """
         Lightning calls this inside the testing loop;
         this can/should be modified depending on your pipeline...
         """
         #######################
         # setup paths and directories to save model exports
         self.step_type = "test"
         inference_outputs_path = str(self.root) + f"/TESTING/"
         outputs_path = inference_outputs_path + f"FOLD_{self.hparams.fold}"
         os.makedirs(inference_outputs_path, exist_ok=True)
         os.makedirs(outputs_path, exist_ok=True)
         #######################

         inputs, targets  = batch
         if batch_idx == 0:
             print(inputs.max())
         og_shape = inputs.size()
         if og_shape[1] == 512:
            inputs = inputs.permute(0,3,1,2)
         in_ = inputs.cpu().numpy()
         og_shape = inputs.size()

         shape = img.size()
         warnings.warn(f'First crop size is {shape},')
         img, targ, center, cropz = self.CropEvalImage(inputs)
         ###########################
         ## SLIDING WINDOW INFERENCE EXAMPLES
         ###########################
         a_time = time.time()
         outputs = swi(img, self.forward, 20)
         warnings.warn("Done iteration 1")
         outputs_ = swi(img.permute(0,1,3,2), self.forward, 20)
         b_time = time.time()
         total_time = b_time - a_time # total inference time in seconds...
         warnings.warn("Done iteration 2")
         outputs_ = outputs_.permute(0,1,2,4,3)
         outputs = torch.mean(torch.stack((outputs, outputs_), dim=0), dim=0)
         warnings.warn(f'Hello size is {outputs.size()},')

         ###########################
         # this is the infernece using built in MONAI...
         # to_crop = RandomCrop3D(
         #           window=self.hparams.window,
         #           mode="test",
         #           factor=200,#self.hparams.crop_factor,
         #           crop_as=self.hparams.crop_as)
         # outputs = swi(img, roi_size, sw_batch_size, self.forward, overlap=0.8, mode='gaussian') # from monai 0.4.0
         # hello is it me you're looking for?
         # img = img[:,cropz[0]:cropz[1]]
         # because we're passing the same image through, mean the outputs...
         # outputs = torch.mean(outputs, 0)
         ############################
         ############################

         if type(outputs) == tuple:
             outputs = outputs[0]
         if self.hparams.crop_as != "3D":
             outputs = outputs.squeeze(2)

         ############################
         ##### This save(s) model outputs...
         ############################
         # targ = targets.clone() #.cpu()
         warnings.warn(f'Hello size is {outputs.size()}')
         out = outputs.clone() #.cpu()
         outs = torch.softmax(out, dim=1)
         warnings.warn(f'Hello size is {outs.size()} AFTER SOFTMAX')
         # sum predictions after softmax BECAUSE originally
         # trained with batch_size == 2
         outs = torch.mean(outs, dim=0)
         outs_raw = outs.cpu().numpy()
         warnings.warn(f'Hello size is {outs.size()} AFTER SOFTMAX')
         outs = torch.argmax(outs, dim=0)
         # runs calculation of evaluation metric...
         self.CalcEvaluationMetric(outs, targets, batch_idx, total_time)
         #######################
         # here we can compute evaluation metrics...
         # both outputs and targets have to be one hot encoded...
         #######################
         inp = inputs[0]
         warnings.warn(f'OUTPUT size is {outs.size()} with inputs {inp.size()}')
         # assert outs.size() == inp.size()
         out_full = torch.zeros(inp.size())
         warnings.warn(f'Hello size is {inp.size()}')
         out_full[cropz[0]:cropz[1], center[0]:center[0]+292,center[1]:center[1]+292] = outs

         idx=0
         targ_path = outputs_path + f'targ_{batch_idx+idx}_FULL.nrrd'
         img_path =  outputs_path + f'input_{batch_idx+idx}_FULL.nrrd'

         # uncomment this if you'd like to resave targets, not necessary...
         # if os.path.isfile(targ_path) is True:
         #     pass
         # else:
         #     in_ = inp.cpu().numpy()
         #     targ_ = targets.cpu().numpy()
         #     nrrd.write(img_path, in_)
         #     nrrd.write(targ_path, targ_[0].astype('uint8'), compression_level=9)
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

    ################################
    ################################
    # Essential Support Functions
    ################################
    ################################

    def get_model_root_dir(self):

        self.ts = time.time()
        self.date = datetime.datetime.fromtimestamp(self.ts).strftime("%Y_%m_%d_%H%M%S")
        self.model_path = self.hparams.model_path

        model_names = [ "DEEPNET", "SIMPLE", "UNET", "WOLNET", "RESUNET",
                        "ATTENTION3D", "VNET", "MODIFIED3DUNET", "TIRAMISU",
                        "TERANUS", "ELEKTRONUNET", "ANATOMY", "DEEPLABV3",
                        "PIPOFAN", "HIGHRESNET", "UNET3+", "UNET3+DEEPSUP",
                        "UNET++", "VGG163D", "RSANET", "HYPERDENSENET",
                        "DENSEVOX", "MEDNET", "SKIPDNET"]

        if self.hparams.model in model_names:
            if self.hparams.model_name is None:
                self.model_name = f"{self.hparams.model}_{self.date}"
                self.hparams.model_name = self.model_name
            else:
                self.model_name = self.hparams.model_name

        self.hparams.root = str(self.model_path + "/" + self.model_name)
        self.root = Path(self.hparams.root)
        self.root.mkdir(exist_ok=True, parents=True)

    def configure_optimizers(self):

        ada = ['ADABOUND', 'AMSBOUND']
        if self.hparams.optim == "ADAM":
            init_optimizer = torch.optim.Adam( self.net.parameters(),lr=self.hparams.lr,
                                               weight_decay=self.hparams.decay)
            warnings.warn("Using ADAM as default optimizer.")
        elif self.hparams.optim == "RADAM":
            # the new RADAM optimizer as defined in https://arxiv.org/pdf/1908.03265.pdf
            init_optimizer = RAdam(self.net.parameters(), lr=self.hparams.lr,
                                   weight_decay=self.hparams.decay)
        elif self.hparams.optim == "RMSPROP":
            init_optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.hparams.lr,
                                                 weight_decay=self.hparams.decay)
        elif self.hparams.optim in ada:
            ams = True if self.hparams.optim == 'AMSBOUND' else False
            init_optimizer = AdaBoundW(self.net.parameters(), lr=self.hparams.lr,
                                       final_lr=0.001,weight_decay=self.hparams.decay,
                                       amsbound=ams)
        else:
            warnings.warn("Using SGD as default optimizer.")
            init_optimizer = torch.optim.SGD( self.net.parameters(), lr=self.hparams.lr,
                                              momentum=0.9, weight_decay=self.hparams.decay,
                                              nesterov=True,)

        # Set up loss decay via PTL scheduler...
        if self.hparams.scheduler is True:
            if self.hparams.scheduler_type == 'plateau':
                scheduler = ReduceLROnPlateau(init_optimizer, factor=self.hparams.gamma,
                                              patience=self.hparams.decay_after,
                                              threshold=0.0001)
            else:
                scheduler = StepLR( init_optimizer, step_size=self.hparams.decay_after,
                                    gamma=self.hparams.gamma,)

            return [init_optimizer], [scheduler]
        else:
            return [init_optimizer]

    def __getDataHparam(self, data):
        '''
        This will define data_config dictionaries based on a dataframe of images
        and structures...
        '''
        folders = list(data["NEWID"])
        if self.hparams.data_path[-1] != "/":
            self.hparams.data_path += "/"
        warnings.warn(self.hparams.data_path)
        # we want this to be a list of list(s)
        # contain the paths to the structures for each patient
        folders = [glob.glob(self.hparams.data_path + fold + "/structures/*") for fold in folders]
        config = getHeaderData(folders, tag=self.tag)
        # config["IMGINFO"]["VOXINFO"] = voxel_info
        return config # ["IMGINFO"]
    
# test above function...
# def __getDataHparam(data, path):
#     '''
#     This will define data_config dictionaries based on a dataframe of images
#     and structures...
#     '''
#     folders = list(data["NEWID"])
#     if path[-1] != "/":
#         path += "/"
#     warnings.warn(path)
#     # we want this to be a list of list(s)
#     # contain the paths to the structures for each patient
#     folders = [glob.glob(path + fold + "/structures/*") for fold in folders]
#     config = getHeaderData(folders, tag="NECKLEVEL")
#     # config["IMGINFO"]["VOXINFO"] = voxel_info
#     return config  # ["IMGINFO"]

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Please inspect individual models in utils/models ...
        Layout model :return: n_classes + 1 (because we need to include background)
        """

        classes = self.hparams.n_classes + 1
        if self.hparams.model == "DEEPNET":
            self.net = DeepUNet(num_classes=classes, sub_enc=self.hparams.sub_enc)
        elif self.hparams.model == "SIMPLE":
            self.net = simpleUNet(num_classes=classes)
        elif self.hparams.model == "UNET":
            self.net = UNet3D( num_classes=classes, scale=self.hparams.scale_factor,
                               deformable=self.hparams.deform, project=self.hparams.project)
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
            self.net = FCDenseNet( in_channels=1, down_blocks=(2, 2, 2, 2, 3),
                                   up_blocks=(3, 2, 2, 2, 2), bottleneck_layers=2,
                                   growth_rate=12, out_chans_first_conv=16,
                                   n_classes=classes,)
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
            self.net = UNet_3Plus_DeepSup(n_classes=classes, factor=8)
        elif self.hparams.model == "RSANET":
            self.net = RSANet(n_classes=classes)
        elif self.hparams.model == "HYPERDENSENET":
            self.net = HyperDenseNet(in_channels=1, num_classes=classes)
        elif self.hparams.model == "DENSEVOX":
            self.net = DenseVoxelNet(in_channels=1, num_classes=classes)
        elif self.hparams.model == "MEDNET":
            self.net = generate_resnet3d(in_channels=1, classes=classes, model_depth=10)
        elif self.hparams.model == "SKIPDNET":
            self.net = SkipDenseNet3D(growth_rate=16, num_init_features=self.hparams.f_maps,
                                      drop_rate=0.1, classes=classes)

    # ------------------
    # Assign Loss
    # ------------------
    def __getWeights(self):

        # self.voxel_info is an ordered dictionary with OAR name linked to voxel information for each set OAR class...
        rois = list(self.voxel_info.keys())
        self.config["ROIS"] = rois
        print(rois)
        # have to add the background class, we give very low weight to background as they far outnumber class specific pixels.
        # define very small weight for base value(s)
        base = np.array([1e-40])
        values = list(self.voxel_info.values())
        print(values)
        values = [np.int(v) for v in values]
        weights = np.array(values)/(np.sum(values)+1e-4)
        weights = np.append(base, weights)
        # take the inverse of the absolute log of the weights...
        weights = np.abs(np.log(weights))
        weights[0] = 0.01
        self.config["weights"] = weights
        warnings.warn(f"Weights are {weights}")
        return weights

    def __get_loss(self):

        # self.class_weights this must be provided and calculated separately...
        # class should be located in self.data_config...
        # usually this will be the amount of voxels given for any OAR class...
        self.class_weights = self.__getWeights()
        self.class_weights = torch.tensor(self.class_weights).float()
        warnings.warn(f"Using weights {self.class_weights}\nWith voxel information: {self.voxel_info}")
        assert len(self.class_weights) == self.hparams.n_classes + 1
        if self.hparams.loss == "FOCALDSC":
            # binary TOPK loss + DICE + HU
            # enables us to tackle class imbalance...
            loss = FocalLoss(weight=self.class_weights)  # Metric from elektronn3
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
        elif self.hparams.loss == "TAL":
            loss = TALWrapper(weight=self.class_weights, do_bg=False)
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

    def get_dataloader( self, df, mode="valid", transform=None, resample=None,
                        shuffle=False, transform2=None, batch_size=None):

        dataset = LoadPatientVolumes(folder_data=df, data_config=self.config,
                                     tag=self.hparams.tag,transform=transform)

        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        # best practices to turn shuffling off during validation...
        validate = ['valid', 'test']
        shuffle = False if mode in validate else True

        return DataLoader( dataset=dataset, num_workers=self.hparams.workers,
                           batch_size=batch_size, pin_memory=True, shuffle=shuffle,
                           drop_last=True,)

    def train_dataloader(self):

        transform = Compose(
            [   HistogramClipping(min_hu=self.hparams.clip_min, max_hu=self.hparams.clip_max),
                RandomFlip3D(), # left and right should be distinguished...
                RandomRotation3D(p=self.hparams.aug_prob/1.5),
                ElasticTransform3D(p=self.hparams.aug_prob/1.5),
                RandomZoom3D(p=self.hparams.aug_prob/1.5),
                RandomCrop3D(window=self.hparams.window, mode="train",
                             factor=self.hparams.crop_factor, crop_as=self.hparams.crop_as,),
                NormBabe(mean=self.mean, std=self.std, type=self.hparams.norm),
            ]
        )

        # add transform to dataloader
        return self.get_dataloader(df=self.train_data, mode="train", transform=transform,
                                   resample=False, batch_size=self.hparams.batch_size,)

    # @pl.data_loader
    def val_dataloader(self):
        # imported from transform.py
        transform = Compose(
            [ HistogramClipping(min_hu=self.hparams.clip_min, max_hu=self.hparams.clip_max),
              RandomCrop3D(window=self.hparams.window, mode="valid",
                           factor=self.hparams.crop_factor,crop_as=self.hparams.crop_as,),
              NormBabe(mean=self.mean, std=self.std, type=self.hparams.norm),
            ]
        )

        return self.get_dataloader( df=self.valid_data, mode="valid",
                                    transform=transform,  # should be default
                                    resample=False, batch_size=self.hparams.batch_size,)
    # @pl.data_loader
    def test_dataloader(self):
        # during inference we will run each model on the test sets according to
        # the data_config which you will provide which each model...
        # should be able to load in own test_csv with folder names just like for trianing/validation...
        if os.path.isfile(self.hparams.test_csv) is not True:
            self.test_data = pd.read_csv(self.hparams.test_csv, index_col=0)
        transform = Compose([ HistogramClipping(min_hu=self.hparams.clip_min,
                                                max_hu=self.hparams.clip_max),
                              NormBabe(mean=self.mean, std=self.std,
                                       type=self.hparams.norm),])

        return self.get_dataloader( df=self.test_data, mode="test",transform=transform, # transform,  # should be default
                                    transform2=None, resample=self.hparams.resample,
                                    batch_size=self.hparams.batch_size,
        )
