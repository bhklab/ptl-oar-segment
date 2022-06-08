import os, torch, time, datetime, warnings, pickle, json, glob, nrrd
from pathlib import Path
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from .prepare import GetSplits, PatientData, LoadPatientSlices, LoadPatientVolumes
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from .scheduler import Poly
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import pytorch_lightning as pl
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
        self.save_hyperparameters(hparams) # 1.3+
        self.get_model_root_dir()
        self.__build_model()

    # def prepare_data(self):
    #     """
    #     This class specific to downloading dataset(s) if defined.
    #     See https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/datamodules.html
    #     """

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
        #  "dataset_mean": np.float(), "dataset_std": np.float(), specifically used for Z-score normalization of images...
        #  "clip_max": 1000 (Recommended), "clip_min": -500 (Recommended)}
        # NOTE: if windowing will be applied to images, mean/std of dataset must reflect the windowed version of the image
        # for testing the following can be used... this can be calculated on the fly ...
        # the voxel counts for each class are given in the meta header of each .nrrd in the structure folder...
        # the unwindowed meanHU and stdHU are saved in the meta header for each image...
        # for clipped image(s) from -500 to 1000; expect mean/std values to
        # fall within the following ranges...
        # -390 < meanHU < -420; 205 < stdHU < 245

        fold = self.hparams.fold
        train_csv_path = str(self.root) + f"/train_fold_{fold}.csv"
        valid_csv_path = str(self.root) + f"/valid_fold_{fold}.csv"
        test_csv_path = str(self.root) + f"/test_fold.csv"
        assert os.path.isfile(train_csv_path) is True
        self.train_data = pd.read_csv(train_csv_path)
        self.valid_data = pd.read_csv(valid_csv_path)
        self.test_data  = pd.read_csv(test_csv_path)
        try:
            if self.hparams.image_processing_config_path is not None:
                self.data_config = getJson(self.hparams.image_processing_config)[self.hparams.fold]
            else:
                self.data_config = self.__getDataHparam(self.train_data)
        except Exception:
            warnings.warn("Path to .json file cannot be read.")
            self.data_config = self.__getDataHparam(self.train_data)

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

    # ---------------------
    # TRAINING
    # ---------------------
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        """
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
        loss = self.criterion(outputs, targets)
        outputs, targets = onehot(outputs, targets)
        # calculate dice for logging...
        # can add other metrics here...
        dices = monmet.compute_meandice(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # get size of dice array,
        # fist dim should be that of batch...
        s = dices.size()
        dices = dices[0] if s[0]==1 else dices.mean(dim=0)
        # Example if wanted to record 95%HD during training...
        # hdfds = monmet.compute_hausdorff_distance(outputs, targets, percentile=95, include_background=True)
        for i, val in enumerate(dices):
            self.log(f'train_dice_{i}', dices[i], on_step=True, prog_bar=True, logger=True)
            # be sure to log 95%HD if uncommented above
            # self.log(f'train_haus_{i}', hdfds[i], on_step=True, logger=True)
        return {'loss':loss}

    # ---------------------
    # Run Validation Step, Runs after Trainning Epoch converges
    # This can be modulated in Trainer() when running train.py
    # ---------------------
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        """
        self.step_type = "valid"
        inputs, targets = batch
        shape = inputs.size()

        if batch_idx == 0:
            print(inputs.max(), inputs.size())
            print(targets.max(), targets.size())
        if inputs.shape != targets.shape:
            warnings.warn("Input Shape Not Same size as label...")

        # Can change this to incorporate sliding window...
        # roi_size = (int(shape[1]), 192, 192) # (64, 192, 192)
        # sw_batch_size = 1 # second sliding window inference
        # outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward)
        # evaluation using sliding window inference only (really) required during testing.

        outputs = self.forward(inputs)
        if type(outputs) == tuple:
            outputs = outputs[0]
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

        # only use if you'd like to plot example of outputs...
        # Note: Best way to do that is to define validation_epoch_end
        # output = OrderedDict( { "val_loss": loss,
        #         "input": torch.from_numpy(in_).type(torch.FloatTensor),
        #         "out": torch.from_numpy(out_).type(torch.FloatTensor),
        #         "targ": torch.from_numpy(targ_).type(torch.FloatTensor),})

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
        #     output[f"val_dice_{i}"] = dices[i]
        #     output[f"val_haus_{i}"] = hdfds[i]
        #
        # return output

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

         inputs, _  = batch
         if batch_idx == 0:
             print(inputs.max())
         in_ = inputs.cpu().numpy()
         og_shape = inputs.size()

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
         shape = img.size()
         warnings.warn(f'First crop size is {img.size()},')
         ##########################
         ##########################

         ###########################
         ## SLIDING WINDOW INFERENCE EXAMPLES
         ###########################
         outputs = swi(img, self.forward, 20)
         warnings.warn(f'Hello size is {outputs.size()},')
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
             outputs = outputs.squeeze(2)  # shouldn't this be conditional??

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
         inp = inputs[0]
         warnings.warn(f'OUTPUT size is {outs.size()} with inputs {inp.size()}')
         # assert outs.size() == inp.size()
         out_full = torch.zeros(inp.size())
         warnings.warn(f'Hello size is {inp.size()}')
         out_full[cropz[0]:cropz[1], center[0]:center[0]+292,center[1]:center[1]+292] = outs

         idx=0
         targ_path = outputs_path + f'targ_{batch_idx+idx}_FULL.nrrd'
         img_path =  outputs_path + f'input_{batch_idx+idx}_FULL.nrrd'

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
            init_optimizer = RAdam(self.net.parameters(), lr=self.hparams.lr,)
                                   weight_decay=self.hparams.decay)
        elif self.hparams.optim == "RMSPROP":
            init_optimizer = torch.optim.RMSprop(self.net.parameters(),
                                                 lr=self.hparams.lr,
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
        pass

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
    def __get_loss(self):

        # self.class_weights this must be provided and calculated separately...
        # class should be located in self.data_config...
        # usually this will be the amount of voxels given for any OAR class...
        self.class_weights = self.data_config["weights"]
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

        dataset = LoadPatientVolumes(
            df=df, transform=transform, transform2=transform2, window=self.hparams.window, # ,
            root=self.root, mode=mode, to_filter=self.hparams.filter, spacing=self.hparams.spacing,
            volume_type=self.hparams.volume_type, oar_version=self.hparams.oar_version,
            external=False,)

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
        transform = Compose([ HistogramClipping(min_hu=self.hparams.clip_min,
                                                max_hu=self.hparams.clip_max),
                              NormBabe(mean=self.mean, std=self.std,
                                       type=self.hparams.norm),])

        return self.get_dataloader( df=self.test_data, mode="test",transform=transform, # transform,  # should be default
                                    transform2=None, resample=self.hparams.resample,
                                    batch_size=self.hparams.batch_size,
        )

    # ---------------------
    # SAVE Figures : Plot to Tensorboard
    # ---------------------
    # def save_metrics(self, batch_idx, loss, dices, hus, name="train"):
    #
    #     if batch_idx % 5 == 0:
    #         # logging during training
    #         step = self.trainer.global_step
    #         # try:
    #         self.logger.experiment.add_scalar(f"{name}/loss", loss, step)
    #         self.logger.experiment.add_scalars(f"{name}/dices", dices, step)
    #         self.logger.experiment.add_scalars(f"{name}/hausd", hus, step)
    #
    # def export_figs(self, inputs, outputs, targets):
    #
    #     # plt.switch_backend('agg')
    #     # assume sigmoid applied to outputs
    #     shape = outputs.shape  # size of outputs
    #
    #     if shape[0] != 1:
    #         # choose random image in batch
    #         a = np.arange(1, shape[0])
    #         a = np.random.choice(a)
    #     else:
    #         # satisfies case when only one image in batch
    #         a = 0
    #
    #     in_pic = inputs[a]
    #     out_pic = outputs[a]
    #     targ_pic = targets[a]
    #
    #     # should be the same dimensions
    #     assert out_pic.shape == targ_pic.shape
    #
    #     in_ = in_pic[self.hparams.window]
    #
    #     if len(targ_pic.shape) == 3:
    #         # plt.imshow(np.sum(targ_pic, axis=0), cmap='jet', alpha=0.5)
    #         targ = np.sum(targ_pic, axis=0)
    #     else:
    #         # targets already in n_classes - 1 (0 background: 1 CTV, 2 GTV)
    #         # plt.imshow(targ_pic, cmap='jet', alpha=0.5)
    #         targ = targ_pic
    #
    #     if len(out_pic.shape) == 3:
    #         # plt.imshow(np.sum(out_pic, axis=0), cmap='jet', alpha=0.5)
    #         out = np.sum(out_pic, axis=0)
    #     else:
    #         # targets already in n_classes - 1 (0 background: 1 CTV, 2 GTV)
    #         # plt.imshow(out_pic, cmap='jet', alpha=0.5)
    #         out = out_pic
    #
    #     return in_, out, targ
    #
    # def save_figs(self, in_, out, targ, name="train", step=None):
    #
    #     if step is None:
    #         step = self.trainer.global_step
    #
    #     plt.switch_backend("agg")
    #     # plot & save ground truth targets
    #     fig1 = plt.figure(1)
    #     plt.axis("off")
    #     plt.imshow(in_, cmap="gray", interpolation="none")
    #     plt.imshow(targ, cmap="jet", alpha=0.5)
    #     # save targets
    #     self.logger.experiment.add_figure(f"{name}/ground_truth", fig1, step)
    #     plt.close()
    #
    #     # plot and save predicated masks
    #     fig2 = plt.figure(2)
    #     plt.axis("off")
    #     plt.imshow(in_, cmap="gray", interpolation="none")
    #     plt.imshow(out, cmap="jet", alpha=0.5)
    #     # save model predictions
    #     self.logger.experiment.add_figure(f"{name}/prediction", fig2, step)
    #     plt.close()
    #
    # def metrics(self, outs, targ, outputs=None):
    #     # Metrics requiring one hot encoded targets, pass through sigmoid or softmax
    #     # convert to one hot encoded target...
    #     shape = targ.size()
    #     batch = shape[0]
    #     # calculate argmax...
    #     outputs = torch.argmax(outs, dim=1)
    #     if len(shape) == 4:
    #         # 3D OUTPUT ARRAY HAS TO BE 5D IF ONE HOT ENCODED
    #         # see what you do when you initialize a new tensor
    #         # https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html
    #         sh = (batch, self.hparams.n_classes + 1, shape[1], shape[2], shape[3])
    #         targets_dice = torch.zeros(sh, dtype=torch.float) # .type_as(outs)
    #         targets_out = torch.zeros(sh, dtype=torch.float) # .type_as(targ)
    #
    #     for i in range(self.hparams.n_classes + 1):
    #
    #         targets_dice[:, i][targ == i] = 1
    #         targets_out[:, i][outputs == i] = 1
    #
    #     # calculate distance accuracy metrics
    #     losses = ['ANATOMY', 'COMBINEDFOCAL']
    #
    #     if self.hparams.loss in losses:
    #         # assert outs without argmax...
    #         loss = (self.criterion2(outs, targets_dice) if self.criterion2 is not None else 0)
    #     else:
    #         loss = (self.criterion2(targets_out, targets_dice) if self.criterion2 is not None else 0)
    #
    #     # if self.criterion3 is not None:
    #     loss += (self.criterion3(targets_out, targets_dice) if self.criterion3 is not None else 0)
    #
    #     dices, hauss = getMetrics(targets_out.detach().numpy(), targets_dice.detach().numpy(), multi=True)
    #
    #     #retrun all metrics including argmaxed outputs...
    #     return dices, hauss, loss, outputs

    # Modify if image logging required after validation completed....
    # can do fancy things here like plot bar graphs of exports, not necessary
    # but nice if you'd like to visualize your progress across all OARs
    # simultaneously...
    # def validation_epoch_end(self, outputs):
    #
    #     # average validation loss
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     print(
    #         f"Average validation loss for Epoch: {self.trainer.current_epoch} is :",
    #         avg_loss.item(),
    #     )
    #     # print('Validation outputs: ', outputs)
    #     # save to progress bar to plot metrics
    #     self.val_loss = avg_loss
    #     tqdm_dict = {"val_loss": avg_loss}
    #
    #     # save images (Should be the size of the batch?)
    #     in_ = torch.stack([x["input"] for x in outputs])
    #     in_ = in_.cpu().numpy()
    #     out_ = torch.stack([x["out"] for x in outputs])
    #     out_ = out_.cpu().numpy()
    #     targ_ = torch.stack([x["targ"] for x in outputs])
    #     targ_ = targ_.cpu().numpy()
    #
    #     shape = out_.shape  # size of outputs
    #
    #     if shape[0] != 1:
    #         # decided to only choose one image from one GPU
    #         a = np.arange(0, shape[0])
    #         a = np.random.choice(a)
    #     else:
    #         a = 0
    #
    #     self.save_figs(in_[a], out_[a], targ_[a], name="val")
    #
    #     mean_dices = []
    #     mean_hauss = []
    #     step = self.trainer.global_step
    #
    #     for i in range(self.hparams.n_classes + 1):
    #         mean_dices.append(torch.stack([x[f"val_dice_{i}"] for x in outputs]))
    #         mean_hauss.append(torch.stack([x[f"val_haus_{i}"] for x in outputs]))
    #     for i, dice in enumerate(mean_dices):
    #         tqdm_dict[f"val_dice_{i}"] = dice.mean()
    #         # .cpu().numpy()
    #     log_dic = tqdm_dict.copy()
    #     for i, haus in enumerate(mean_hauss):
    #         log_dic[f"val_haus_{i}"] = haus.mean()  # .cpu().numpy()
    #
    #     dices = torch.stack(mean_dices).cpu().numpy()
    #     hauss = torch.stack(mean_hauss).cpu().numpy()
    #
    #     if self.hparams.backend != "ddp":
    #         shape = dices[1].shape
    #         #used to be dices[1] for 'dp'
    #         # print('dices', dices)
    #         print('shape of dices tensor:', shape)
    #         factor = shape[0] * shape[1]
    #     else:
    #         print('Shape of output dice tensor:', dices.shape)
    #
    #     # changed in new version to support multiple and/or binary...
    #     if self.hparams.backend != "ddp":
    #         dice_data = [dice.reshape(1, factor)[0] for i, dice in enumerate(dices)]
    #         hu_data = [hu.reshape(1, factor)[0] for i, hu in enumerate(hauss)]
    #     else:
    #         dice_data = [dice for i, dice in enumerate(dices)]
    #         hu_data = [haus for i, haus in enumerate(hauss)]
    #
    #     rois = ["GTV", "BRAIN","BSTEM","SPCOR","ESOPH","LARYNX","MAND",
    #         "POSTCRI","LPAR","RPAR","LACOU","RACOU","LLAC","RLAC","RRETRO",
    #         "LRETRO","RPLEX","LPLEX","LLENS","RLENS","LEYE","REYE","LOPTIC",
    #         "ROPTIC","LSMAN","RSMAN","CHIASM","LIPS","OCAV","IPCM","SPCM",
    #         "MPCM"]
    #
    #     if self.hparams.oar_version == 16:
    #         idxs = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]
    #     elif self.hparams.oar_version == 1:
    #         idxs = [1]
    #     elif self.hparams.oar_version == 19:
    #         idxs = [3,4,5,6,7,9,10,11,12,17,18,19,20,21,22,23,24,27,28]
    #
    #     chosen = [rois[i - 1] for i in idxs]
    #
    #     plt.switch_backend("agg")
    #     fig1 = plt.figure()
    #     if self.hparams.oar_version == 1:
    #         plt.title("VAL DICE Overlap for OAR Segmentation in HNSCC")
    #     else:
    #         plt.title("VAL DICE Overlap for GTV Segmentation in HNSCC")
    #     plt.boxplot(dice_data[1:])
    #     plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
    #     plt.ylabel("DSC")
    #     plt.xlabel("Target Volume")
    #     self.logger.experiment.add_figure(f"val/dice_plot", fig1, step)
    #
    #     # for now we need the last 3 classes excluding background
    #     fig2 = plt.figure()
    #     if self.hparams.oar_version == 1:
    #         plt.title("Valid HU for GTV Segmentation in HNSCC (mm)")
    #     else:
    #         plt.title("Valid HU for OAR Segmentation in HNSCC (mm)")
    #     plt.boxplot(hu_data[1:])
    #     plt.xticks(range(1, len(idxs) + 1), chosen, rotation="vertical")
    #     plt.ylabel("d(mm)")
    #     plt.xlabel("Target Volume")
    #     self.logger.experiment.add_figure(f"val/hauss_plot", fig2, step)
    #     plt.close()
    #
    #     assert len(mean_dices) == self.hparams.n_classes + 1
    #
    #     for i, dice in enumerate(mean_dices):
    #         print(
    #             f"\n Max Val Dice for class {i} is {dice.mean()}"
    #         ) if self.hparams.verbose else None
    #         print(
    #             f"\n Max Val HausDist for class {i} is {mean_hauss[i].mean()}"
    #         ) if self.hparams.verbose else None
    #
    #     print(f"BOOYA - End of Validation for {self.trainer.current_epoch}.")
    #
    #     return OrderedDict( {"val_loss": avg_loss, "progress_bar": tqdm_dict, "log": log_dic})
