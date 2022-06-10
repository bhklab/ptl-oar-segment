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
        Uses train_dataloader class to send batch at every step during trainig
        Need to define custom Dataset class for dataloader.
        return {'loss':loss}
        """


    # ---------------------
    # Run Validation Step, Runs after Trainning Epoch converges
    # This can be modulated in Trainer() when running train.py
    # ---------------------
    def validation_step(self, batch, batch_idx):
        """
        Uses valid_dataloader class to send batch at every step during validation
        Need to define custom Dataset class for dataloader.
        use validation_epoch_end to save images to logger...
        return {'loss':loss}
        """

    def test_step(self, batch, batch_idx):
         """
        Uses train_dataloader class to send batch at every step during inference
        Need to define custom Dataset class for dataloader.
        return {'loss':loss}
         """

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
        # classes = self.hparams.n_classes + 1
        self.net = {INSERT HERE}

    # ------------------
    # Assign Loss
    # ------------------
    def __get_loss(self):

        # self.class_weights this must be provided and calculated separately...
        # class should be located in self.data_config...
        # usually this will be the amount of voxels given for any OAR class...
        self.class_weights = self.data_config["weights"]
        assert len(self.class_weights) == self.hparams.n_classes + 1
        self.criterion = {INSERT HERE}

    def get_dataloader( self, df, mode="valid", transform=None, resample=None,
                        shuffle=False, transform2=None, batch_size=None):

        dataset = {INSERT HERE}

        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        # best practices to turn shuffling off during validation...
        validate = ['valid', 'test']
        shuffle = False if mode in validate else True

        return DataLoader( dataset=dataset, num_workers=self.hparams.workers,
                           batch_size=batch_size, pin_memory=True, shuffle=shuffle,
                           drop_last=True,)

    def train_dataloader(self):

        transform = {INSERT HERE}

        # add transform to dataloader
        return self.get_dataloader(df=self.train_data, mode="train", transform=transform,
                                   resample=False, batch_size=self.hparams.batch_size,)

    # @pl.data_loader
    def val_dataloader(self):
        # imported from transform.py
        transform = {INSERT HERE}

        return self.get_dataloader( df=self.valid_data, mode="valid",
                                    transform=transform,  # should be default
                                    resample=False, batch_size=self.hparams.batch_size,)
    # @pl.data_loader
    def test_dataloader(self):
        # during inference we will run each model on the test sets according to
        # the data_config which you will provide which each model...
        transform = {INSERT HERE}

        return self.get_dataloader( df=self.test_data, mode="test",transform=transform, # transform,  # should be default
                                    transform2=None, resample=self.hparams.resample,
                                    batch_size=self.hparams.batch_size,
        )
