import os, glob, random, warnings, pickle, cv2, nrrd
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import scipy.ndimage.measurements as measure
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

np.random.seed(7)

ROIS = ["External", "GTVp", "LCTVn", "RCTVn", "Brainstem", "Esophagus",
        "Larynx", "Cricoid_P", "OpticChiasm", "Glnd_Lacrimal_L",
        "Glnd_Lacrimal_R", "Lens_L", "Lens_R", "Eye_L", "Eye_R",
        "Nrv_Optic_L", "Nrv_Optic_R", "Parotid_L", "Parotid_R",
        "Trachea", "SpinalCord", "SpinalCanal", "Mandible_Bone", "Glnd_Submand_L",
        "Glnd_Submand_R", "Cochlea_L", "Cochlea_R", "Lips",
        "Spc_Retrophar_R", "Spc_Retrophar_L", "BrachialPlex_R", "BrachialPlex_L",
        "Brain", "Glnd_Pituitary", "OralCavity", "Musc_Constrict_I",
        "Musc_Constrict_S", "Musc_Constrict_M"]

def getROIOrder(custom_order=None, rois=ROIS):
    # ideally this ordering has to be consistent to make inference easy...
    if custom_order is None:
        order_dic = {roi:i for i, roi in enumerate(ROIS)}
    else:
        roi_order = [rois[i] for i in custom_order]
        order_dic = {roi:i for i, roi in enumerate(roi_order)}
    return order_dic

# testing the getROIOrder function > can use this to rearange ordering of mask loading...
# this was the custom ordering used to develop the model in question...
# custom_order = [4,20,5,6,22,17,18,25,26,30,31,11,12,13,14,15,16,8,27]
# v_ = getROIOrder(custom_order=custom_order)

class LoadPatientVolumes(Dataset):
    def __init__(self, folder_data, data_config, external=False):
        """
        This is the class that our Dataloader object
        will take to create batches for training.
        param: image_data : This is a dataframe with image path,
                            and folder location of structures
        param: data_config: Dictionary with all loading and
        """
        self.data = folder_data
        self.config = data_config

    def __len__(self):
        return len(self.data)

    def load_data(self, idx):
        """
        Loading of the data...
        """
        self.patient = str(self.data.iloc[idx][0])
        self.img_path = self.config["data_path"] + f'/{self.patient}/CT_IMAGE.nrrd'
        self.structure_path = self.config["data_path"] + f'/{self.patient}/structures/'
        # can write your own custom finction to load in structures here...
        # assumes directory structure where patient name is enclosing folder...
        custom_order = self.config['roi_order']
        self.order_dic = getROIOrder(custom_order=custom_order)
        self.load_nrrd()

    def load_nrrd(self):
        # load image using nrrd...
        warnings.warn('Using nrrd instead of sitk.')
        self.img = nrrd.read(self.img_path)
        self.img = self.img[0].transpose(2,0,1)
        shape = self.img.shape
        # Loading with SITK...
        # mask = sitk.ReadImage(img_path)
        # img = sitk.ReadImage(img_path)
        # self.img = sitk.GetArrayFromImage(img)
        # self.mask = sitk.GetArrayFromImage(mask)
        # Generate MASK according to order defined in config
        # Can also define custom_order in data_config...
        # mask_path should be list to folder of structures
        # names should match the labels given in ROI list above...
        if mask_path[-1] != '/':
            mask_path+= '/'
        mask_paths = glob.glob(self.structure_path + "*")
        # version 1 ...
        # set empty mask... useful for weightedtopkCE loss
        self.mask = np.zeros(shape)
        # version 2 ... for other losses one hot encoding might be required
        # self.mask = []
        for path in mask_paths:
            oar = path.split('/')[-1].partition('.')[0]
            class_value = self.order_dic[oar]
            mask = nrrd.read(path)
            mask = mask.transpose(2,0,1)
            assert mask.shape == shape
            # version 1 > not one hot encoded
            self.mask[mask>0] = class_value

    def __getitem__(self, idx):

        self.load_data(idx)
        if self.transform is not None:
            if self.mask.max() > 0:
                self.img, self.mask = self.transform(self.img.copy(), self.mask.copy())
                if self.transform2 is not None:
                    img2, _ = self.transform2(self.img.copy(), self.mask.copy())
            else:
                # only load if mask is zero from start...
                warnings.warn(f'Check {self.patient}...')
                assert self.mask.max() > 0

        assert self.mask.max() > 0

        img = torch.from_numpy(self.img).type(torch.FloatTensor)
        mask = torch.from_numpy(self.mask).type(torch.LongTensor)

        if self.transform2 is not None:
            img2 = torch.from_numpy(img2).type(torch.FloatTensor)
            return img2, img, mask
        else:
            return img, mask
