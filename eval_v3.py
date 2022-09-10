import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import pandas as pd
import SimpleITK as sitk
from skimage.io import imread
import monai.metrics as monmet
import torch

if __name__ == '__main__':
    test_data_folder = '/h/jiananc/RADCURE/pseudo_test_data'
    prediction_folder = '/h/jiananc/nnunet/test_pred'

    ROIS = [ "GTVp", "External", "LCTVn", "RCTVn", "Brainstem", "Esophagus",
            "Larynx", "Cricoid_P", "OpticChiasm", "Glnd_Lacrimal_L",
            "Glnd_Lacrimal_R", "Lens_L", "Lens_R", "Eye_L", "Eye_R",
            "Nrv_Optic_L", "Nrv_Optic_R", "Parotid_L", "Parotid_R",
            "SpinalCord", "Mandible_Bone", "Glnd_Submand_L",
            "Glnd_Submand_R", "Cochlea_L", "Cochlea_R", "Lips",
            "Spc_Retrophar_R", "Spc_Retrophar_L", "BrachialPlex_R",
            "BrachialPlex_L", "BRAIN", "OralCavity", "Musc_Constrict_I",
            "Musc_Constrict_S", "Musc_Constrict_M"]

    oars = []
    dice = []
    haus = []
    asds = []
    eval = []
    pats = []
    p_idx = []

    pid = 0
    for patient in os.listdir(test_data_folder):
        ID = patient.split('_')[0]
        pred_mask_path = os.path.join(prediction_folder, ID+'.nii.gz')
        pred_mask = sitk.ReadImage(pred_mask_path)
        pred_mask_array = sitk.GetArrayFromImage(pred_mask)

        folder = os.path.join(test_data_folder, patient)
        image_path = os.path.join(folder, 'CT_IMAGE.nrrd')
        image = sitk.ReadImage(image_path)
        image_arr = sitk.GetArrayFromImage(image)
        structure_folder = os.path.join(folder, 'structures')

        image_path = os.path.join(folder, 'CT_IMAGE.nrrd')
        image = sitk.ReadImage(image_path)
        image_arr = sitk.GetArrayFromImage(image)
        structure_folder = os.path.join(folder, 'structures')

        for i, roi in enumerate(ROIS):
            try:
                roi_path = os.path.join(structure_folder, roi + '.nrrd')
                roi_image = sitk.ReadImage(roi_path)
                if i == 0:
                    roi_array = sitk.GetArrayFromImage(roi_image)
                else:
                    new_array = sitk.GetArrayFromImage(roi_image)
                    roi_array += new_array * (i+1)
            except:
                print(f'File not found for {patient}_{roi}')
            # to exclude external
        try:
            if image_arr.shape==roi_array.shape:
                roi_array[roi_array>35]=0
            exist_ROIs = np.unique(roi_array)
            if i != 1 and (i+1) in exist_ROIs:
                roi_gt = torch.from_numpy((roi_array == i+1).astype(int))
                roi_pred = torch.from_numpy((pred_mask_array == i+1).astype(int))
                roi_gt = roi_gt[None, None,:]
                roi_pred = roi_pred[None, None,:]
                ###############################
                dc = monmet.compute_meandice(roi_pred, roi_gt)
                h = monmet.compute_hausdorff_distance(roi_pred, roi_gt,
                                                   percentile=95, include_background=False)
                s = monmet.compute_average_surface_distance(roi_pred, roi_gt,
                                                         include_background=False)
                # print(self.patient, c, dc, h)
                # save metrics...
                oars.append(roi)
                dice.append(dc[0][0].item())
                haus.append(h[0][0].item())
                asds.append(s[0][0].item())
                eval.append(100*dc[0][0].item() / (h[0][0].item() + s[0][0].item()))
                pats.append(ID)
                p_idx.append(pid)
        except:
            print(f'Failed for {patient}_{roi}.')
        pid+=1

    data = {"ID": p_idx, "PATIENT": pats, "OAR": oars, "DICE": dice, "95HD": haus, "ASD": asds, "EVAL": eval}
    eval_data = pd.DataFrame.from_dict(data)
    eval_data.to_csv(f"{prediction_folder}/UNetFTW_test.csv")






