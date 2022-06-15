import pandas as pd
import numpy as np
import nrrd, glob

# given a list of paths
paths = ["/Users/joemarsilla/vector/sample/SAD0001/",
         "/Users/joemarsilla/vector/sample/SAD0002/"]
image_paths = [p+"CT_IMAGE.nrrd" for p in paths]
structure_paths = [p+"structures/*" for p in paths]
folders_ = [glob.glob(p) for p in structure_paths]
structure_paths
# taken from prepare.py in utils
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

# first step get ROI order
def getHeaderData(folders, structures=True, roi_order=getROIOrder()):
    voxel_dic = {}
    img_dic = {}
    # com_dic = {}
    for list in folders:
        for i, p in enumerate(list):
            oar = p.split('/')[-1].partition('.')[0]
            class_idx = roi_order[oar]
            header = nrrd.read_header(p)
            voxels = header["Voxels"]
            try:
                data = voxel_dic[oar]
                voxel_dic[oar] = (data + voxels)/2.
                if i==0:
                    mean = img_dic["meanHU"]
                    std = img_dic["stdHU"]
                    img_dic = {"meanHU":(mean+header["meanHU"])/2.,
                               "stdHU":(std+header["stdHU"])/2}

                # can do the same thing if com data was in mask...
                # data = com_dic[oar]
                # com_dic[oar] = (data + voxels)/2.

            except Exception:
                voxel_dic[oar] = voxels
                if i==0:
                    img_dic = {"meanHU":header["meanHU"],
                               "stdHU":header["stdHU"]}

                # com_dic[oar] = com

    return {"VOXINFO":voxel_dic, "IMGINFO":img_dic}

data = getHeaderData(folders_)
data

#########################
# make dataset splits...
import pandas as pd
import glob
csvs = glob.glob("./config/wolnet_ensemble_splits/*")
len(csvs)
#########################
data = pd.read_csv(csvs[3])
data.head()
dat = pd.read_csv("./config/radcure_mapping_final.csv")
#########################
mrns = list(data['2'])
len(mrns)
mrns
dat[dat["MRN"].isin(mrns)]
dat[dat["MRN"]==672003]
###############################################################################
## Testing
###############################################################################
# test header extraction functions...
# mapping = pd.read_csv("/Users/joemarsilla/ptl-oar-segmentation/config/radcure_OPC_mapping.csv")
# mapping.head()
# mrn = list(mapping["MRN"])
# new_id = list(mapping["RADCURE"])
#
# mapping_ = pd.read_csv("/Users/joemarsilla/ptl-oar-segmentation/config/tcia_submission.csv")
# mapping_.head()
# mrn+= list(mapping_["Local Patient ID"])
# new_id += list(mapping_["Anonymized Patient ID"])
#
# data_mapping = {"MRN":mrn, "NEWID":new_id}
# data= pd.DataFrame.from_dict(data_mapping)
# data.head()
# len(data)
# data.to_csv("/Users/joemarsilla/ptl-oar-segmentation/config/new_radcure_mapping.csv")
# # every other MRN not in this list, assign random RADCURE_VALUE
# data
