import torch, warnings, nrrd
import numpy as np
from skimage import measure, morphology
import string, random, glob

# folders_ = [glob.glob(p) for p in structure_paths]
# taken from prepare.py in utils
# These are the ROIs that must be segmented
# ROIS= [ "GTVp", "LCTVn", "RCTVn", "Brainstem", "Esophagus",
#         "Larynx", "Cricoid_P", "OpticChiasm", "Glnd_Lacrimal_L",
#         "Glnd_Lacrimal_R", "Lens_L", "Lens_R", "Eye_L", "Eye_R",
#         "Nrv_Optic_L", "Nrv_Optic_R", "Parotid_L", "Parotid_R",
#         "SpinalCord",  "Mandible_Bone", "Glnd_Submand_L",
#         "Glnd_Submand_R", "Cochlea_L", "Cochlea_R", "Lips",
#         "Spc_Retrophar_R", "Spc_Retrophar_L", "BrachialPlex_R", "BrachialPlex_L",
#         "BRAIN",  "OralCavity", "Musc_Constrict_I",
#         "Musc_Constrict_S", "Musc_Constrict_M"]

ROIS = ["External", "GTVp", "LCTVn", "RCTVn", "Brainstem", "Esophagus",
        "Larynx", "Cricoid_P", "OpticChiasm", "Glnd_Lacrimal_L",
        "Glnd_Lacrimal_R", "Lens_L", "Lens_R", "Eye_L", "Eye_R",
        "Nrv_Optic_L", "Nrv_Optic_R", "Parotid_L", "Parotid_R",
        "SpinalCord", "Mandible_Bone", "Glnd_Submand_L",
        "Glnd_Submand_R", "Cochlea_L", "Cochlea_R", "Lips",
        "Spc_Retrophar_R", "Spc_Retrophar_L", "BrachialPlex_R",
        "BrachialPlex_L", "BRAIN", "OralCavity", "Musc_Constrict_I",
        "Musc_Constrict_S", "Musc_Constrict_M"]

# this was the original order used for the OG segmentation study...
# custom_order = [4,5,6,8,11,12,13,14,15,16,17,18,20,22,25,26,27,30,31]
#################################
# original ROI standardization...
# rois = ["GTV", "BRAIN","BSTEM","SPCOR","ESOPH","LARYNX","MAND",
#     "POSTCRI","LPAR","RPAR","LACOU","RACOU","LLAC","RLAC","RRETRO",
#     "LRETRO","RPLEX","LPLEX","LLENS","RLENS","LEYE","REYE","LOPTIC",
#     "ROPTIC","LSMAN","RSMAN","CHIASM","LIPS","OCAV","IPCM","SPCM",
#     "MPCM"]
#################################

def getROIOrder(tag, rois=ROIS, inverse=False, include_external=True):
    # ideally this ordering has to be consistent to make inference easy...
    custom_order = getCustomOrder(tag)
    if custom_order is None:
        if include_external is False:
            order_dic = {roi:i for i, roi in enumerate(ROIS)} if inverse is False else {i:roi for i, roi in enumerate(ROIS)}
        else:
            order_dic = {roi:i+1 for i, roi in enumerate(ROIS)} if inverse is False else {i+1:roi for i, roi in enumerate(ROIS)}
    else:
        roi_order = [rois[i] for i in custom_order]
        order_dic = {roi:i+1 for i, roi in enumerate(roi_order)} if inverse is False else {i+1:roi for i, roi in enumerate(roi_order)}
    return order_dic

##########################
# first step get ROI order
##########################
def getCustomOrder(tag):
    if tag == "NECK":
        # includes GTV...
        custom_order = [1,2,3]
    elif tag == "BRACP":
        custom_order = [29,28]
    elif tag == "NECKMUS":
        custom_order = [32,33,34]
    elif tag == "SPINE":
        custom_order = [5,4,6,19]
    elif tag == "TOPHEAD":
        custom_order = [30,8,11,12,13,14,15,16]
    elif tag == "MIDHEAD":
        custom_order = [9,10,17,18,19,20,21,22,23,24]
    elif tag == "OTHER":
        custom_order = [7,31,4,25,26,27,5,6]
    else:
        custom_order=custom_order
        warnings.warn("Tag not specified...using general ordering.")
        # will load in custom_order in utils.py...
    return custom_order

def getHeaderData(folders, structures=True, tag=None):
    roi_order=getROIOrder(tag)
    assert len(folders) > 1
    voxel_dic = {}
    img_dic = {}
    # com_dic = {}
    oars_used = list(roi_order.keys())
    for list_ in folders:
        for i, p in enumerate(list_):
            oar = p.split('/')[-1].partition('.')[0]
            if oar in oars_used:
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
                    img_dic = {"meanHU":header["meanHU"],
                               "stdHU":header["stdHU"]}
            else:
                warnings.warn(f"{oar} not in list of chosen ROIS. If this is a mistake please update roi_order.")
                pass
                # com_dic[oar] = com
    return {"VOXINFO":voxel_dic, "IMGINFO":img_dic}

def swi(image, net, n_classes):

    # in our case we're looking for a 5D tensor...
    if len(image.size()) < 5:
        image = image.unsqueeze(0)

    shape = image.size()
    warnings.warn(f'Image shape is {image.size()}')

    # Z ...
    start = [0, shape[2]//4, shape[2]-shape[2]//4 - 112, shape[2]-112]
    end = [112, shape[2]//4+112, shape[2]-shape[2]//4, shape[2]]
    # Y
    start_y = [0, shape[3]-176, shape[3]//12, shape[3]//6, shape[3] - shape[3]//4 - 176, shape[3] - shape[3]//6 - 176, shape[1] - shape[1]//12 - 176]
    end_y = [176, shape[3], shape[3]//12 + 176, shape[3]//6 + 176, shape[3] - shape[3]//4, shape[3] - shape[3]//6, shape[1] - shape[1]//12]
    # X
    start_x = [0, shape[4]-176, shape[4]//12, shape[4]//4, shape[4]//6, shape[4] - shape[4]//4 - 176, shape[4] - shape[4]//6 - 176, shape[4] - shape[4]//12 - 176]
    end_x = [176, shape[4], shape[4]//12 + 176, shape[4]//4+176, shape[4]//6 + 176, shape[4] - shape[4]//4, shape[4] - shape[4]//6, shape[4] - shape[4]//12]
    output_shape = (2, n_classes, shape[2], shape[3], shape[4])

    reference_ = torch.zeros(output_shape).to(image.device)
    reference = torch.zeros(output_shape).to(image.device)
    iter_=0
    for i, val in enumerate(start):
        for j, v in enumerate(start_y):
            for k, va in enumerate(start_x):
                im = image[:,:,val:end[i], v:end_y[j], va:end_x[k]]
                sh = im.size() # shoud be 5D tensor...
                if (sh[1], sh[2], sh[3], sh[4])!= (2, 112, 176, 176):
                    pass
                else:
                    reference_[:,:,val:end[i], v:end_y[j], va:end_x[k]]+=1
                    # RUN THE NETWORK
                    im=im[0]
                    warnings.warn(f'NET IN SIZE IS {im.size()}')
                    output=net(im)
                    warnings.warn(f'NET OUTS SIZE IS {output.size()}')
                    reference[:,:, val:end[i], v:end_y[j], va:end_x[k]] += output
                    iter_ += 1
    warnings.warn(f'Iterated {iter_} times with sliding window.')
    return reference/reference_

#takes in an image slice and returns a list of all contours indexed by the class labels
def get_contours(img, num_classes, level):
    contour_list = []
    for class_id in range(num_classes):
        out_copy = img.copy()
        out_copy[out_copy != class_id] = 0
        outed = out_copy
        contours = measure.find_contours(outed, level)
        contour_list.append(contours)
    return contour_list

#Create contour set for the entire volume: expected input in a volume of size (Z,Y,X)
#label_names are the names corresponding to each label index in model output ["background", "brain", "optic chiams", ...]
#                                                                               0            1           2       etc
#len(label_names) should equal num_classes
#Return format:
# {
#   "Brain":
#   [
#     [(x, y, z1), (x, y, z1), (x, y, z1), (x, y, z1)], # slice 1
#     [(x, y, z2), (x, y, z2), (x, y, z2), (x, y, z2)], # slice 2
#     [(x, y, z3), (x, y, z3), (x, y, z3), (x, y, z3)], # slice 3
#     .
#     .
#     .
#     [(x, y, zn), (x, y, zn), (x, y, zn), (x, y, zn)] # slice n
#   ]
# }
# Test notebook is on local

def numpy_to_contour(arr, num_classes, level, label_names):
    #create empty dict
    contour_set = {}
    for label in label_names:
        contour_set[label] = []

    #Go though each slice
    for z_value, arr_slice in enumerate(arr):
        contours_list = get_contours(arr_slice, num_classes, level) #get all contours for this slice

        #Go through each label's contours
        for label, site_contours in enumerate(contours_list):
            #Go through each contour in that specific label and append the z value
            for contour_id, contour in enumerate(site_contours):
                contours_list[label][contour_id] = np.insert(contours_list[label][contour_id], 2, z_value, axis=1) #add z value into contour coordinates

        #Append into the dictionary
        for i, label in enumerate(label_names): #for each organ
            for array in contours_list[i]:
                array = array.tolist() #convert from numpy array to python list
                contour_set[label].append(array)
                #contour_set[label] = np.append(contour_set[label], array)

    #can use json.dump() to save this as a json file
    return contour_set

##OLD
# def numpy_to_contour(arr, num_classes, level, label_names):
#
#     #create empty dict
#     contour_set = {}
#     for label in label_names:
#         contour_set[label] = []
#
#     #Go though each slice
#     for z_value, arr_slice in enumerate(arr):
#         contours_list = get_contours(arr_slice, num_classes, level) #get all contours for this slice
#
#         #Go through each label's contours
#         for label, site_contours in enumerate(contours_list):
#             #Go through each contour in that specific label and append the z value
#             for contour_id, contour in enumerate(site_contours):
#                 contours_list[label][contour_id] = np.insert(contours_list[label][contour_id], 2, z_value, axis=1) #add z value into contour coordinates
#
#         #Append into the dictionary
#         for i, label in enumerate(label_names):
#             for array in contours_list[i]:
#                 contour_set[label] = np.append(contour_set[label], array.flatten())
#
#     #can use json.dump() to save this as a json file
#     return contour_set
