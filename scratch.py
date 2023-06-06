import pandas as pd
import numpy as np
import glob
from .utils import *

images = glob.glob("/storage/data/ml2022/RADCURE_VECTOR/*")
folders = [glob.glob(img+"/structures/*") for img in images]
order = getROIOrder()
weight = np.sum(np.array(list(vals.values()))+1)
valid_pats = []
oars = list(vals.keys())
no_structs = []
for i, fold in enumerate(folders):
    if len(fold) == 0:
        no_structs.append(i)
    else:
        pat = fold[0].split("/")[-3]
        structs = [f.split("/")[-1].partition(".")[0] for f in fold]
        count = 0
        for s in structs:
            if s in oars:
                count += vals[s]+1

        if count == weight:
            valid_pats.append(pat)


### code to update file header data...
import nrrd, numpy as np, pandas as pd, glob
from scipy.ndimage import binary_fill_holes, center_of_mass

def segment_head(img):
    # function to make (fake) external for center cropping of image...
    try:
        img = img.cpu().numpy()
    except Exception as e:
        pass
    
    otsu = threshold_otsu(img)  # Compute the Ostu threshold
    binary_img = np.array(img > otsu, dtype=int )  # Convert image into binary mask (numbers bigger then otsu set to 1, else 0)
    fill = binary_fill_holes(binary_img)  # Fill any regions of 0s that are enclosed by 1s
    
    return fill

def updateImg(img_path):
    img = nrrd.read(img_path+"/CT_IMAGE.nrrd")
    head = img[1]
    img = img[0]
    # mask patient region...
    mask = segment_head(img)
    # calculage center of mass
    com = center_of_mass(mask)
    img_ = img.copy()
    img_[img_<-500]=-500
    img_[img_>1000]=1000
    # crop according to center of mass
    img_ = img[int(com[0])-56:int(com[0])+56, int(com[1])-88:int(com[1])+88,int(com[2])-88:int(com[2])+88]
    # save new data into header
    head["center_of_mass"] = com
    head["mean_after_crop"] = np.mean(img_)
    head["std_after_crop"] = np.std(img_)
    # re-save original image
    nrrd.write(img_path+"/CT_IMAGE.nrrd", img, header=head)
    print(f"Saved {img_path}")
    return np.mean(img_), np.std(img_)