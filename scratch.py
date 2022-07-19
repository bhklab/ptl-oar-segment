import pandas as pd
import numpy as np
import glob
from .utils import *

images = glob.glob("/storage/data/ml2022/RADCURE_VECTOR/*")
folders = [glob.glob(img+"/structures/*") for img in images]
order = getROIOrder()
weight = np.sum(np.array(list(vals.values()))+1)
valid_pats = []
oars = list(vals.keys())s
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