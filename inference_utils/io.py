import SimpleITK as sitk
import numpy as np
import torch, warnings, os
from .processing import fix_outputs

def getImage(path:str="./0522c0014/CT_IMAGE.nrrd"):
    """
    Reads image at path
    Args:
        path (str): path of image
    Returns:
        tensor with shape (1, D, H, W)
    """
    
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)

    img = torch.from_numpy(img)
    img = img.unsqueeze(0)

    return img

def saveImage(pred:torch.tensor, img:sitk.Image, path:str="./scratch/"):
    if not os.path.exists(path):
        os.mkdir(path)

    _, x, _ = pred.size()
    pred = pred.numpy()
    pred = fix_outputs(pred,x)

    labels = ['Brainstem','SpinalCord','Esophagus','Larynx','Mandible_Bone','Parotid_L','Parotid_R','Acoustic_L','Acoustic_R',
              'BrachialPlex_R','BrachialPlex_L','Lens_L','Lens_R','Eye_L','Eye_R','Nrv_Optic_L','Nrv_Optic_R','OpticChiasm','Lips']

    for idx, label in enumerate(labels):
        slc = np.copy(pred)
        slc[slc != idx+1] = 0
        slc = sitk.GetImageFromArray(slc)
        sitk.WriteImage(slc, os.path.join(path, f'{label}.nrrd'), useCompression=True)
    
    pred = sitk.GetImageFromArray(pred)
    sitk.WriteImage(pred, os.path.join(path, f'Full.nrrd'), useCompression=True)

    warnings.warn(f"Saved model outputs in {path}")