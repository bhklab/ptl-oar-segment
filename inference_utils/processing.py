import warnings
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

def padTensor(img):
  '''
  padding image if less than 112 in both dimensions...
  pad 3rd to last dim if below 112 with MIN value of image
  use only if padding required...
  '''
  shape = img.size()
  a, diff = (None, None)
  if shape[1]<112:
      difference = 112 - shape[1]
      a = difference//2
      diff = difference-a
      pad_ = (0,0,0,0,a,diff)
      img = F.pad(img, pad_, "constant", img.min())

  return img, (a,diff)

def getCropZ(img):
    # we'd like to crop the img in the z plane
    # assumes first and last eight of image are fluff

    shape = img.size()
    if 180<=shape[1]:
        cropz = (shape[1]//12, shape[1]-shape[1]//12)
    elif 165 <= shape[1] < 180:
        diff = 180 - shape[1]
        cropz = (shape[1]//13, shape[1]//13+152-diff)
    else:
        cropz = (0, shape[1])
    img = img[:,cropz[0]:cropz[1]]

    return img, cropz

def new_crop(img, crop_factor, com=None):

    if not isinstance(img, np.ndarray):
        img = img.numpy()

    if len(img.shape) == 4:
        img = img[0]
    if len(img.shape) == 5:
        img = img[0][0]
    
    if com is None:
        com_z, _, _ = ndimage.center_of_mass(img)
        com_z = com_z if com_z > 0 else 1
        img_ = img[int(com_z)]
        com_y, com_x = ndimage.center_of_mass(img_)
    else:
        _, com_y, com_x = com

    startx = int(com_x) - (crop_factor // 2) - 1
    starty = int(com_y) - (crop_factor // 2) - 1

    startx = startx if startx > 0 else 1
    starty = starty if starty > 0 else 1

    img = img[:, starty: starty + crop_factor, startx: startx + crop_factor]

    return img, (starty, startx)


def fix_outputs_necks(out_:np.array, val:int=292):

    # LevelIB...
    # right side...
    sl = out_[:,:,:val//2]
    sl[sl==8] = 2
    out_[:,:,:val//2] = sl
    # left side...
    sl = out_[:,:,val//2:]
    sl[sl==2] = 8
    out_[:,:,val//2:]=sl

    # # LEVELIII
    sl = out_[:,:,:val//2]
    sl[sl==9] = 3
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==3] = 9
    out_[:,:,val//2:]=sl

    # LEVELII
    sl = out_[:,:,:val//2]
    sl[sl==10] = 4
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==4] = 10
    out_[:,:,val//2:]=sl

    # LEVELIV
    sl = out_[:,:,:val//2]
    sl[sl==11] = 5
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==5] = 11
    out_[:,:,val//2:]=sl

    # LEVELVIIA
    sl = out_[:,:,:val//2]
    sl[sl==12] = 6
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==6] = 12
    out_[:,:,val//2:]=sl

    # LEVELV
    sl = out_[:,:,:val//2]
    sl[sl==13] = 7
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==7] = 13
    out_[:,:,val//2:]=sl

    return out_

def fix_outputs(out_:np.array, val:int=292):

    # Parotids...
    sl = out_[:,:,:val//2]
    sl[sl==6] = 7
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==7] = 6
    out_[:,:,val//2:]=sl

    # # Acoustics...
    sl = out_[:,:,:val//2]
    sl[sl==8] = 9
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==9] = 8
    out_[:,:,val//2:]=sl

    # Plexus
    sl = out_[:,:,:val//2]
    sl[sl==11] = 10
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==10] = 11
    out_[:,:,val//2:]=sl

    # Lens
    sl = out_[:,:,:val//2]
    sl[sl==12] = 13
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==13] = 12
    out_[:,:,val//2:]=sl

    # Eyes
    sl = out_[:,:,:val//2]
    sl[sl==14] = 15
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==15] = 14
    out_[:,:,val//2:]=sl

    # Eyes
    sl = out_[:,:,:val//2]
    sl[sl==14] = 15
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==15] = 14
    out_[:,:,val//2:]=sl

    # # Optic Nerves...
    sl = out_[:,:,:val//2]
    sl[sl==16] = 17
    out_[:,:,:val//2] = sl
    sl = out_[:,:,val//2:]
    sl[sl==17] = 16
    out_[:,:,val//2:]=sl

    return out_
