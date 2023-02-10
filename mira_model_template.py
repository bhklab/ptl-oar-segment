import logging, platform, torch, os, glob, nrrd
from typing import Tuple
import numpy as np
from utils import SegmentationModule, ROIS, custom_order
# packages that were dependent for model training
from lightning import Trainer, seed_everything
from torch.utils.data import Dataset
# from pytorch_lightning.callbacks import ModelCheckpoint
import SimpleITK as sitk

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

# (x, y, z)
# Point = Tuple[float, float, float]

__MODEL_NAME__ = "WOLNET_ENSEMBLE"
__MODEL_VERSION__ = "0.0.1"
__MODEL_PATH__ = ""
    
def run_masks(save_path:str, roi_list:list, custom_order:list) -> dict[str, sitk.Image]:
    
    os.mkdir(save_path)  # 'wolnet-sample/FINAL_'
    folders = glob.glob('wolnet-sample/FOLD_*')
    folders.sort()
    imgs = glob.glob(folders[0]+'/*_RAW*')
    for b in range(len(imgs)):
        im = None
        for i, fold in enumerate(folders):
            # img_ = torch.tensor(np.load(fold+f'/outs_{b}_RAW.npy'))
            img_ = torch.tensor(nrrd.read(fold+f'/outs_{b}_RAW.nrrd')[0])
            if im is None:
                im = img_
            else:
                im = torch.stack([im, img_])
                im = torch.mean(im, dim=0)
            print(f'LOADED {i}')
        print(im.size())
        im = torch.argmax(im, dim=0)
        print(im.size())  # correlate to original input size
        center = np.load(fold+f'/center_{b}.npy')
        # orig = np.load('./RAW/'+f'/input_{b}_FULL.npy')
        orig = nrrd.read('wolnet-sample/RAW/'+f'/input_{b}_FULL.nrrd')[0]
        orig = torch.zeros(orig.shape)
        orig[center[0]:center[1], center[2]:center[2] +
            292, center[3]:center[3]+292] = im
        orig = orig.cpu().numpy()  # np.save(f'./FINAL_/outs_{b}_FULL.npy', orig)
        # saves image in nrrd format
        nrrd.write(f'wolnet-sample/FINAL_/outs_{b}_FULL.nrrd', orig)
        print(f'Done {b}')
        # one image or multiple...
    
    patient_masks = {}
    names = [roi_list[i] for i in custom_order]
    for i, name in enumerate(names):
        sample = orig.copy()
        sample[sample != i+1] = 0
        sample[sample == i+1] = 1
        patient_masks[name] = sitk.GetImageFromArray(sample)
        warnings.warn(f'Saving {name}')

    return patient_masks

# def run_points(model, exam_image: sitk.Image) -> dict[str, [[Point]]]:
#     return {
#         "A": [[(0, 0, 0)]],
#         "B": [[(0, 0, 0)]],
#         "C": [[(0, 0, 0)]]
#     }

# be sure to save images in wolnet-sample folder in ptl-oar-segmentation directory...
# can be modified to fit the paths that you set...
def run_model(model_path: str, patient:str) -> dict[str, sitk.Image]:
    
    # ensure model_path is location of ptl_oar_segmentation folder...
    # inference may require GPU(s) depending on avaliability adjust gpus/cpus flag acordingly
    trainer = Trainer(gpus=1, default_root_dir=model_path) # strategy='ddp_notebook'
    splits = glob.glob(model_path+"/wolnet-sample/weights/*.ckpt")
    for split in splits:
        model = SegmentationModule.load_from_checkpoint(checkpoint_path=split)
        # model takes in two inputs per batch, for the sake of inference we will load the same image twice
        # be sure the patient folder is in the following format:
        # wolnet-sample/patient_1/
        #                        ...CT_IMAGE.nrrd
        #                        ...structures/mask_1.nrrd
        # if we need to modify it a differeny way
        model.test = [patient, patient]
        trainer.test(model)
    
    patient_masks = run_masks('wolnet-sample/FINAL_', ROIS, custom_order)

    return patient_masks

if __name__ == "__main__":
    # init logging
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    run_meta = {
        "PythonVersion": sys.version,
        "Platform": platform.platform(),
        "ModelName": __MODEL_NAME__,
        "ModelVersion": __MODEL_VERSION__,
    }

    logger.info(f"meta: {run_meta}")
    # id of patient folder save in wolnet-sample...
    contour_data = run_model(__MODEL_PATH__, patient_id)

    # model set up
    # trainer = load_model('/path/to/model')
    # process image(s)
    # exam_image = sitk.ReadImage('/path/to/exam')
    # contour_data = run_masks(model, exam_image)
    # OR
    # contour_points = run_points(model, exam_image)
