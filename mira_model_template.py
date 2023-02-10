import logging
import platform
import sys
from typing import Tuple
import numpy as np
from utils import SegmentationModule, config

# packages that were dependent for model training
from lightning import Trainer, seed_everything
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import SimpleITK as sitk

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

# (x, y, z)
Point = Tuple[float, float, float]


__MODEL_NAME__ = "WOLNET_ENSEMBLE"
__MODEL_VERSION__ = "0.0.1"

    
    
def run_masks(model, exam_image: sitk.Image) -> dict[str, sitk.Image]:
    # need origin, spacing, direction
    return {
        "A": sitk.Image(),
        "B": sitk.Image(),
        "C": sitk.Image()
    }

def run_points(model, exam_image: sitk.Image) -> dict[str, [[Point]]]:
    return {
        "A": [[(0, 0, 0)]],
        "B": [[(0, 0, 0)]],
        "C": [[(0, 0, 0)]]
    }


def load_model(model_path: str):
    
    trainer = Trainer(gpus=1, default_root_dir='/content/drive/My Drive/ptl-oar-segmentation/') # strategy='ddp_notebook'
    
    return trainer 

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

    # model set up
    trainer = load_model('/path/to/model')
    
    # process image(s)
    # exam_image = sitk.ReadImage('/path/to/exam')

    contour_data = run_masks(model, exam_image)
    # OR
    contour_points = run_points(model, exam_image)
