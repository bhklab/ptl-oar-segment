"""
Runs a presaved model on a single node across N-gpus.
See https://williamfalcon.github.io/pytorch-lightning/
"""
import os, torch, warnings, glob
import numpy as np
from utils import SegmentationModule, config
from pytorch_lightning import Trainer # , seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint

"""
.. warning:: `logging` package has been renamed to `loggers` since v0.7.0.
 The deprecated package name will be removed in v0.9.0.
"""

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())
def main(args):

    """
    Main training routine specific for this project.
    Participents can define modify the base SegmentationModule that will be
    given in the offical repo for the challenge (utils - modules.py).
    Checkpointing is built into boilerplate, weights must be loaded into your
    custom SegmentationModule object. Please provide your code in the format
    given in the repo. You are cable to add/change any part of the code given
    in the repo including losses, models and optimizers.
    Please zip your utils folder and send it along with your model weights saved
    duirng checkpointing.

    For your reference following script will be used for inference. This will be
    using test_step function in modules.py. NOTE: Adaptive sliding window(s) will
    be used during evaluation, something similar to swi() in utils.py.

    Note: empty boilerplate will be delivered before challenge begins
    on the offical project repo.

    :param hparams: weights_path
    
    """
    # implement override with user passed path
    # assert args.weights_path is not None
    warnings.warn('Using presaved weights...')
    # inference for custom model...
    weights_path = "/cluster/projects/radiomics/Temp/joe/models-1222/WOLNET_2023_01_31_190855/weights_CUSTOM/"
    checkpoints = glob.glob(weights_path + "*.ckpt")
    for checkpoint in checkpoints:
        warnings.warn(f'Loading save model from {checkpoint}.')
        model = SegmentationModule.load_from_checkpoint(checkpoint_path=checkpoint)
        trainer = Trainer(gpus=1, strategy='ddp',
                          default_root_dir=model.hparams.root)
        trainer.test(model)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    args = config.add_args(return_='args')
    main(args)
