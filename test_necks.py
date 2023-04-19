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
    weights_path = "/cluster/projects/radiomics/Temp/joe/models-1222/WOLNET_2023_04_12_141555/neckTEST/" # "/cluster/projects/radiomics/Temp/joe/models-1222/WOLNET_2023_01_31_190855/weights_NECKLEVELS/"
    checkpoints = glob.glob(weights_path + "*.ckpt")
    # hparams = glob.glob(weights_path + "*.yaml")
    model = SegmentationModule()
    trainer = Trainer(gpus=1, default_root_dir=model.hparams.root, resume_from_checkpoint=checkpoints[0])
    trainer.test(model)
    
    # checkpoints.sort()
    # hparams.sort()
    # if we have the ensemble working...
    # for i, checkpoint in enumerate(checkpoints):
    #     warnings.warn(f'Loading save model from {checkpoint}.')
    #     checkpoint_ = torch.load(checkpoint)
    #     model_weights = checkpoint_["state_dict"]
    #     # update keys by dropping `auto_encoder.`
    #     for key in list(model_weights):
    #         model_weights[key.replace("criterion.", "").replace("ce.weight", "")]=model_weights.pop(key)
    #         # model_weights[key.replace("ce.weight", "")] = model_weights.pop(key)
    #         warnings.warn(f'Dropping key {key} from checkpoint.')
        
    #     try: 
    #         model_weights.pop("")
    #     except Exception:
    #         warnings.warn(f'Checkpoint {checkpoint} does not contain any out of order model weights.')
    #         pass
        
    #     checkpoint_["state_dict"] = model_weights
    #     # update checkpoint 
    #     torch.save(checkpoint_, checkpoint)
    #     # save model weights
    #     torch.save(model_weights, weights_path + f"weights_{i+1}.ckpt")
    #     model = SegmentationModule.load_from_checkpoint(checkpoint_path=checkpoint) 
    #     trainer = Trainer(gpus=1, default_root_dir=model.hparams.root)
    #     trainer.test(model)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    args = config.add_args(return_='args')
    main(args)
