"""
Runs a model on a single node across N-gpus.

See
https://williamfalcon.github.io/pytorch-lightning/

"""
import os, torch, warnings
import numpy as np
from utils import SegmentationModule, config
from pytorch_lightning import Trainer, callbacks#, seed_everything

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
    Main training routine specific for this project
    :param hparams:
    """
    assert args.weights_path is not None
    warnings.warn('Using presaved weights...')
    warnings.warn(f'Loading save model from {args.weights_path}.')
    model = SegmentationModule.load_from_checkpoint(checkpoint_path=args.weights_path) #, strict=False) #,

    # put this true if loading from previously saved networks...
    # checkpoint_callback = callbacks.ModelCheckpoint( monitor="val_loss",
                                                     # mode="min",
                                                     # save_last=True,
                                                     # save_top_k=3,
                                                     # verbose=False)

    trainer = Trainer(gpus=1, strategy='ddp')
            # distributed_backend=args.backend,
            # default_root_dir=model.hparams.root,
            # max_epochs=model.hparams.n_epochs,
            # weights_summary='top',
            # log_gpu_memory='min_max',
            # callbacks=[checkpoint_callback])
            # resume_from_checkpoint=args.weights_path)

    trainer.test(model) # , ckpt_path=args.weights_path)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    args = config.add_args(return_='args')
    main(args)
