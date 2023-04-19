"""
Runs a model on a single node across N-gpus.

See
https://williamfalcon.github.io/pytorch-lightning/

"""
import os, torch, warnings
import numpy as np
from utils import SegmentationModule, config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

"""
.. warning:: `logging` package has been renamed to `loggers` since v0.7.0.
 The deprecated package name will be removed in v0.9.0.
"""

SEED = 234
# torch.manual_seed(SEED)
# np.random.seed(SEED)
seed_everything(SEED, workers=True)
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())

def main(args):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # try:
    #     
    #     warnings.warn('Using presaved weights...')
    #     warnings.warn(f'Loading save model from {args.weights_path}.')
    # assert args.weights_path is not None
    # warnings.warn(f'Loading save model from {checkpoint}.')
    # checkpoint_ = torch.load(checkpoint)
    # model_weights = checkpoint_["state_dict"]
    # # update keys by dropping `auto_encoder.`
    # for key in list(model_weights):
    #     model_weights[key.replace("criterion.", "").replace("ce.weight", "")]=model_weights.pop(key)
    #     # model_weights[key.replace("ce.weight", "")] = model_weights.pop(key)
    #     warnings.warn(f'Dropping key {key} from checkpoint.')
        
    # try: 
    #     model_weights.pop("")
    # except Exception:
    #     warnings.warn(f'Checkpoint {checkpoint} does not contain any out of order model weights.')
    #     pass
   
    model = SegmentationModule(args) # .load_from_checkpoint(args.weights_path)
    # trainer = Trainer(resume_from_checkpoint=args.weights_path)
    # except Exception:
    #     warnings.warn('Using randomized weights...')
    #     model = SegmentationModule(args)
    checkpoint_callback = ModelCheckpoint( monitor="val_loss",
                                           filename=str(args.model + '-epoch{epoch:02d}-val_loss{val/loss:.2f}'),
                                           auto_insert_metric_name=False,
                                           mode="min",
                                           save_last=True,
                                           save_top_k=3,)
    trainer = Trainer(
            accelerator="gpu",
            devices=-1, # set to -1 to use all avaliable gpus...
            strategy='ddp', # should be same as args.backend..., # stochastic_weight_avg=True, # pass to callbacks if required...
            reload_dataloaders_every_n_epochs=1,
            limit_train_batches=0.2,
            # limit_train_batches=0.6,
            limit_val_batches=0.2,
            default_root_dir=model.hparams.root,
            max_epochs=model.hparams.n_epochs,
            # log_gpu_memory='min_max',
            sync_batchnorm=True,
            # precision=16,
            accumulate_grad_batches={150:2, 400:4},#2, # changing this parameter affects outputs
            callbacks=[checkpoint_callback],
            # checkpoint_callback=checkpoint_callback)# < 1.4.0
            resume_from_checkpoint=args.weights_path)

    # ------------------------
    # 3 START TRAINING
    # ------------------------

    trainer.fit(model)

if __name__ == '__main__':

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # import args to add to basic lightning module args
    args = config.add_args(return_='args')

    main(args)
