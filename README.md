# PyTorch Lightning 3D OAR Segmentation for Automated Radiation Therapy Planning

Original Repo for "Evaluating clinical acceptability of organ-at-risk segmentation In head & neck cancer using a compendium of open-source 3D convolutional neural networks" (Marsilla et. al, 2022)
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLW2Lp30kdmSflVdw-3Ea48OKP0iPEsnSyMzUFqucyuejpPrTezcSn0vg1ja1iau7KUWAnWdyL1-naaSe6CxOpykJQ0O4ewUmf7lxVjb64xy8pZIXlSvpLhfA_0M6OsBdcIiCFud0bNbYZ9XcxGVIXHamg=w965-h600-no?authuser=0)
![enter image description here](https://lh3.googleusercontent.com/CxJb6Xruah0V1wOD9O1su-5mt69lHtmhK14IYe_oznW7PEYXtZGQnrr-3MjCORJX4-fHnQEwhUeFEhXq2jmGMdQznV4jZ4bdtVrdiqbhpf9qZzCkYEtAQ1CmawCO-vdl2eAGhMx9t7rWnWuPRCGjlWBQqH4n9e0j1I1hzjJl2YDa47Tgrbt09hdyGjgRqRWN-vxquTox6SqiAB2oiIaybWWCODz0smrhAgXkKYluF_dohK3DcKeQNYj01pP24iWMgb5getrjNpsCxdoTbiki7Fvzj3v_T0KWzv5idykLAXdoSgTNXqAKqy4owG51qX7rtVvsYLT7kczzob3cZujXXjPMCP_Uu4Iz1yTnCJjf0wq8NWCtHk5pvoVBE85LXenvfT_TEKdwUippJOYkA2UAa0ggL3nAZvZIQyZG9K6bPMCmKkdBkBStj-5qqraFYJ6EELbVawN2bhCtIw68jx3LrMrzAP2GPFXO3ULuR39obC7-UiJrX3rAEANSpWvtlXVZ8xaOm33APntD8qGkD5UTL6Ces1JUoHrb-0LuwE3LKbmAzn7GDeUAU0tr4vD8y971gc4nOV7BpQ3G8ZuMgwMot-dtVT88R-EVb0rhaF5QfchRzQSYVao7ERtP-uhG1mu5ZEYifB40ZECfcqEyan_LCDP1tsgy_84UaG_J3KNgPW9T1LLo5g6tff_k1RuBllJUox3668HFq8CeEXQdZ_Iw-FsIQw=w535-h265-no?authuser=0)

![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLWhk2cZtKq5yOkWd1YN5rXcolozEKUEWGfeizB5-UnfW-6zdU3ZrNGTMBL9DbCq_D7sX5D5GG9kLL2rZY6rixfb8QjVkr_A84-MAlKS7EAmc1H-bEl3-Xng67G_nQK2H2QOpjLlfoYgBkDQ2Dt3wxT0-Q=w981-h533-no?authuser=0)

## 1. Setup Environment
More instructions [here](https://stackoverflow.com/questions/41274007/anaconda-export-environment-file).

    conda env create -f utils/environment.yml

## 2. Running WOLNET ensemble on custom dataset (no-finetuning required).

In order to run the pre-trained WOLNET network on your own dataset, certain aspects of our Lightning Module must be customized. Because we will be in testing mode, the weights of each model during inference will be frozen.

You can use the slurm templates (wol-oar-test-1.sh ... wol-oar-test-5.sh) as base bash scripts for inference. The only things that need to be edited in this file are:

1. Clone repo.
2. Update local path to model weights (Download them [here](https://drive.google.com/drive/folders/15yc--RfBaxuxKTEHaQ3OwbPdCqTOZdgH?usp=sharing))

3. Update local path to test.py file.
4. If you havn't done so already use custom pre-processing scripts (/process) or our [med-imagetools package](https://github.com/bhklab/med-imagetools) to extract raw data from .dicom images
5. Create a custom .csv with two columns - one for image and one for corresponding mask (if you have avaliable; if un-avaliable use image-path again in the second column). Easy to do in python:

        import pandas as pd
        imgs = glob.glob( "PATH_TO_IMG_FOLDER/*.nrrd")
        # if mask folder avaliable
        msks = [i.replace(PATH_TO_IMG_FOLDER, PATH_TO_MSK_FOLDER) for i in imgs]
        # create .csv we need to load same image twice (in same batch) during inference
        img = []
        msk = []
        for i, im in imgs:
            img.append(im)
            img.append(im)
    	    msk.append(msks[i])
    	    msk.append(msks[i])

       data = {'0':img, '1':msk}
       data = pd.DataFrame.from_dict(data)
       data.to_csv('path_to_csv')

 5. Once we have custom .csv we can run inference (change path in . Once predictions for each model have been exported, use process_masks3.py as a template to average the predictions.
 6. **Optional:** to compare results against ground truth data extracted for your own internal datasets, use notebooks provided.
 7. To establish run your own QUANNOTATE server for clinical acceptability testing of countours generated, please [clone our open-source repo](https://github.com/bhklab/quannotate).
 8. All files including model weights, ensemble predictions on each external dataset as well as pre-processed versions of each dataset can be found [here](https://drive.google.com/drive/folders/15yc--RfBaxuxKTEHaQ3OwbPdCqTOZdgH?usp=sharing).

## 3. Slurm Scripts

PyTorch Lightning has been specifically engineered to optimize training across large computing clusters governed by job handlers. In this study we use slurm commands to run the python script (connecting us to the right GPU nodes). These can be found in the /slurm folder. If you don't use slurm, these files can be run by using simple bash commands.

## Citation

Pending MedRxiv approval.
