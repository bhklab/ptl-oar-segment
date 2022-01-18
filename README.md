# PyTorch Lightning 3D OAR Segmentation for Automated Radiation Therapy Planning

Original Repo for "Evaluating clinical acceptability of organ-at-risk segmentation In head & neck cancer using a compendium of open-source 3D convolutional neural networks" (Marsilla et. al, 2022)
![enter image description here](https://lh3.googleusercontent.com/pw/AM-JKLW2Lp30kdmSflVdw-3Ea48OKP0iPEsnSyMzUFqucyuejpPrTezcSn0vg1ja1iau7KUWAnWdyL1-naaSe6CxOpykJQ0O4ewUmf7lxVjb64xy8pZIXlSvpLhfA_0M6OsBdcIiCFud0bNbYZ9XcxGVIXHamg=w965-h600-no?authuser=0)
![enter image description here](https://lh3.googleusercontent.com/CONQ5Jfm5GHbdZF5GkO0IDCFn2RhXxLQep34QzaOVyO43QUCOW0BoO08emMtw5yeG_N1bWifkK-QZXb9euF2pAewHpFb_Q_sWsjBoIO3LYCe6YlcYKT0O_AAlWrRXlSUW6XV4X0UKEnJKaqf8gAAFLM1KfmOWEpDq-jIZvcGEsM7vP0ao5Dwqlh0brK-s4JvuiUnTI1qAEY9OCahNi4x5CfACyohE-qjXSG6xZtjmoU84ZUK-D23VAPdeerPA7Xh3qm-HXDgzJ2_GbC8kt6ckOTuh3s2WapuOQMVrrKXEl20DX7VGd_2s8NHRiTx5qnEO6Bs0BWMSsQ9CvT62JtaSu99FSKXgDqKaPjRqcmHNPLm_fE5U2W3Syh9gvb6sSznGN97Wls3xj96huQtj-S0QS_WgLarwQ9TG0PbhWSfpLXnbtyQRtGD1RRyTaKboiqjiTQA2WdQNu6Z1wGr54QHtBhowR7KK9RJRtif_QdCPJsmOj1oGlTrYp-C6_IjMERgztLGjalprmsBNQu9O8s8HdkkI_0ex8GxFZ1J7JvACgERLeT3uL4sK6u5oaYpE6W1QgOpthoy1aW3R6rQgyF9LNjz5mpn9ZsBrA8mataL-PwFiiH8nXId7HUKVk5oC0rtq0XIYNoMh10SB95m6Pl6x9QTGeIrw5xg5BnhGlJ-OKIoNyE9ACYwCH5YqD2UbhSLOmmKna83vj7ggMUscYexxO0RQw=w1211-h600-no?authuser=0)

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
