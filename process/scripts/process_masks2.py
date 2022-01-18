import os, warnings, time, gc, datetime, glob, multiprocessing
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import SimpleITK as sitk, numpy as np, nrrd

def add_args(return_="parser"):

    parser = ArgumentParser()
    arg = parser.add_argument
    # can add multiple arguments from bash script...
    arg("--volume", default="OARS", help="Targes vs OARs")
    arg("--site",default="ALL",help="Only export contours for patient with set primary site...",)
    arg("--dataset", default="radcure", help="Preprocessing dataset as...?")
    arg("--input", default="/cluster/projects/radiomics/Temp/joe/OAR0820", help="Input path as...?")
    arg("--output", default="/cluster/projects/radiomics/RADCURE-images/", help="Output path as...?")

    if return_ == "parser":
        return parser
    else:
        args = parser.parse_args()
        return args

args = add_args(return_="args")
inputPath = args.input
outputPath = args.output
dataset = args.dataset
os.makedirs('/cluster/projects/radiomics/Temp/joe/RC/', exist_ok=True)
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H%M%S")

def save_bad(patient, save_path='/cluster/projects/radiomics/Temp/joe/RC/bad'):
    os.makedirs(save_path, exist_ok=True)
    bad_patient_files = [patient]
    df = pd.DataFrame(bad_patient_files, columns=["Bad_IMGs"])
    df.to_csv(f"{str(save_path)}/{patient}.csv", index=False)
    print(f"{patient} has a buggy in its tummy")

def working():

    laptop = "C://Users/Sejin/Documents/SegmentHN/"  # FOR LAPTOP
    server = "/cluster/home/sejinkim/SegmentHN/"  # FOR H4H
    joeServer = "/cluster/home/jmarsill/SegmentHN/process/"
    wkdir = laptop if os.path.isdir(laptop) else server

    try:
        os.chdir(wkdir)
        return wkdir

    except:
        os.chdir(joeServer)
        return joeServer

wkdir = working()
print(f"Working on: {wkdir}")
# config, exceptions = getConfig(wkdir, dataset)
# print(config)
# print(exceptions)

def process(path, extract=args.volume, cpath='/cluster/projects/radiomics/Temp/joe/RADCURE_joe/conditional_tensors2/'):

    os.makedirs(cpath, exist_ok=True)
    # processes DICOM to numpy PER PATIENT
    temp, bad = [], ""
    dic = {}
    # print(folder)
    pat = path.split('/')[-1].partition('.')[0]
    val = range(35)
    a = []
    image = nrrd.read(path)[0]
    for v in val:
     if v in image:
       a.append(1)
     else:
       a.append(0)
    a = np.array(a)
    print('Done ', sum(a==1), pat)
    np.save( f'/cluster/projects/radiomics/Temp/joe/RADCURE_joe/conditional_tensors2/{pat}.npy', a)

        # img = sitk.ReadImage(path)
        # img = sitk.GetArrayFromImage(img)
        # for i in range(2):
        #     if len(img[img==i]) > 0:
        #         counts.append(len(img[img==i]))
        #     else:
        #         counts.append(0)
        #
        # img = sitk.ReadImage(path.replace('masks', 'images'))
        # img = sitk.GetArrayFromImage(img)
        # img[img<-500] = -500
        # img[img>1000] = 1000
        # metrics = [np.mean(img), np.std(img)]
        # dic[0] = [counts, metrics]
        # data = pd.DataFrame.from_dict(dic)
        # data.to_csv(f'{cpath}/{patient}.csv')

    return bad, pat, temp

def main(inputPath, outputPath):

    # os.mkdirs('/cluster/projects/radiomics/Temp/joe/OAR0820/bad')
    print(inputPath, outputPath)
    args = add_args(return_="args")
    start = time.time()
    print("hi", gc.isenabled())
    # print (inputPath, outputPath, exception_dict)
    # get contours data types, we're looking for CTV & L/R CTV
    cpath='/cluster/projects/radiomics/Temp/joe/RADCURE_joe/conditional_tensors2/'
    os.makedirs(cpath, exist_ok=True)
    dicom_folders = glob.glob('/cluster/projects/radiomics/Temp/joe/RADCURE_joe/masks_compressed/*.nrrd')
    done = glob.glob('/cluster/projects/radiomics/Temp/joe/RADCURE_joe/conditional_tensors2/*.npy')
    done = [d.split('/')[-1].partition('.')[0] for d in done]
    dicom_folders = [fold for fold in dicom_folders if fold.split('/')[-1].partition('.')[0] not in done]
    dicom_folders = sorted(dicom_folders)
    print(f"Done {len(done)} folders. Only {len(dicom_folders)} remaining.")
    p = multiprocessing.Pool()
    results = p.map(process, dicom_folders)
    print(results)
    print(f"Script ran for {round((time.time()-start), 2)} seconds")
    return


if __name__ == "__main__":
    main(inputPath, outputPath)  # on desktop
