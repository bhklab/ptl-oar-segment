import os, warnings, time, gc, datetime, glob, multiprocessing, torch, nrrd
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

def process(b, extract=args.volume, cpath='/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_STRUCTSEG_19'):

    # os.mkdir(cpath+'/FINAL_')
    temp, bad = [], ""
    folders=glob.glob(cpath+'/FOLD_*')
    folders.sort()
    imgs = glob.glob(folders[0]+'/*_RAW*')
    # for b in range(len(imgs)):
    im = None
    for i, fold in enumerate(folders):
        # img_ = torch.tensor(np.load(fold+f'/outs_{b}_RAW.npy'))
        img_=torch.tensor(nrrd.read(fold+f'/outs_{b}_RAW.nrrd')[0])
        if im is None:
            im = img_
        else:
            im = torch.stack([im, img_])
            im = torch.mean(im, dim=0)
        print(f'LOADED {i}')

    print(im.size())
    im = torch.argmax(im, dim=0)
    print(im.size()) # correlate to original input size
    center = np.load(fold+f'/center_{b}.npy')
    # orig = np.load('./RAW/'+f'/input_{b}_FULL.npy')
    orig = nrrd.read(f'{cpath}/RAW/input_{b}_FULL.nrrd')[0]
    orig = torch.zeros(orig.shape)
    orig[center[0]:center[1], center[2]:center[2]+292,center[3]:center[3]+292] = im
    orig = orig.cpu().numpy() # np.save(f'./FINAL_/outs_{b}_FULL.npy', orig)
    nrrd.write(f'{cpath}/FINAL_/outs_{b}_FULL.nrrd', orig)
    print(f'Done {b}')

        # os.makedirs(cpath, exist_ok=True)
        # # processes DICOM to numpy PER PATIENT
        # dic = {}
        # # print(folder)
        # pat = path.split('/')[-1].partition('.')[0]
        # val = range(34)
        # a = []
        # image = nrrd.read(path)[0]
        # for v in val:
        #  if v in image:
        #    a.append(1)
        #  else:
        #    a.append(0)
        # a = np.array(a)
        # print('Done ', sum(a==1), pat)
        # np.save( f'/cluster/projects/radiomics/Temp/joe/RADCURE_joe/conditional_tensors/{pat}.npy', a)

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

    return bad, b, temp

def main(inputPath, outputPath):

    # os.mkdirs('/cluster/projects/radiomics/Temp/joe/OAR0820/bad')
    print(inputPath, outputPath)
    args = add_args(return_="args")
    start = time.time()
    print("hi", gc.isenabled())
    # print (inputPath, outputPath, exception_dict)
    # get contours data types, we're looking for CTV & L/R CTV
    # dicom_folders = glob.glob('/cluster/projects/radiomics/Temp/joe/RADCURE_joe/masks_compressed/*.nrrd')
    # done = glob.glob('/cluster/projects/radiomics/Temp/joe/RADCURE_joe/conditional_tensors/*.npy')
    # done = [d.split('/')[-1].partition('.')[0] for d in done]
    # dicom_folders = [fold for fold in dicom_folders if fold.split('/')[-1].partition('.')[0] not in done]
    # dicom_folders = sorted(dicom_folders)
    # print(f"Done {len(done)} folders. Only {len(dicom_folders)} remaining.")
    cpath = '/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_STRUCTSEG_19'
    # cpath='/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_TCIA_HNSCC'
    # cpath='/cluster/projects/radiomics/Temp/joe/OAR-TESTING/AI_dataset23' # Next ONE...
    # os.mkdir(cpath+'/FINAL_', exist_ok=True)
    folders=glob.glob(cpath+'/FOLD_*')
    folders.sort()
    imgs = glob.glob(folders[0]+'/*_RAW*')
    dicom_folders=range(len(imgs))
    p = multiprocessing.Pool()
    results = p.map(process, dicom_folders)
    print(results)
    print(f"Script ran for {round((time.time()-start), 2)} seconds")
    return


if __name__ == "__main__":
    main(inputPath, outputPath)  # on desktop
