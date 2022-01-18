import os, warnings, time, gc, datetime, glob, random
import pandas as pd
from utils import *
from pathlib import Path
import multiprocessing
from argparse import ArgumentParser


def add_args(return_="parser"):

    parser = ArgumentParser()
    arg = parser.add_argument
    # can add multiple arguments from bash script...
    arg("--volume", default="OARS", help="Targes vs OARs")
    arg("--site",default="ALL",help="Only export contours for patient with set primary site...",)
    arg("--dataset", default="radcure", help="Preprocessing dataset as...?")
    arg("--input", default="/cluster/projects/radiomics/Temp/OAR0820", help="Input path as...?")
    arg("--output", default="/cluster/projects/radiomics/RADCURE-images/", help="Output path as...?")

    if return_ == "parser":
        return parser
    else:
        args = parser.parse_args()
        return args


# Lightning Specific GPU Args
# from joblib import Parallel, delayed

args = add_args(return_="args")
inputPath = args.input
outputPath = args.output
dataset = args.dataset  # "radcure"

ts = time.time()

date = datetime.datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H%M%S")
root = Path(f"/cluster/home/jmarsill/EXPORT_{date}")
root_bad = Path(f"/cluster/home/jmarsill/EXPORT_{date}/bad")

# root_data = Path(f"/cluster/home/jmarsill/EXPORT_{date}/data")
# root.mkdir(exist_ok=True, parents=True)
# root_bad.mkdir(exist_ok=True, parents=True)
# root_data.mkdir(exist_ok=True, parents=True)


def save_bad(patient, save_path):

    bad_patient_files = [patient]
    df = pd.DataFrame(bad_patient_files, columns=["Bad_IMGs"])
    df.to_csv(f"{str(save_path)}/{patient}.csv", index=False)
    print(f"{patient} has a buggy in its tummy")


def working():
    laptop = "C://Users/Sejin/Documents/SegmentHN/"  # FOR LAPTOP
    server = "/cluster/home/sejinkim/SegmentHN/"  # FOR H4H
    joeServer = "/cluster/home/jmarsill/ProcessHN/"
    wkdir = laptop if os.path.isdir(laptop) else server
    try:
        os.chdir(wkdir)
        return wkdir

    except:
        os.chdir(joeServer)
        return joeServer


wkdir = working()
print(f"Working on: {wkdir}")
print(f"Starting to process dataset {dataset}")

# process/config/processing-opc1.yml
# process/config/roi_exceptions_opc1.yaml
config, exceptions = getConfig(wkdir, dataset)
print(config)
print(exceptions)
# inputPath = config["DATA_RAW"]
# outputPath = '/cluster/projects/radiomics/Temp/OAR0820/' #config["DATA_OUTPUT"]


def process(folder, bad_path=root_bad, extract=args.volume):
    # processes DICOM to numpy PER PATIENT
    temp, bad = [], ""
    # try:
    print(folder)
    startTime = time.time()
    inputFolder = inputPath + folder
    sub_dir = os.listdir(inputFolder)

    if len(sub_dir) < 2:

        if dataset == "radcure":
            inputFolder += "/" + sub_dir[0]

        if extract == "OARS":

            rois = [
                "GTV",
                "LNECK",
                "RNECK",
                "BRAIN",
                "BSTEM",
                "SPCOR",
                "ESOPH",
                "LARYNX",
                "MAND",
                "POSTCRI",
                "LPAR",
                "RPAR",
                "LACOU",
                "RACOU",
                "LLAC",
                "RLAC",
                "RRETRO",
                "LRETRO",
                "RPLEX",
                "LPLEX",
                "LLENS",
                "RLENS",
                "LEYE",
                "REYE",
                "LOPTIC",
                "ROPTIC",
                "LSMAN",
                "RSMAN",
                "CHIASM",
                "LIPS",
                "OCAV",
                "IPCM",
                "SPCM",
                "MPCM", # "NODES",
            ]

        elif extract == "TARGETS":

            rois = [
                "GTV",
                "LNECK",
                "RNECK",
                "LOWRCTV",
                "MIDRCTV",
                "HIGHRCTV",
            ]

        elif extract == "NODES":
            rois = ["NODES"]

        else:
            rois = ["GTV"]

        dicom = Dicom(inputFolder, dataset=dataset, rois=rois, resample=True)

        warnings.warn(f"Saving {folder} in {outputPath}")
        dicom.export(
            outputPath, folder, exclude=["img", "masks_save"], mode="itk",
            resample=False, slices=False, spacing=None,)
        temp = dicom.roi_select
        print(dicom.spacing, dicom.origin)
        gc.collect()

    else:
        warnings.warn(f"More than one plan/folders found in {folder}...")

    # except Exception as e:
    #
    #     warnings.warn ("Something bad happened...")
    #     warnings.warn(e)
    #     print(e)
    #     bad = folder
    # save_bad(folder, f'{str(bad_path)}')

    endTime = time.time()
    total_time = startTime - endTime
    print("Total Processing Time:", total_time)

    return bad, folder, temp


def main(inputPath, outputPath, exception_dict):
    print(inputPath, outputPath)
    os.makedirs(outputPath, exist_ok=True)
    args = add_args(return_="args")
    start = time.time()
    print("hi", gc.isenabled())
    # print (inputPath, outputPath, exception_dict)
    # get contours data types, we're looking for CTV & L/R CTV
    dicom_folders = get_immediate_subdirectories(inputPath)
    dicom_folders = sorted(dicom_folders)
    dicom_folders = [str(i) for i in dicom_folders]
    print("Total:", len(dicom_folders))

    try:
        # if the masks are already processed, do not recompute...
        # completed = os.listdir (outputPath + "/masks")
        # this works regularly, just debugging...
        completed = glob.glob(outputPath + "/masks/*.nrrd")
        # should be outputPath..
        # completed = glob.glob('/cluster/projects/radiomics/Temp/NODES/masks/*.nrrd')
        completed = [i.split("/")[-1].partition(".")[0] for i in completed]
        print(f"Already Completed {len(completed)} images.")

    except:
        completed = []

    dicom_folders = [i for i in dicom_folders if i not in completed]
    random.shuffle(dicom_folders)
    print(f"First value found: {dicom_folders[0]}")
    site = args.site
    print("site: ", args.site)
    # "Nasopharynx"
    vals = ["ALL", "CUSTOM"]
    if site not in vals:
        data = pd.read_csv(
            "/cluster/home/jmarsill/Lightning/data/valid_mrns_by_dssite_new2.csv"
        )
        folders = data[site]
        folders = [str(int(x)) for x in folders if str(x) != "nan"]
        dicom_folders = [i for i in dicom_folders if str(int(i)) in folders]
        print(f"Number of patients in {site} is {len(dicom_folders)}.")

    if site == "CUSTOM":

        completed = glob.glob(outputPath + "/masks/*.nrrd")
        # should be outputPath..
        # completed = glob.glob('/cluster/projects/radiomics/Temp/NODES/masks/*.nrrd')
        completed = [i.split("/")[-1].partition(".")[0] for i in completed]
        print(f"Already Completed {len(completed)} images.")

        # folders = [677007, 4097139, 754753, 293788, 557196, 569989, 958461, 844880,
        #            438291, 593166, 936812, 962089, 519307, 707021, 919362, 595939,
        #            807582, 937535, 167966, 663581, 743655, 976744, 732668, 106580,
        #            672003, 297646, 878521, 640328, 223108, 726899, 413044, 816576,
        #            553942, 812274, 847536, 198305, 797330, 760849, 181341, 649574,
        #            282399]
        # Uopdated Feb 11th, 2021...
        # folders = ['2084569', '4268718', '3051481', '4122804', '3669785', '2450020',
        #            '4037219', '3881909', '0776915', '3834952', '3405642', '3391243',
        #            '2221469', '0620955', '4212677', '3722836', '4223563', '0321621', '4153776']
        folders = ['3861111', '4329819', '4321095', '3670519', '3176262', '3720433', '4238916', '3839173']

        dicom = [str(i) for i in folders]
        dicoms = []
        for path in dicom:
            if len(path) == 6:
                s = '0' + path
                dicoms.append(s)
            else:
                dicoms.append(path)
        dicom_folders = [str(i) for i in dicoms if str(i) not in completed]
        random.shuffle(dicom_folders)
        # dicom_folders = [i for i in dicom_folders if i not in completed]

    # taken from /cluster/home/jmarsill/Lightning/data/valid_mrns_by_dssite_new2.csv

    print(f"Only {len(dicom_folders)} remaining.")
    print("Done:", len(completed))
    print("To do:", len(dicom_folders))

    # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cores)(delayed(process)(inputPath, outputPath, folder) for folder in dicom_folders)
    p = multiprocessing.Pool()
    results = p.map(process, dicom_folders)
    print(results)

    # save logs
    info, problem = Info(), BadFiles()
    for bad, patient, names in results:
        print(bad, patient, names)
        info.add_patient(patient, names)
        problem.add_bad(bad)

    os.makedirs(outputPath, exist_ok=True)
    info.export(outputPath)  # uh fix this lol
    problem.export(outputPath)

    print(f"Script ran for {round((time.time()-start), 2)} seconds")

    return


if __name__ == "__main__":
    main(inputPath, outputPath, exceptions)  # on desktop
