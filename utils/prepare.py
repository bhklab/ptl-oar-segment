import os, glob, random, warnings, pickle, cv2, nrrd
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import scipy.ndimage.measurements as measure
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

np.random.seed(7)
radcure_path = "/cluster/projects/bhklab/RADCURE/img"
mask_path = "/cluster/projects/bhklab/RADCURE/masks"

class GetDirs:
    def __init__(self, path):
        """
        param: path - this is a directory path.
        """
        self.a_path = path

    def from_sub(self):
        """
        get_dirs.sub() will export a list of all folder names of sub_dirs
        from parent dir in self.path.
        """
        return [
            name
            for name in os.listdir(self.a_path)
            if os.path.isdir(os.path.join(self.a_path, name))
        ]

    def from_file(self):

        patients = glob.glob(self.a_path + "/*.npy")
        return [
            patient_path.split("/")[-1].partition(".")[0] for patient_path in patients
        ]

    def __len__(self):
        return len(GetDirs.sub(self))


class GetSplits:
    def __init__(
        self,
        args,
        mask_path=mask_path,
        mrn_path=None,
        metrics_name="METRICSGNECK_2020_02_18_100330",
        site="Oropharynx",
        volume_type="targets",
        mode="default",
        from_dir=False,
        nfolds=6,
        test_split=1,
        to_filter=True,
        height=128,
        width=200,
        classes=3,
        dir_path="/home/gpudual",
    ):
        """
        param: nfolds - number of folds to split dataset
        param: test_split - .9 (10% of patients will be saved as independent
                test set and will not be used for training or validation)
        param: mode == default
            Expect(s) path == directory with patient folders.
        param:  mode == csv
            Expects(s) path to .csv of folder name (indexed)
        """
        self.path = mask_path
        self.by_mrns = mrn_path
        self.nfolds = nfolds
        self.test_split = test_split
        self.mode = mode
        self.from_dir = from_dir
        self.site = site
        self.args = args
        self.tofilter = to_filter
        self.height = height
        self.width = width
        self.new_metrics = None
        self.metrics_name = metrics_name
        self.volume_type = volume_type
        self.n_classes = classes
        self.dir_path = dir_path
        self.split()

    def patients(self):
        # list of sorted patient folders...
        if self.mode == "default":

            folders = GetDirs(self.path).from_file()
            random.shuffle(folders)
            warnings.warn(f"Using files from {self.path}.")
            warnings.warn(f"Threre are {len(folders)} patients.")

        elif self.mode == "csv":

            try:
                # over ride default...
                # import subs|et of images used for initial training of model(s)
                warnings.warn(f"Using CSV of Patient IDs instead of image folder.")
                csv_path = f"{self.dir_path}/SegmentHN/archive/data/valid_mrns_by_dssite_new2.csv"
                if os.path.isfile(csv_path) is True:
                    # this is a list of folders in defined disease site that
                    # we have masks for.
                    data = pd.read_csv(f"{self.dir_path}/SegmentHN/archive/data/valid_mrns_by_dssite_new2.csv")
                    if self.site == "ALL":
                        listed = list(data)[1:]
                        folders = []
                        for lst in listed:
                            folders += list(data[lst])
                    else:
                        folders = data[self.site]

                else:
                    warnings.warn( "No .csv file at that path. Please fix that Dumbo. (--mrn-csv-path)")

                # taken from /cluster/home/jmarsill/SegmentHN/data/valid_mrns_by_dssite_new2.csv

                # define image path...
                folders = [str(int(x)) for x in folders if str(x) != "nan"]

                if self.tofilter == True:
                    print("HA")
                    print("Using new resampled data.")
                    # filter for the patients in the dataset most recently processed...
                    # was METRICSGNECK_2020_02_18_100330

                    try:
                        self.new_metrics = self.load_obj(
                            f"{self.dir_path}/SegmentHN/archive/data/{self.metrics_name}.pkl"
                        )
                        print(f"Only {len(self.new_metrics)} avaliable.")
                        assert self.new_metrics is not None

                    except Exception:
                        warnings.warn('Loading from default data object.')
                        if self.volume_type == 'targets':
                            self.new_metrics = self.load_obj(
                                f"{self.dir_path}/SegmentHN/archive/data/METRICSGTV_2020_05_18_124515_2708.pkl"
                                # METRICSGNECK_2020_02_18_100330.pkl"
                            )

                        else:
                            self.new_metrics = self.load_obj(
                                f"{self.dir_path}/SegmentHN/archive/data/METRICSOAR_2020_05_13_190628_582.pkl"
                                # METRICSGNECK_2020_02_18_100330.pkl"
                            )

                    filter_folders = list(self.new_metrics.keys())
                    folders = [x for x in filter_folders if str(int(x)) in folders]
                    print(f"There are {len(folders)} patients.")

                else:
                    self.new_metrics = None
                    mask_pths = glob.glob(self.path + "/*.npy")
                    mask_ids = [
                        msk.split("/")[-1].partition(".")[0] for msk in mask_pths
                    ]
                    folders = [x for x in folders if x in mask_ids]
                    folders = list(folders)

                random.shuffle(folders)

            except:
                warnings.warn("If no csv with patient_ids, use mode=default")
                pass

        elif self.mode == "csv_full" and self.site == "ALL":

            try:
                warnings.warn("Using all patient images.")
                warnings.warn(f"Using CSV of Patient IDs instead of image folder.")
                if os.path.isfile(self.by_mrns):
                    csv_path = self.by_mrns
                else:
                    raise TypeError(
                        "No .csv file at that path. Please fix that Dumbo. (--mrn-csv-path)"
                    )

                data = pd.read_csv(csv_path)
                print("csv", data)
                assert data
                # define image path...
                mask_pths = glob.glob(self.path + "/*.npy")
                mask_ids = [msk.split("/")[-1].partition(".")[0] for msk in mask_pths]
                names = list(data)[1:]
                edited = names[: len(names) - 1]

                # import bad image paths...
                bad_path = (
                    "/cluster/home/jmarsill/archive/SegmentHN/outputs/BAD_RADGCTV3.csv"
                )

                try:
                    bad_data = pd.read_csv(bad_path)
                    bad_values = list(bad_data.values[:, 0])
                    bad = [str(x) for x in bad_values]

                except Exception:
                    bad = []

                if self.tofilter == True:
                    print("Using new resampled data.")
                    # filter for the patients in the dataset most recently processed...
                    # METRICSGNECK_2020_02_18_100330
                    try:
                        self.new_metrics = self.load_obj(
                            f"{self.dir_path}/SegmentHN/archive/data/{self.metrics_name}.pkl"
                        )
                        filter_folders = list(self.new_metrics.keys())
                        print(filter_folders)
                        assert self.new_metrics is not None

                    except Exception:
                        print("Failed to load data object...")
                        self.new_metrics = self.load_obj(
                            f"{self.dir_path}/SegmentHN/archive/data/METRICSGNECK_2020_02_18_100330.pkl"  # default can be set to anthing...
                        )

                # establish 5 classes, everything else give unknown label
                classes = [edited[1], edited[2], edited[3], edited[4], edited[5]]
                folders = []
                indexes = []

                for name in edited:
                    if name in classes:
                        idx = classes.index(name)
                    else:
                        idx = 5

                    indexes.append(idx)
                    folders_ = data[
                        str(name)
                    ]  # only used when parsing RADCURE dataset...
                    folders_ = [str(int(x)) for x in folders_ if str(x) != "nan"]

                    if self.tofilter == True:
                        print(name)
                        folders_ = [
                            str(x) for x in filter_folders if str(int(x)) in folders_
                        ]
                        print(len(folders_))
                    else:
                        folders_ = [x for x in folders_ if x not in bad]
                        folders_ = [x for x in folders_ if x in mask_ids]

                    # will only append patients whose mask exist and who are not in the bad_csv above...
                    folders_ = list(folders_)
                    random.shuffle(folders_)
                    folders.append(folders_)

                self.folders = folders
                self.indexes = indexes

            except:
                warnings.warn("If no csv with patient_ids, use mode=default")
                pass

        self.folders = folders

        return folders

    def export_metrics_old(self):
        # This set will be used for KFOLD vlaidation
        train_folds = []
        valid_folds = []

        # make this another parameter which can be changed on the fly
        obj_means = self.load_obj(
            f"{self.dir_path}/SegmentHN/archive/data/RADCURE_means_2019_11_28_204023.pkl"
        )
        obj_counts = self.load_obj(f"{self.dir_path}/SegmentHN/archive/data/COUNTS.pkl")

        means = []
        stds = []
        weights_ = []

        # assumes training_folders is a list of all the possible folders
        # that we'd be training on...

        for train_idx, test_idx in self.kf.split(self.training_folders):
            X_train, X_test = (
                [self.training_folders[x] for x in train_idx],
                [self.training_folders[y] for y in test_idx],
            )
            train_folds.append(X_train)
            # export mean, std for training set for each fold
            mean = 0
            std = 0
            ctv = 0
            gtv = 0
            back = 0

            for pat in X_train:

                mean += obj_means[pat][2][0]
                std += obj_means[pat][2][1]

                # calculate class imbalance
                weights = obj_counts[pat]
                total_slices = weights[0]
                total_px = (
                    total_slices * 256 * 256
                    if self.args.model != "UNET"
                    else 64 * 200 * 200
                )  # 256*256 # 64 slices in the volume model.
                total_gtv = weights[3]
                total_ctv = weights[2]
                total_background = (
                    total_px - total_gtv - total_gtv
                    if self.args.model != "VNET"
                    else weights[1]
                )

                # find single class frequency
                # normalize by total count
                gtv += 1.0 / (total_gtv / total_px)
                ctv += 1.0 / (total_ctv / total_px)
                back += 1.0 / (total_background / total_px)

            # normalize for all patients in training set
            gtv /= len(X_train)
            ctv /= len(X_train)
            back /= len(X_train)
            weights = np.array([back, ctv, gtv])

            means.append(mean / len(X_train))
            stds.append(std / len(X_train))
            weights_.append(list(weights))

            # append validation folds
            valid_folds.append(X_test)

        # can save this separately and use to iterate through
        # check if split folds correctly...
        assert len(train_folds) == len(valid_folds)

        # we will use these values when validating also
        self.means, self.stds, self.weights = means, stds, weights_
        self.train, self.valid = train_folds, valid_folds

    def export_metrics_new(self):
        print("Processing Resampled Data.")
        # automatically will export weights for all 3 classes...
        # This set will be used for KFOLD vlaidation
        train_folds = []
        valid_folds = []

        # this will be same length as train_folds
        means = []
        stds = []
        weights_ = []

        # assumes training_folders is a list of all the possible folders
        # that we'd be training on...

        for train_idx, test_idx in self.kf.split(self.training_folders):

            X_train, X_test = (
                [self.training_folders[x] for x in train_idx],
                [self.training_folders[y] for y in test_idx],
            )

            # append training folds
            a = np.int(len(X_test) * 0.5)
            X_train += X_test[:a]
            train_folds.append(X_train)
            # append validation folds
            # only really need half for validation...
            Test = X_test[a:]
            valid_folds.append(Test)

            # export mean, std for training set for each fold
            mean_ = np.zeros(len(X_train))
            std_ = np.zeros(len(X_train))

            weights = np.zeros((self.n_classes, len(X_train)))

            for i, pat in enumerate(X_train):
                metrics = self.new_metrics[str(pat)]

                if str(metrics[0]) != "nan":
                    mean_[i] = np.int(metrics[0])
                    std_[i] = np.int(metrics[1])

                # background class weight
                # set crop factor to
                size = self.height * (self.width ** 2)
                total_size = metrics[2][0] * metrics[2][1] * metrics[2][2] * 2
                # assumes GTV is captured in the volume
                # skips CTV...
                if self.volume_type == "targets":
                    choose=[0,1]
                else:
                    # organs at risk...
                    # for the smaller version...
                    # originally will train on 704 patients across 18 OAR classes...
                    # remove GTV add submandib...
                    choose = [0,3,4,5,6,7,9,10,11,12,17,18,19,20,21,22,23,24,27,28]

                # size=192*192*64
                counts = metrics[3]
                counts = counts[choose]

                for j, count in enumerate(counts):

                    if j == 0:
                        # original background is for whole mask...
                        count = np.sum(counts[1:])
                        count = size - count

                    weights[j][i] = 1.0 / (count / size)


            means.append(np.mean(mean_))
            print(mean_)
            stds.append(np.mean(std_))
            print(std_)
            # should be an array of len == n_classes + 1
            # inludes background...
            weights_.append(list(np.mean(weights, axis=1)))
            print(weights_)

        # can save this separately and use to iterate through
        # check if split folds correctly...
        assert len(train_folds) == len(valid_folds)

        # we will use these values when validating also
        self.means, self.stds, self.weights = means, stds, weights_
        self.train, self.valid = train_folds, valid_folds

    def split(self):
        # five fold cross validation
        # if self.mode == 'full':
        # if len(list(self.folders[i])) > 1:
        # then run the following code in a loop
        # then split by class to the same training, validation and test folders_
        folders = GetSplits.patients(self)
        self.kf = KFold(n_splits=self.nfolds)
        if self.test_split < 1:
            if self.mode == "csv_full":
                training_folders = []
                test_folders = []
                # test_index = []
                for i, fold_list in enumerate(folders):
                    index = np.int(len(fold_list) * self.test_split)
                    test_folders += fold_list[index:]
                    # gives the class label and # of patients that belong to this class
                    # so when you parse originally DON't SHUFFLE...
                    # test_index += [self.indexes[i], len(fold_list)]
                    training_folders += fold_list[:index]
            else:
                # index of split (70/20/10)
                index = np.int(len(folders) * self.test_split)
                # hold some back for testing...
                test_folders = folders[index:]
                warnings.warn(
                    "No test lables for classification prediction. If labels exist set mode to csv_full."
                )
                # separate into training set
                training_folders = folders[:index]
        else:
            test_folders = []
            warnings.warn("No test folders for prediction. Use independent test set.")
            training_folders = folders

        self.training_folders = training_folders
        self.test = test_folders

        if self.tofilter == True:
            print("HI")
            self.export_metrics_new()
        else:
            self.export_metrics_old()

    def __len__(self):

        return len(self.train)

    @staticmethod
    def load_obj(name):
        with open(name, "rb") as f:
            return pickle.load(f)

class PatientData:
    def __init__(
        self,
        file_name,
        folds,
        nfold=0,
        home_path="/home/gpudual/SegmentHN/archive/data",
        transform=True,
        mode="default",
        labels_exist=False,
        dir_path='/home/gpudual',
    ):
        """
        # RERUN DATABUNCH ON NEW IMAGE IDS #
        (THIS WILL ENABLE (MORE EFFECTIVE) PARSING...)
        param: folds     : Taking a fold split from GetSplits to split Dataframe of patient slices from dictionary
                           of every possible avaliable slice for training created in process.databunch.py
        param: nfold     : Uses user defined cross validation fold (args.fold input) to split data.
        param: home_path : Ideal path to where process.databunch.pkl file is saved.
        param: transform : This transforms the dataframe (inverts rows >> columns)
        param: mode      : Default: Expects len(folds) > 1 ; single expects one fold, ie len(folds) == 1
        """
        # '/Users/josephmarsilla/github/SegmentHN/outputs'
        self.path = home_path
        self.name = file_name
        self.folds = folds
        self.nfold = nfold
        self.transform = transform
        self.mode = mode
        self.labels_exist = labels_exist
        self.dir_path = dir_path

    def load(self):
        # To-do
        # re-configure this from load directly from file name...
        # can also over ride with a specific home path to the file...
        path = f"{self.dir_path}/SegmentHN/archive/data"
        path = self.path if os.path.isdir(self.path) else path
        self.file_path = str(path + "/" + self.name + ".pkl")
        if not os.path.isfile(self.file_path):
            warnings.warn(f"{self.file_path} directory does not exist!!")

        with open(self.file_path, "rb") as f:
            return pickle.load(f)

    def df(self):
        test_data = pd.DataFrame.from_dict(self.load())
        test_data = test_data.T if self.transform else test_data

        return test_data

    def dataset(self):
        data = self.df()
        if self.mode == "single":
            # There is only one fold given in list
            warnings.warn("Using single fold.")
            data_new = data.loc[data[2].isin(self.folds)].reset_index()
        else:
            # There is more than one fold given in folds
            sect = self.folds[self.nfold]
            sect = np.array(sect) if self.labels_exist is True else sect
            # if instance = false if labels exist...
            if isinstance(sect, list) is False:
                # checks if fold is an array...
                new_fold = list(sect[:, 0])
                self.indexes = list(sect[:, 1])
                # these are the labels of the fold...
                data_new = data.loc[data[2].isin(new_fold)].reset_index()
                # use new_fold as a sorter...
            else:
                new_fold = sect
                self.indexes = range(len(new_fold))
                # not class labels but defines original order of output fold...
                warnings.warn("No labels.")

            data_new = data.loc[data[2].isin(new_fold)].reset_index()
            # Create the dictionary that defines the order for sorting
            sorterIndex = dict(zip(new_fold, self.indexes))
            # rank the dataframe with class values...
            data_new[4] = data_new[2].map(sorterIndex)

        return data_new

    def __len__(self):
        # return amount of training slices used...
        return len(self.dataset())

class LoadPatientVolumes(Dataset):
    def __init__(
        self,
        df,
        root,
        window=22,
        resample=None,
        transform=None,
        transform2=None,
        mode="train",
        to_filter=False,
        spacing=False,
        volume_type="targets",
        oar_version=None,
        external=False,
        img_path="/home/gpudual/bhklab/private/jmarsill/img",
        mask_path="/home/gpudual/bhklab/private/jmarsill/masks",
    ):
        """
        This is the class that our Dataloader object
        will take to create batches for training.
        param: df         : This is the sliced dataframe output of PatientData
        param: window     : This is the margin to crop the slice from a center
                            reference slice # in the z plane.
        param: to_augment : If to_augment == True ; augment data
        param: resample   : Resamples the df (int) - recommended to use when
                            to_augment set True.
        param: norm       : For Normalizing Wrongly Normalized Radcure Images
                            If Images properly normalized set to False
        """

        # self.data = pd.concat([df] * resample) if resample else df
        self.window = window
        self.transform  = transform
        self.transform2 = transform2
        self.resample = resample
        self.mode = mode
        # original state of image/mask
        self.mask = None
        self.img = None
        self.pat_idx = 0
        self.count = 0
        self.tofilter = to_filter
        self.spacing = spacing
        self.get_com = False
        self.volume_type = volume_type
        self.oar_version = oar_version
        # self.img_path = img_path
        # self.mask_path = mask_path
        self.external = external
        self.data = df
        self.root = root+"wolnet-sample/sample/"

        # sliding window, will be used to create slice(s) +/- center slice

    def __len__(self):
        return len(self.data)

    # def check(self, img, mask):
    #     # check x/y
    #     assert img[0].shape == (self.size, self.size)
    #     assert mask[0].shape == (self.size, self.size)
    #     # assert mask.max() > 0  # == self.args.n_classes # should be n_classes

    def load_data(self, idx):

        """
        This is called to load the imaged used in slicing.
        Will only load a new image iff the paths of the image behind
        a new one don't match!

        Set's self variable that will be called to slice the image/mask.
                # loading in mask path (without shuffling it)
                # want a full patient to be represented in a batch
                # this wil greatly reduce loading time...
        """

        # load dataframe
        self.patient = self.data[idx]
        self.img_path = f"{self.root}{self.patient}/CT_IMAGE.nrrd"
        self.mask_path = f"{self.root}{self.patient}/structures/Mandible_Bone.nrrd"
        self.load_from_sitk() 
        self.com = None

    def resample_sitk(self, image, mode="linear", new_spacing=None, filter=False ): # new_spacing=np.array((1.0, 1.0, 2.0)) , filter=True
        if new_spacing is not None: # originally taken from https://github.com/SimpleITK/SimpleITK/issues/561
            resample = sitk.ResampleImageFilter()
            if mode == "linear":
                resample.SetInterpolator = sitk.sitkLinear  # use linear to resample image
            else: # use sitkNearestNeighbor interpolation # best for masks
                resample.SetInterpolator = sitk.sitkNearestNeighbor
            orig_size = np.array(image.GetSize(), dtype=np.int)
            orig_spacing = np.array(image.GetSpacing())
            resample.SetOutputDirection(image.GetDirection())
            resample.SetOutputOrigin(image.GetOrigin())
            resample.SetOutputPixelType(image.GetPixelIDValue())
            new_spacing = new_spacing
            resample.SetOutputSpacing(new_spacing)
            new_size = orig_size * (orig_spacing / new_spacing)
            new_size = np.ceil(new_size).astype(np.int)
            new_size = [int(s) for s in new_size]
            resample.SetSize(new_size)
            if filter is True: # fights artifacts produced by analaising # only do this when resampling image (not mask...)
                img = resample.Execute(sitk.SmoothingRecursiveGaussian(image, 2.0))
            else:
                img = resample.Execute(image)
        else: # do nothing to the image...
            img = image
        return img

    def load_from_sitk(self):

        # if self.external is False: # OAR1204 OAR0720
        #     img_path=f'/cluster/projects/radiomics/Temp/joe/RADCURE_Joe/img/{self.patient}.nrrd'
        #     mask_path=f'/cluster/projects/radiomics/Temp/joe/RADCURE_Joe/masks/{self.patient}.nrrd'
        # else:
        img_path = self.img_path if self.img_path else None
        mask_path = self.mask_path if self.mask_path else None

        try:
            assert os.path.isfile(mask_path)
            assert os.path.isfile(img_path)
        except Exception:
            pass

        self.img = nrrd.read(img_path)
        self.img = self.img[0] #.transpose(2,0,1)
        self.mask= nrrd.read(mask_path)
        self.mask= self.mask[0] #.transpose(2,0,1)
        warnings.warn('Using nrrd instead of sitk.')

        self.to_mask = None
        assert self.mask is not None
        assert self.img is not None
        return
    
    def __getitem__(self, idx):

        self.load_data(idx)
        if self.transform is not None:
            if self.mask.max() > 0:
                self.img, self.mask = self.transform(self.img.copy(), self.mask.copy())
                if self.transform2 is not None:
                    img2, _ = self.transform2(self.img.copy(), self.mask.copy())
                    # only for dataset23
                    # img2 = img
            else:
                # only load if mask is zero from start...
                warnings.warn(f'Check {self.patient}...')
                self.z,self.x,self.y = self.img.shape
                img = self.img[:,self.x//2-192//2:self.x//2+192//2,self.y//2-192//2:self.y//2+192//2]
                mask = self.mask[:,self.x//2-192//2:self.x//2+192//2,self.y//2-192//2:self.y//2+192//2]
        if self.volume_type == "targets":
           # GTV is captured
           # final thing that's done, only use GTV...
           self.mask[self.mask>1] = 0
           try:
               assert self.mask.max() == 1
           except Exception:
               warnings.warn('Loading a Non-zero mask (Cropped out GTV)...')
        else:
            try:
                assert self.mask.max() == 19
            except Exception:
                warnings.warn('Loading a mask that has cropped out least one OAR...')
                print(self.mask.max())
                assert self.mask.max() > 0
                # self.check(img, mask)
        # because they are volumes should be the same shape
        # self.check(img, mask)
        img = torch.from_numpy(self.img).type(torch.FloatTensor)
        mask = torch.from_numpy(self.mask).type(torch.LongTensor)
        if self.transform2 is not None:
            img2 = torch.from_numpy(img2).type(torch.FloatTensor)
            return img2, img, mask
        else:
            return img, mask

class LoadPatientSlices(Dataset):
    def __init__(
        self, df, window, root, transform=None, resample=None, mode="train", factor=512
    ):
        """
        This is the class that our Dataloader object
        will take to create batches for training.
        param: df         : This is the sliced dataframe output of PatientData
        param: window     : This is the margin to crop the slice from a center
                            reference slice # in the z plane.
        param: to_augment : If to_augment == True ; augment data
        param: resample   : Resamples the df (int) - recommended to use when
                            to_augment set True.
        param: norm       : For Normalizing Wrongly Normalized Radcure Images
                            If Images properly normalized set to False
        """

        self.data = pd.concat([df] * resample) if resample else df
        self.window = window
        self.transform = transform
        self.resample = resample
        self.factor = factor
        # original state of image/mask
        self.mask = None
        self.img = None
        self.pat_idx = 0
        self.count = 0
        # sliding window, will be used to create slice(s) +/- center slice

    def __len__(self):
        return len(self.data)

    def check(self, img, mask):
        # check x/y
        assert img[0].shape == (self.factor // 2, self.factor // 2)
        assert mask.shape == (self.factor // 2, self.factor // 2)
        assert img.shape[0] == 11
        assert mask.max() > 0
        # == self.args.n_classes # should be n_classes

    def load_data(self, idx):

        """
        This is called to load the imaged used in slicing.
        Will only load a new image iff the paths of the image behind
        a new one don't match!

        Set's self variable that will be called to slice the image/mask.
                # loading in mask path (without shuffling it)
                # want a full patient to be represented in a batch
                # this wil greatly reduce loading time...
        """

        # load dataframe
        df = self.data
        # load mask
        mask_path_back = df.iloc[idx - 1][0] if idx > 0 else None  # None
        mask_path = str(df.iloc[idx][0])
        self.center = df.iloc[idx][1]
        self.mask_path = mask_path
        assert os.path.isfile(mask_path)
        # change mask path to image path.
        # load image
        old_patient_folder = str(df.iloc[idx - 1][2]) if idx > 0 else None
        patient_folder = str(df.iloc[idx][2])
        self.patient = patient_folder
        # mask_lbl = 'masks' if self.dataset_name == 'RADCURE' else 'mask'
        # img_path = str(mask_path).replace(mask_lbl, 'img')
        img_path = (f"/cluster/projects/radiomics/Temp/RADCURE-npy/img/{patient_folder}_img.npy")
        self.img_path = img_path
        assert os.path.isfile(img_path)

        # load patient folder information
        if mask_path_back == mask_path and self.mask is not None:
            # check that there is an image
            self.mask = self.mask
            self.img = self.img
            # check that image is also non_zero...
            assert self.img is not None

        else:
            # laod image & mask
            self.img = np.load(img_path)
            self.mask = np.load(str(mask_path))
            # sanity check image...
            assert self.mask is not None
            assert self.img is not None
            assert self.mask.max() > 0
            # == self.args.n_classes

        # check that mask and image dimensions are the same...
        # self.reset_masking()
        assert self.mask.shape == self.img.shape

    def __getitem__(self, idx):

        """
        Returns 3D arrays:
        1. Sliced Image (Z-window:z+window+1, H, W)
        2. Sliced Mask  (C, H, W) # definied by number of classes

        When used in a pytorch Dataloader the batches of outputs/targets are 4D tensors.
            #  Normalize for wrongly normalized Radcure images...
            #  Originally normalized in HU range -1000 to 400
            #  New Range -100 to 400 (.64 to 1.) # actually this will work perfectly
            #  New Range from -100 to 200 (.64 to .86) (not good for segmentation)
            #  For structSeg images, normalized already -100 to 400 therefore for -100 to 200 [0.,.6]
        """

        self.load_data(idx)
        # gets the center slice
        center_idx = np.int(self.center)
        # Define cropping dimentions
        window = self.window
        bottom = center_idx - window
        top = center_idx + window + 1

        # if center_idx > z size of the image. reload everything...
        if center_idx >= self.mask.shape[0] - 7:
            # reload image & mask
            self.img = np.load(str(self.img_path))
            self.mask = np.load(str(self.mask_path))

        # crop image & its mask...
        img_ = self.img[bottom:top, :, :].copy()
        mask_ = self.mask[center_idx, :, :].copy()
        # image shapes
        mask_shape = mask_.shape
        imshape = img_.shape

        if img_.shape[0] != 11 or mask_.max() == 0:
            # import image & mask again...
            self.img = np.load(str(self.img_path))
            self.mask = np.load(str(self.mask_path))
            # self.reset_masking()
            img_ = self.img[bottom:top, :, :].copy()
            mask_ = self.mask[center_idx, :, :].copy()
            assert img_.shape[0] == 11
            assert mask_.max() > 0

        assert img_[0].shape == (512, 512)
        assert mask_.shape == (512, 512)
        assert mask_.max() > 0

        if self.transform is not None:
            # can add normalization & cropping to this
            # make sure last & second last transformation
            img_, mask_ = self.transform(img_, mask_)
            # check img/mask
            self.check(img_, mask_)

        img_ = torch.from_numpy(img_).type(torch.FloatTensor)
        mask_ = torch.from_numpy(mask_).type(torch.LongTensor)

        return img_, mask_

        # Note: output image tensor should be 4D output mask tensor should be 4D if single class, 5D for multi class...


class TestLoader(Dataset):

    """
    Created on Wed May  9 17:01:04 2018
    @author: joseph
    Taken from pyGempick.mod
    Link: https://github.com/jmarsil/pygempick/blob/master/pygempick/modeling.py

    """

    def __init__(self, sample, dim=512 // 2, volume=1, to_augment=False):
        self.sample = (
            sample  # imput this as range/arange of number of test images there are...
        )
        self.dim = dim
        self.volume = volume
        self.to_augment = to_augment

    def stack(self, image):
        # ADD NOISE TO MASKED REGION..
        dim = self.dim
        image[image == 0] = 0.3
        # a = np.random.poisson(.6, (512, 512))
        # image *= a/np.linalg.norm(a)
        image *= np.random.normal(0.67, 0.1, (dim, dim))
        sigmas = np.arange(0.8, 0.9, 0.1)
        # sigma = random.choice(sigmas)
        # image = gaussian_filter(image, sigma=sigma)
        image[image > 1] = 1
        # if self.volume == 1:
        return image

    def make_img(self):
        dim = self.dim
        image = np.zeros((dim, dim), np.float32)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def __getitem__(self, idx):
        item = self.sample[idx]
        dim = self.dim
        image = TestLoader.make_img(self)
        ##bernouli trial to draw either circle or elippse...
        flip = np.random.rand()
        radrange = np.arange(50 // 2, 150 // 2, 1)
        ##picks a random particle radius between 4 and 8 pixels
        axis = random.choice(radrange)
        # N = width * height / 4
        ##chooses a random center position for the circle
        w = np.int(np.random.uniform(150 // 2, dim - 150 // 2))
        h = np.int(np.random.uniform(150 // 2, dim - 150 // 2))

        if flip < 0.5:
            # draw a circle
            cv2.circle(image, (h, w), np.int(axis), (255, 255, 255), -1)
        else:
            # draw an elippse...
            cv2.ellipse(
                image,
                (h, w),
                (int(axis) * 2, int(axis)),
                0,
                0,
                360,
                (255, 255, 255),
                -1,
            )

        # CONVERT BACK TO GRAY VALUES
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[image == 255] = 1

        # EXPAND DIMS IF 2D MASK
        mask = image.copy()
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, 0)  # expand dims on image plane...
        # stack & noisify image...
        image = TestLoader.stack(self, image)
        # normalize image...
        # check if 3D or 2D volume...
        if len(image.shape) == 2:
            # normalize image...
            image = np.expand_dims(image, 0)

        # Augment image...
        #         if self.to_augment:
        #             image, mask = utils.augment(image, mask)

        assert image.shape == mask.shape

        # assert len(image.shape) == 3
        # assert len(mask.shape) == 3
        # check the mask, for every mask value in targets that is 0
        # simply don't count that value in the loss calculation...
        # This can allow us to use patient information without penalizing the model
        # if it produces a wrong or inconsistent class...

        return (
            torch.from_numpy(image).type(torch.FloatTensor),
            torch.from_numpy(mask).type(torch.FloatTensor),
        )

    def __len__(self):
        return len(self.sample)


###############################
############## SCRATCH ########
###############################

        
        ##################
        # in old load_data
        # df = self.data
        # load mask
        # 54 is the right window width
        # [path, Z_COM, patient] >> anything else doesn't matter
        # from com save Z slice
        # if self.external is True:
        #     path = '/cluster/projects/radiomics/Temp/michal/keynote/data/'
        #     img_path = path  + df.iloc[idx][0]
        #     mask_path = path  + df.iloc[idx][1]
        #     self.patient = f'Keynote {mask_path}'
        #     self.tumour_type = None
        #     self.counts = None # for now
        # else:
        # comment this back out after done messing...
        # uncomment for original...
        # if self.volume_type=='targets':
        #     df = self.data
        #     self.patient = str(df.iloc[idx][0])
        # else:
        #     if self.external is False:
        #         # mask_path = str(df.iloc[idx][0])
        #         if self.mode == 'test':
        #             self.patient = str(df.iloc[idx][0])
        #         else:
        #             self.patient = str(df.iloc[idx][2])
        #         # self.counts = np.array(df.iloc[idx][3])
        #         self.version = 1
        #     else:
        #         self.mask_path = 
        #         # structseg...
        #         # self.mask_path = df.iloc[0][0]
        #         # self.img_path = df.iloc[0][0].replace('masks', 'imgs')
        #         # # only needed for testing otherwise not needed...
        #         # # comment out otherwise...
        #         # self.version = df.iloc[idx]['version']
        #         # PDDCA...should work for ALL external datsets...
        #         self.mask_path = df.iloc[idx][1]
        #         self.img_path = df.iloc[idx][0]
        #         # .replace('masks', 'imgs')
        # self.load_from_sitk()
        # # 1x1x1
        # if self.get_com:
        #     self.com = df.iloc[idx][1]
        # else:
        #     self.com = None
        ######################
        # in old load_from_sitk()
        # try:
        #     try:
        #         mask = sitk.ReadImage(mask_path)
        #     except Exception:
        #         warnings.warn(f"Couldn't load mask at {mask_path}")
        #         mask = sitk.ReadImage(img_path)
        #     img = sitk.ReadImage(img_path)
        #     # if self.external is True:
        #     #     img = self.resample_sitk(img, new_spacing=np.array((1.0, 1.0, 2.0)))
        #     #     mask = self.resample_sitk(mask, mode="nearest", new_spacing=np.array((1.0, 1.0, 2.0)))
        #     self.img = sitk.GetArrayFromImage(img)
        #     self.mask = sitk.GetArrayFromImage(mask)

        # except Exception as e:
        # print(e)
        #########################

    
        # if self.external is False:

        #     if self.volume_type == "targets":
        #         # only have GTV, updated for MERK study
        #         warnings.warn('Using GTV contour only...')
        #         self.mask[self.mask>1] = 0
        #         assert self.mask.max() > 0
        #         assert self.mask is not None
        #         assert self.img is not None

        #     else:

        #         if self.oar_version == 16:
        #             # ~1220 patients for this cohort
        #             choose = np.array( [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22] )

        #         elif self.oar_version == 18:
        #             # 647 patients in this cohort
        #             choose = np.array( [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22] )

        #         elif self.oar_version == 19:
        #             # with the 07 arrays
        #             choose = np.array([0,3,4,5,6,7,9,10,11,12,17,18,19,20,21,22,23,24,27,28])

        #             # OAR07
        #             # rois = ["GTV","BRAIN","BSTEM","SPCOR","ESOPH","LARYNX","MAND",
        #             #     "POSTCRI","LPAR","RPAR","LACOU","RACOU","LLAC","RLAC","RRETRO",
        #             #     "LRETRO","RPLEX","LPLEX","LLENS","RLENS","LEYE","REYE","LOPTIC",
        #             #     "ROPTIC","LSMAN","RSMAN","CHIASM","LIPS","OCAV","IPCM","SPCM",
        #             #     "MPCM"]

        #             # self.mask[self.mask == 1] = 0
        #             # self.mask[self.mask == 2] = 0
        #             # self.mask[self.mask == 8] = 0
        #             # self.mask[self.mask == 13] = 0
        #             # self.mask[self.mask == 14] = 0
        #             # self.mask[self.mask == 15] = 0
        #             # self.mask[self.mask == 16] = 0
        #             # self.mask[self.mask == 25] = 0
        #             # self.mask[self.mask == 26] = 0
        #             # self.mask[self.mask > 28] = 0
        #             # self.mask[self.mask > 26] -= 2
        #             # self.mask[self.mask > 16] -= 4
        #             # self.mask[self.mask > 8] -= 1
        #             # self.mask[self.mask > 2] -= 2

        #             # OAR08
        #             # rois = ["GTV","LCTV", "RCTV", "BRAIN","BSTEM","SPCOR","ESOPH","LARYNX","MAND",
        #             #     "POSTCRI","LPAR","RPAR","LACOU","RACOU","LLAC","RLAC","RRETRO",
        #             #     "LRETRO","RPLEX","LPLEX","LLENS","RLENS","LEYE","REYE","LOPTIC",
        #             #     "ROPTIC","LSMAN","RSMAN","CHIASM","LIPS","OCAV","IPCM","SPCM",
        #             #     "MPCM"]

        #             self.mask[self.mask == 1] = 0
        #             self.mask[self.mask == 2] = 0
        #             self.mask[self.mask == 3] = 0
        #             self.mask[self.mask == 4] = 0
        #             self.mask[self.mask == 10] = 0
        #             self.mask[self.mask == 15] = 0
        #             self.mask[self.mask == 16] = 0
        #             self.mask[self.mask == 17] = 0
        #             self.mask[self.mask == 18] = 0
        #             self.mask[self.mask == 27] = 0
        #             self.mask[self.mask == 28] = 0
        #             self.mask[self.mask > 30] = 0
        #             self.mask[self.mask > 28] -= 2
        #             self.mask[self.mask > 18] -= 4
        #             self.mask[self.mask > 10] -= 1
        #             self.mask[self.mask > 4] -= 4

        # else:
        #     # pass
        #     # # only for dataset23
        #     # self.img = self.img.transpose(2,1,0)
        #     # self.mask = self.img
        #     pass

# if self.to_mask is not None:
# to_mask = torch.tensor(self.to_mask).type(torch.FloatTensor)
# return img, mask, to_mask
# else:

# modified for OAR segmentation...
# class LoadPatientVolumes(Dataset):
#     def __init__(
#         self,
#         df,
#         root,
#         factor=512,
#         window=22,
#         augment_data=None,
#         resample=False,
#         transform=None,
#         mode="train",
#         to_filter=False,
#         spacing=False,
#         volume_type="targets",
#         oar_version=None,
#         img_path="/home/gpudual/bhklab/private/jmarsill/img",
#         mask_path="/home/gpudual/bhklab/private/jmarsill/masks",
#     ):
#         """
#         This is the class that our Dataloader object
#         will take to create batches for training.
#         param: df         : This is the sliced dataframe output of PatientData
#         param: window     : This is the margin to crop the slice from a center
#                             reference slice # in the z plane.
#         param: to_augment : If to_augment == True ; augment data
#         param: resample   : Resamples the df (int) - recommended to use when
#                             to_augment set True.
#         param: norm       : For Normalizing Wrongly Normalized Radcure Images
#                             If Images properly normalized set to False
#         """
#
#         self.data = pd.concat([df] * augment_data) if augment_data  else df
#         self.resample=resample
#         self.window = window
#         self.transform = transform
#         self.resample = resample
#         self.mode = mode
#         # original state of image/mask
#         self.mask = None
#         self.img = None
#         self.pat_idx = 0
#         self.count = 0
#         self.size = factor // 2
#         self.tofilter = to_filter
#         self.spacing = spacing
#         self.get_com = False
#         self.volume_type = volume_type
#         self.oar_version = oar_version
#         self.img_path = img_path
#         self.mask_path = mask_path
#
#         # sliding window, will be used to create slice(s) +/- center slice
#
#     def __len__(self):
#         return len(self.data)
#
#     def check(self, img, mask):
#         # check x/y
#         assert img[0].shape == (self.size, self.size)
#         assert mask[0].shape == (self.size, self.size)
#         # assert mask.max() > 0  # == self.args.n_classes # should be n_classes
#
#     def load_data(self, idx):
#
#         """
#         This is called to load the imaged used in slicing.
#         Will only load a new image iff the paths of the image behind
#         a new one don't match!
#
#         Set's self variable that will be called to slice the image/mask.
#                 # loading in mask path (without shuffling it)
#                 # want a full patient to be represented in a batch
#                 # this wil greatly reduce loading time...
#         """
#
#         # load dataframe
#         df = self.data
#         self.mask_path = str(df.iloc[idx][0])
#         self.img_path = self.mask_path.replace('OARS050120/masks','DDNN/img')
#         self.patient = str(df.iloc[idx][2])
#
#         try:
#             self.counts = np.array(df.iloc[idx][3])
#         except Exception:
#             self.counts = None
#
#         if self.tofilter == True:
#             # using pre-calculated COM...
#             self.load_from_sitk()
#             # 1x1x1
#             if self.get_com:
#                 self.com = df.iloc[idx][1]
#             else:
#                 self.com = None
#
#         else:
#             self.center = df.iloc[idx][1]
#             assert os.path.isfile(self.mask_path)
#             self.img_path = f"/cluster/projects/radiomics/Temp/RADCURE-npy/img/{self.patient}_img.npy"
#             assert os.path.isfile(self.img_path)
#             # load patient folder information
#             # laod image & mask
#             self.img = np.load(self.img_path)
#             self.mask = np.load(str(self.mask_path))
#
#             # sanity check image...
#             assert self.mask is not None
#             assert self.img is not None
#             assert self.mask.max() > 0
#
#     def get_param(self):
#
#         if self.com is None:
#             # using full mask, can use one common OAR...
#             msk = self.mask.copy()
#             assert msk.max() > 0
#
#             if self.oar_version == 'targets':
#                 # want to only crop around gtv...
#                 msk[msk>1] = 0
#                 # weight GTV class more...
#
#             self.com = np.round(measure.center_of_mass(msk))
#             msk = None
#
#         self.center = self.com[0]
#
#         if self.mode == "train":
#             # shifts the z plane crop...
#             if self.spacing == "3mm":
#                 #  # 1mm, 1mm, 1mm
#                 # vnet used 20 as range...
#                 a = np.arange(-10, 11, 1.0)  # 3mm, 1mm, 1mm
#             else:
#                 a = np.arange(-30, 31, 1.0)
#
#             b = np.random.choice(a)
#             self.center += b
#
#         shape = self.mask.shape
#         end = shape[0] - self.window
#
#         if self.center < self.window - 1:
#             self.center = self.window + self.window // 2
#
#         elif self.center > end:
#             self.center = self.center - self.window // 2
#
#         bottom = self.center - self.window  # //2 # input 54//2
#         top = self.center + self.window  # //2+1 # input 54 //2
#
#         # can't crop wrong...
#         # crop array to make everything a bit more manageable...
#         try:
#             assert bottom > 0
#         except Exception:
#             warnings.warn('Cropping starting from z==1.')
#             bottom = 1
#             top = 1 + self.window*2
#
#         try:
#             assert top < shape[0]
#         except Exception:
#             warnings.warn(f'Cropping ending at {shape[0]}.')
#             bottom = shape[0] - 1 - self.window*2
#             top = shape[0] - 1
#
#         msk = self.mask[np.int(bottom) : np.int(top)]
#
#         try:
#             assert msk.shape[0] == self.window*2
#             self.mask = msk
#             msk = None
#
#         except Exception:
#             warnings.warn(f'Check patient mask: {self.patient}, shape: {self.mask.shape}, Now using center crop...')
#             # print(f'Check patient mask: {self.patient}, shape: {self.mask.shape}, Now using center crop...')
#             bottom = self.mask.shape[0]//2 - self.window
#             top = self.mask.shape[0]//2 + self.window
#             self.mask = self.mask[np.int(bottom) : np.int(top)]
#
#         assert self.mask.max() > 0
#         self.img = self.img[np.int(bottom) : np.int(top)]
#         # save images at original resolution...
#
#     def resample_sitk(self, image, mode="linear", new_spacing=None, filter=False ): # new_spacing=np.array((1.0, 1.0, 3.0)) , filter=True
#
#         if new_spacing is not None:
#             # originally taken from https://github.com/SimpleITK/SimpleITK/issues/561
#             resample = sitk.ResampleImageFilter()
#             if mode == "linear":
#                 resample.SetInterpolator = sitk.sitkLinear  # use linear to resample image
#             else:
#                 # use sitkNearestNeighbor interpolation
#                 # best for masks
#                 resample.SetInterpolator = sitk.sitkNearestNeighbor
#
#             orig_size = np.array(image, dtype=np.int)
#             orig_spacing = np.array(image.GetSpacing())
#             resample.SetOutputDirection(image.GetDirection())
#             resample.SetOutputOrigin(image.GetOrigin())
#             resample.SetOutputPixelType(image.GetPixelIDValue())
#
#             new_spacing = new_spacing
#             resample.SetOutputSpacing(new_spacing)
#             new_size = orig_size * (orig_spacing / new_spacing)
#             new_size = np.ceil(new_size).astype(np.int)  #  Image dimensions are in integers
#             new_size = [int(s) for s in new_size]
#             resample.SetSize(new_size)
#
#             if filter is True:
#                 # fights artifacts produced by analaising
#                 img = resample.Execute(sitk.SmoothingRecursiveGaussian(image, 2.0))
#             else:
#                 img = resample.Execute(image)
#         else:
#             # do nothing to the image...
#             img = image
#
#         return img
#
#     def load_from_sitk(self, mask_path=None, img_path=None):
#         # load sitk image & mask...
#         # can make these paths better...
#         warnings.warn(f'Loading mask from {self.mask_path}')
#         assert os.path.isfile(str(self.mask_path))
#         assert os.path.isfile(str(self.img_path))
#
#         # load sitk images
#         mask = sitk.ReadImage(self.mask_path)
#         img = sitk.ReadImage(self.img_path)
#
#         # export image arrays
#         self.img = sitk.GetArrayFromImage(img)
#         self.mask = sitk.GetArrayFromImage(mask)
#
#         if self.resample is False:
#
#             if self.volume_type == "targets":
#                 assert self.mask.max() > 0
#
#             else:
#
#                 self.mask[self.mask == 1] = 0
#                 self.mask[self.mask == 2] = 0
#                 self.mask[self.mask == 8] = 0
#                 self.mask[self.mask == 13] = 0
#                 self.mask[self.mask == 14] = 0
#                 self.mask[self.mask == 15] = 0
#                 self.mask[self.mask == 16] = 0
#                 self.mask[self.mask == 25] = 0
#                 self.mask[self.mask == 26] = 0
#                 self.mask[self.mask > 28] = 0
#                 self.mask[self.mask > 26] -= 2
#                 self.mask[self.mask > 16] -= 4
#                 self.mask[self.mask > 8] -= 1
#                 self.mask[self.mask > 2] -= 2
#
#         assert self.mask is not None
#         assert self.img is not None
#
#         return
#
#     def __getitem__(self, idx):
#
#         self.load_data(idx)
#         self.get_param()
#
#         if self.transform is not None:
#
#             img, mask = self.transform(self.img.copy(), self.mask.copy())
#
#         if self.volume_type == "targets":
#            # GTV is captured
#            # final thing that's done, only use GTV...
#            mask[mask>1] = 0
#
#            try:
#                assert mask.max() == 1
#
#            except Exception:
#                warnings.warn('Loading a Non-zero mask (Cropped out GTV)...')
#
#         else:
#
#             try:
#                 assert self.mask.max() == 19
#
#             except Exception:
#                 warnings.warn('Loading a mask that has cropped out least one OAR...')
#                 assert mask.max() > 0
#                 # self.check(img, mask)
#
#
#         # because they are volumes should be the same shape
#         assert self.mask.shape == self.img.shape
#         self.check(img, mask)
#         img = torch.from_numpy(img).type(torch.FloatTensor)
#         mask = torch.from_numpy(mask).type(torch.LongTensor)
#         return img, mask