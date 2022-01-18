import operator, os, warnings, time, re, yaml, datetime, gc, pickle, json, glob
import pandas as pd
import dicom_contour.contour as dcm
import pydicom as pdcm
import numpy as np
import SimpleITK as sitk
from matplotlib import *
from PIL import ImageDraw, Image
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
import scipy.misc as misc
import scipy.ndimage.interpolation as inter
import umap
import nibabel as nib
import warnings
from regex import *

def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_immediate_subdirectories(a_dir):
    """
    link: https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    :param a_dir:
    :return:
    """
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_files(a_dir):
    """
    link: https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    :param a_dir:
    :return:
    """
    return [ name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

def getConfig(wkdir, dataset):
    try:
        with open(f"{wkdir}config/processing-{dataset}.yml") as config_file:
            config = yaml.safe_load(config_file)

    except:
        config = {}

    try:
        with open(f"{wkdir}config/roi_exceptions_{dataset}.yaml") as exception_file:
            except_yaml = yaml.safe_load(exception_file)
    except:
        except_yaml = {}

    return config, except_yaml


class Dicom:
    def __init__(
        self,
        path,
        structurePath=None,
        dataset="opc1",
        exceptions={},
        rois=[],
        roi_only=False,
        resample=True,
        neck_levels=False,
        folder=None,
    ):
        """
        :params:
        path (str):         file input path including folder name (Ex: /cluster/projects/radiomics/RADCURE-images/1234567/)
        dataset (str):      defines dataset to navigate dataset-specifc folder structures
        exceptions (dict):  dictionary with patientID as key and dictionary of [roi_name]: desired_name
        rois (list):        list of rois you want to process
        roi_only (bool):    boolean to process only roi names from RTSTRUCT

        :returns <Dicom object>:
        """
        start = time.time()
        self.path = path
        self.structurePath = structurePath
        self.dataset = dataset
        self.neck_levels = neck_levels
        self.roi_only = roi_only
        self.ts = time.time()
        self.date = datetime.datetime.fromtimestamp(self.ts).strftime("%Y_%m_%d_%H%M%S")
        self.folder = folder
        # self.rois = rois
        self.rois=sccgen

        if not roi_only:
            # if resample==True, will resample spacing to 1x1x1 mm^3
            self.image = self.get_images(path)
            # for path in self.paths:
                # try:
            contours = self.get_contours(structurePath, exceptions, rois)
            self.masks = self.create_masks(contours)
                # break
                # except Exception as e:
                #     warnings.warn(f"{path} is bad, error is: {e}")

            self.orig_sitk_img = self.read_dicom_image()

        else:
            self.get_contours(path, exceptions, rois)

        print(f"Processing {path} took {round (time.time()-start, 2)} seconds")

    def get_images(self, path, resample=False):
        """
        :params:
        path (str):file input path including folder name (Ex: /cluster/projects/radiomics/RADCURE-images/1234567/)
        :returns:
        sorted_slices (ndarray): normalized numpy array of the image (Dimensions: (z, 512, 512))
        """
        self.patient = path.split("/")[-1]
        if len(self.patient) == 0:
            self.patient = path.split("/")[-2]
        if path[-1] != "/":
            path += "/"

        # modified radcure outputs...
        # if self.dataset == "radcure":
        #     a = os.listdir(path)
        #     for name in a:
        #         if "ImageSet" in name:
        #             path += name + "/"
        #             self.dicom_path = path
            # sorted(os.listdir(path)) [0]
            # path += "ImageSet_0.DICOM/"
        warnings.warn(f"Using {path} to load .DICOMs")
        self.dicom_path = path
        # This is what read_dicom_image is doing automatically
        # do this iteration to find right path...
        # fancy string manips
        # parent = path.split(self.patient)[0]
        # paths = glob.glob(parent + '*')
        # paths = [p for p in paths if path != p]
        # paths.sort()
        # paths = [path] + paths
        # self.paths = paths
        # warnings.warn(f'Iterating through {paths}.')
        # for path in paths:
        #     try:
        slices, self.spacing, self.origin, self.thickness, self.z_levels = ( {},0, 0, 0, [],)
        for s in os.listdir(path):
            if ("RTDOSE" not in s) and ("RTSTRUCT" not in s) and ("RTPLAN" not in s):
                f = pdcm.read_file(path + "/" + s)
                img = f.pixel_array
                # Remove -2000 noise values which lie outside of patient
                img[img == -2000] = 0
                # convert to Hounsfield Units (HU)
                intercept = f.RescaleIntercept
                slope = f.RescaleSlope
                if slope != 1:
                    img = slope * img.astype(np.float64)
                    img = img.astype(np.int16)
                try:
                    img += np.int16(intercept)
                except:
                    img += np.int16(intercept).astype(img.dtype)
                slices[f.ImagePositionPatient[2]] = img
                if f.PixelSpacing != self.spacing:
                    if self.spacing != 0:
                        warnings.warn(
                            f"Pixel spacing between slices is inconsistent!! {f.PixelSpacing}"
                        )
                    self.spacing = f.PixelSpacing
                if f.ImagePositionPatient[:2] != self.origin:
                    if self.origin != 0:
                        warnings.warn(
                            f"New x-y plane origin!! {f.ImagePositionPatient[:2]}"
                        )
                    self.origin = f.ImagePositionPatient[:2]
                if f.SliceThickness != self.thickness:
                    if self.thickness != 0:
                        warnings.warn(
                            f"Pixel thickness between slices is inconsistent!! {f.SliceThickness}"
                        )
                    self.thickness = f.SliceThickness
                self.z_levels.append(round(float(f.ImagePositionPatient[2]), 2))

            # except Exception as e:
            #     warnings.warn(f'Path is bad because of {e}.')
        # self.paths.remove(path)
        # automatically resample
        sorted_slices = np.array([slices[s] for s in sorted(slices)])
        self.z_levels = sorted(self.z_levels)
        # return self.normalize(sorted_slices)
        # sorted_slices, self.new_spacing = self.resample(sorted_slices) if resample is True else sorted_slices
        sorted_slices = sorted_slices.astype(np.int16)

        return sorted_slices

    def resample(self, image, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing
        spacing = map(float, ([self.thickness] + self.spacing))
        spacing = np.array(list(spacing))
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = inter.zoom(image, real_resize_factor)
        return image, new_spacing

    def normalize(self, img):
        MIN_BOUND = -1000.0
        MAX_BOUND = 400
        image = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.0
        image[image < 0] = 0.0
        return image

    def get_contours(self, path, exceptions, select):
        """
        :params:
        path (str):         file input path including folder name (Ex: /cluster/projects/radiomics/RADCURE-images/1234567/)
        dataset (str):      defines dataset to navigate dataset-specifc folder structures
        exceptions (dict):  dictionary with patientID as key and dictionary of [roi_name]: desired_name
        select (list):      list of rois you want to process

        :returns:
        contours (list):    list of contour points per slice
        """
        if path[-1] != "/":
            path += "/"
        if not self.structurePath:
            rtstruct = self.get_contour_file(path)
        else:
            # if self.structurePath != "/":
            #     self.structurePath+="/"
            rtstruct = pdcm.read_file(self.structurePath) #+ "000000.dcm"
            # rtstruct = self.get_contour_file(self.structurePath) # if not provided, search path...
        self.roi_list = [x.upper() for x in dcm.get_roi_names(rtstruct)]
        # normal extraction...
        # self.roi_select = self.select_rois(exceptions, select)
        self.roi_select = self.roi_list
        self.rois = self.roi_list

        if self.roi_only:
            return

        contours = [[] for i in self.roi_select]
        print(contours)

        for n, roi in enumerate(rtstruct.ROIContourSequence):
            name = self.roi_list[n]
            if name in self.roi_select:
                # initialize contour points array with len (number of slices)
                temp_contour = [[] for i in range(self.image.shape[0])]

                for slice in roi.ContourSequence:
                    # slice z-axis level
                    z_level = slice.ContourData[2]
                    # save slice contour
                    try:
                        temp_contour[self.z_levels.index(round(z_level, 0))].append(
                            self.format_contours(slice.ContourData))
                    except:
                        try:
                            temp_contour[self.z_levels.index(round(z_level, 1))].append(
                                self.format_contours(slice.ContourData))
                        except:
                            temp_contour[self.z_levels.index(round(z_level, 2))].append(
                                self.format_contours(slice.ContourData))
                # save roi contour
                try:
                    # when we only take one ROI name
                    contours[self.roi_select.index(name)] = temp_contour
                except Exception:
                    # used for nodes...
                    warnings.warn(f"Using {name} for contour.")
                    contours[name] = temp_contour

        return contours

    def format_contours(self, points):
        formatted_points = self.format_raw_contours(points)
        new_points = []
        spacing = self.spacing
        origin = self.origin
        for point in formatted_points:
            new_points.append( (np.ceil((point[0] - origin[0]) / spacing[0]),
                                np.ceil((point[1] - origin[1]) / spacing[1]),))
        return new_points

    def format_raw_contours(self, points):
        formatted_points = []
        for n in range(0, len(points), 3):
            formatted_points.append((float(points[n]), float(points[n + 1])))
        return formatted_points

    def get_contour_file(self, path):
        fpaths = [path + f for f in os.listdir(path) if ".dcm" in f]
        n = 0
        contour_file = None
        print("RTSTRUCT files:", fpaths)

        for fpath in fpaths:
            fname = fpath.split("/")[-1].upper()
            if "RTSTRUCT" in fname:
                print("yes")
                f = pdcm.read_file(fpath)
                if "ROIContourSequence" in dir(f):
                    contour_file = f
                n += 1

        if n > 1:
            warnings.warn("There are multiple contour files, returning the last one!")
        if contour_file is None:
            print("No contour file found in directory")

        return contour_file

    def select_rois(self, exceptions, select):
        if not select:
            select = ["EXT", "GTV", "CTV", "LNECK", "RNECK", "NODES"]

        selected_rois = self.exceptions(exceptions, select)
        regex_match_rois = self.get_names(roi_order=sccgen, regex=cgenregex) #2 for nsclc tcia...
        self.rois = list(regex_match_rois.keys())
        # self.rois=[] # added after the fact...

        if select[0] != "NODES":

            for n, roi in enumerate(select):

                if roi not in selected_rois:  # if roi hasn't been found by
                    try:
                        selected_rois[n] = regex_match_rois[roi]
                        # added after the fact...
                        # self.rois.append(roi)
                    except:
                        pass

                elif roi == 'NODES':
                    # everything after class 30 we can export to nodes...
                    for j, val in regex_match_rois[roi]:
                        selected_rois[n+j] = val
        else:
            selected_rois = regex_match_rois["NODES"]

        print(selected_rois)
        return selected_rois

    def exceptions(self, exceptions, select):
        rois = ["" for i in range(len(select))]

        for roi in exceptions:
            rois[select.index(roi)] = exceptions[roi]
            # insert name exception into appropriate index
        return rois

    def get_names(self, roi_order=None , regex=None):

        if roi_order:
            assert regex  # list of regex strings to parse dicoms
            assert len(roi_order) == len(regex)  # has to be of equal length...
        else:
            roi_order = ROIS
            regex = REGEX

        dict_rois = {}

        for i, roi in enumerate(roi_order):
            try:
                if roi == "LNECK" or roi == "RNECK":
                    dict_rois[roi] = sorted( [n for j, n in enumerate(self.roi_list) if re.fullmatch(regex[i], n) ] )[0]

                elif roi == "NODES":
                    warnings.warn("Extracting all nodes baby.")
                    nodes = sorted([n for j, n in enumerate(self.roi_list) if re.fullmatch(regex[i], n)])
                    dict_rois[roi] = nodes
                    # print(nodes)
                    # for i, name in enumerate(nodes):
                    #     dict_rois[i] = name
                else:
                    dict_rois[roi] = sorted([n for j, n in enumerate(self.roi_list) if re.fullmatch(regex[i], n)])[-1]
            except:
                pass

        return dict_rois

    def create_masks(self, contours, mode="merge"):

        masks = []
        for roi in contours:
            mask = np.zeros_like(self.image)

            for j, slice in enumerate(roi):
                temp = np.zeros((512, 512))
                for contour in slice:
                    try:
                        img = Image.new("L", (512, 512), 0)
                        ImageDraw.Draw(img).polygon(contour, outline=1, fill=1)
                        temp_mask = np.array(img)
                        if mode == "merge":
                            temp += temp_mask
                            a[
                                a == 2
                            ] = 1  # overlap of mask --> bring ceiling back down to 1
                        elif mode == "largest":
                            a = np.sum(np.array(temp))
                            b = np.sum(temp_mask)
                            if a < b:
                                temp = temp_mask
                    except:
                        pass

                mask[j] = temp

            masks.append(mask)

        return np.array(masks, dtype="int8")

    # taken from Michal's script to extract image arrays from simpleITK
    # so we can resample directly OR save the un-resampled image & let
    # pyradiomics do it itself...ß
    def read_dicom_image(self):
        # initiate simple ITK image extractor
        reader = sitk.ImageSeriesReader()
        # path is the directory of the dicom folder
        dicom_names = reader.GetGDCMSeriesFileNames(self.dicom_path)
        reader.SetFileNames(dicom_names)
        warnings.warn("Loading DICOM successful.")
        return reader.Execute()

    # save images at original resolution...
    def resample_sitk(self, image, mode="linear", new_spacing=None, filter=False ):
    #  new_spacing=np.array((1.0, 1.0, 3.0)) , filter=True
        if new_spacing is not None:
            # originally taken from https://github.com/SimpleITK/SimpleITK/issues/561
            resample = sitk.ResampleImageFilter()
            if mode == "linear":
                resample.SetInterpolator = sitk.sitkLinear  # use linear to resample image
            elif mode =='bspline':
                resample.SetInterpolator = sitk.sitkBSpline
            else:
                resample.SetInterpolator = sitk.sitkNearestNeighbor

            orig_size = np.array(self.orig_sitk_img.GetSize(), dtype=np.int)
            orig_spacing = np.array(self.orig_sitk_img.GetSpacing())
            resample.SetOutputDirection(self.orig_sitk_img.GetDirection())
            resample.SetOutputOrigin(self.orig_sitk_img.GetOrigin())
            resample.SetOutputPixelType(self.orig_sitk_img.GetPixelIDValue())

            new_spacing = new_spacing
            resample.SetOutputSpacing(new_spacing)
            new_size = orig_size * (orig_spacing / new_spacing)
            new_size = np.ceil(new_size).astype(np.int)  #  Image dimensions are in integers
            new_size = [int(s) for s in new_size]
            resample.SetSize(new_size)

            if filter is True:
                img = resample.Execute(sitk.SmoothingRecursiveGaussian(image, 2.0))
            else:
                img = resample.Execute(image)
        else:
            # do nothing to the image...
            img = image

        return img

    def export(
        self,
        path,
        folder,
        exclude=["masks_save"],
        mode="numpy",
        resample=True,
        slices=False,
        spacing=np.array((1.0, 1.0, 3.0)),
    ):

        """
        Exports image and corresponding mask arrays into /HN-export/ folder.
        Exports masks pyplot figure into /HN-export/ folder.

        Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
            img (ndarray): 3D numpy array of image
            roi (ndarray): 4D numpy array of masks of regions of interests
            info (ndarray): 2D array of information about masks
        # TODO:
            early stop / start plotting on first non-empty
        """
        # img, roi, dm_roi = self.img, self.masks, self.dm_masks
        columns = 4
        rows = 4
        x = [0, 0]

        os.makedirs(path, exist_ok=True)
        if mode == "nifti":
            os.makedirs(f"{path}/nifti", exist_ok=True)

        if "img" not in exclude:
            if mode == "nifti":
                os.makedirs(f"{path}/nifti/img/", exist_ok=True)
                img = nib.Nifti1Image(self.image, np.eye(4))
                nib.save(img, f"{path}/nifti/img/{folder}.nii.gz")

            else:
                if mode == "numpy":
                    warnings.warn("Saving in numpy format...")
                    os.makedirs(f"{path}/img/", exist_ok=True)
                    np.save(f"{path}/img/{folder}", self.image)

                elif mode == "itk":
                    # added this to enable effective pyradiomic feature extraction
                    # try:
                    os.makedirs(f"{path}/{folder}/", exist_ok=True)
                    # should we need to export image we processed ourselves :D
                    sitk_img = sitk.GetImageFromArray(self.image)
                    sitk_img.CopyInformation(self.orig_sitk_img)
                    if resample is True:
                        # for 2D DDNN used np.array((1., 1., 3.))
                        sitk_img = self.resample_sitk(sitk_img, new_spacing=spacing)
                    sitk.WriteImage(sitk_img, f"{path}/{folder}/CT_IMAGE.nrrd")
                    # output means from clipped image...
                    os.makedirs(f"{path}/pkl/", exist_ok=True)
                    toclip = HistogramClipping()
                    img = toclip(self.image)
                    # create dictionary to record metrics...
                    dic = {}
                    dic['mean'] = np.mean(img)
                    dic['std'] = np.std(img)
                    img = 0.
                    sitk_img = 0.

        if "masks" not in exclude:
            if mode == "nifti":
                os.makedirs(f"{path}/nifti/masks/", exist_ok=True)
                for n, mask in enumerate(self.masks):
                    img = nib.Nifti1Image(mask, np.eye(4))
                    nib.save(img, f"{path}/nifti/masks/{folder}_{n}.nii.gz")
            else:
                print(f"Max for {folder} is {self.masks.max()}")

                if mode == "numpy":
                    warnings.warn("Saving mask in numpy format...")
                    # change this back to masks for general processing...
                    os.makedirs(f"{path}/masks/", exist_ok=True)
                    np.save(f"{path}/masks/{folder}", self.masks)

                elif mode == "itk":
                    os.makedirs(f"{path}/{folder}/structures/", exist_ok=True)
                    # should we need to export image we processed ourselves :
                    for i, mask_ in enumerate(self.masks):
                        # resample each mask individually...
                        if mask_.max()>0:
                            sitk_mask = sitk.GetImageFromArray(mask_)
                            sitk_mask.CopyInformation(self.orig_sitk_img)
                            # a = sitk.StatisticsImageFilter()
                            # a.Execute(sitk_mask)
                            if resample is True:
                                # resample to 1mm^3 spacing...
                                sitk_mask = self.resample_sitk(sitk_mask, mode="nearest", new_spacing=spacing)
                            sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
                            sitk.WriteImage(sitk_mask, f"{path}/{folder}/structures/{self.rois[i]}.nrrd") #, useCompression=True)
                            print(f"Extraction of {folder} mask completed.")
                            gc.collect()
                            warnings.warn(f'Extracted {self.rois[i]} for {folder}.')

                    self.masks = None

                #             mask = 0.
                #             sitk_mask = 0.
                #
                #             save dictionary to json...
                #             dic['counts'] = counts
                #             save_obj(dic, f"{path}/pkl/{folder}.pkl")
                #             sitk_mask1 = sitk.GetArrayFromImage(sitk_mask)
                #
                #         sitk_mask1[sitk_mask1 > 0] = 1
                #         if i == 0:
                #             shape = sitk_mask1.shape
                #             mask.append(np.zeros(shape))
                #             counts.append(0)
                #             # should be GTV
                #             # something wrong with EXTERNAL export...
                #             mask.append(sitk_mask1 * 2)
                #         # elif i == 1:
                #         #     # GTV contour
                #         #     mask.append(sitk_mask1)
                #         else:
                #             # any other OARS...
                #             mask.append(sitk_mask1)
                #         counts.append(np.sum(sitk_mask1))
                #         sitk_mask1 = None
                #         mask = np.argmax(np.array(mask), axis=0)
                #         # turns out we don't care that much about CTV...
                #         # mask[mask>0] -= 1
                #     if slices is True:
                #         # for 2D DDNN
                #         os.makedirs(f"{path}/mask-slices/", exist_ok=True)
                #         # os.makedirs (f'{path}/img-slices2/', exist_ok=True)
                #         # img = sitk.GetArrayFromImage(sitk_img)
                #
                #         print(f"Saving {folder} as individual slices.")
                #
                #         for i, slice_ in enumerate(mask):
                #             if slice_.max() > 0:
                #
                #                 np.save(f"{path}/mask-slices/{folder}_{i}.npy", slice_)
                #                 # np.save(f'{path}/img-slices2/{folder}_{i}.npy', img[i])
                #
                #     sitk_mask=None
                #     self.masks = None
                #     except Exception:
                #         print(f'Saving numpy of {folder}. ')
                #         os.makedirs (f'{path}/masks/', exist_ok=True)
                #         np.save (f'{path}/masks/{folder}', self.masks)
                #         warnings.warn(f"Couldn't Save .nrrd for {folder}.")
                #
                # np.save (f'{path}/masks/{folder}', self.masks)
                # npad = ((1, 0), (0, 0), (0, 0), (0, 0))
                # warnings.warn(f'Padding mask for {folder}.')
                # mask = np.pad(self.masks , pad_width=npad, mode='constant', constant_values=0)
                # mask = np.argmax(mask, axis=0)
                # warnings.warn(f'Saving mask for patient id: {folder}.')
                # warnings.warn(f'Max mask value is: {mask.max()}.')
                # assert mask.max() > 0
                # didn't work as expected...
                # have to include background class of all zeros...
                # self.mask_final = np.stack([np.zeros(self.masks[0].shape), self.masks], axis=0)
                # should have dim == 5 on axis == 0
                # self.mask_ = np.copy(self.masks)
                # for n, mask in enumerate(self.mask_):
                #     self.mask_[n] *= n
                # self.mask_ = np.argmax(self.mask_, axis=0)
                #
                # added this to enable effective pyradiomic feature extraction
                # if mode == 'itk':
                #     try:
                #         os.makedirs (f'{path}/itk/masks/', exist_ok=True)
                #         for n, mask in enumerate(self.masks):
                #             if n != 0:
                #                 os.makedirs (f'{path}/itk/masks/{folder}/', exist_ok=True)
                #                 sitk_mask = sitk.GetImageFromArray(mask)
                #                 sitk_mask.CopyInformation(self.orig_sitk_img)
                #                 sitk.WriteImage(sitk_mask, f'{path}/itk/masks/{folder}/{n}.nrrd')
                #     except:
                #         warnings.warn(f"Couldn't Save .nrrd masks for {folder}." )

        if "masks_save" not in exclude:

            roi = self.masks
            start = 0
            os.makedirs(f"{path}/plot_masks/", exist_ok=True)
            for roiIndex, mask in enumerate(roi):
                for i, slice in enumerate(mask):
                    if roiIndex == 1:
                        if np.array(slice).sum() > 5:  # finds first layer with a mask
                            start = i
                            break
                    else:
                        break

            argmax_roi = np.argmax(roi[::-1], axis=0)
            fig = pyplot.figure(
                figsize=(rows * 5, columns * 5)
            )  # creates matplotlib Figure object

            for i in range(1, rows * columns + 1):
                try:
                    subFig = fig.add_subplot(rows, columns, i)
                    subFig.imshow(argmax_roi[start + i])
                except:
                    break
            fig.savefig(f"{path}/plot_masks/{folder}_masks_{start}.png")
            pyplot.close(fig)

        self.masks = None
        self.image = None
        self.orig_sitk_img = None

class MTTransform(object):
    def __call__(self, sample):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample):
        raise NotImplementedError("You need to implement the undo_transform() method.")

class HistogramClipping(MTTransform):
    def __init__(
        self,
        percent=False,
        min_percentile=84.0,
        max_percentile=99.0,
        min_hu=-500,
        max_hu=1000.,
    ):

        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.percent = percent

    def __call__(self, img, mask=None):

        array = np.copy(img)

        if self.percent is True:
            percentile1 = np.percentile(array, self.min_percentile)
            percentile2 = np.percentile(array, self.max_percentile)
            array[array <= percentile1] = percentile1
            array[array >= percentile2] = percentile2
        else:
            array[array <= self.min_hu] = self.min_hu
            array[array >= self.max_hu] = self.max_hu

        if mask is not None:
            return array, mask
        else:
            self.img = array
            return array

    @staticmethod
    def output_dic(self):

        dic = {}
        dic['mean'] = np.mean(self.img)
        dic['std'] = np.std(self.img)

        return dic

class Info:
    def __init__(self, ending=''):
        self.info = {}
        self.ending = ending

    def add_patient(self, name, roi):
        self.info[name] = roi

    def export(self, path):
        if path[-1]!='/':
            path+='/'
        try:
            df = pd.DataFrame.from_dict(self.info, orient="index")
            df.to_csv(f"{path}overall_info_{self.ending}.csv", encoding="utf-8", header=True)
        except Exception:
            warnings.warn('Exporting as .json')
            with open(f"{path}overall_info.json", "w") as outfile:
                json.dump(self.info, outfile)



class BadFiles:
    def __init__(self):
        self.bad = []

    def add_bad(self, name):
        self.bad.append(name)

    def export(self, path):
        df = pd.DataFrame(self.bad)
        df.to_csv(f"{path}/badOnes_2.csv", encoding="utf-8", header=True)


class BinaryMask:
    def __init__(self, path):
        nArr = np.load(path)
        self.gtv = nArr[0]
        self.lctv = nArr[1]
        self.rctv = nArr[2]
        self.ext = nArr[3]
        self.shape = nArr.shape


class Projections:
    def __init__(self, mask):
        self.axial = np.sum(mask, axis=0)
        self.sagittal = np.flipud(np.sum(mask, axis=2))
        self.coronal = np.flipud(np.sum(mask, axis=1))

        # binary projections
        self.axialBin = np.copy(self.axial)
        self.axialBin[self.axialBin > 0] = 1
        self.sagittalBin = np.copy(self.sagittal)
        self.sagittalBin[self.sagittalBin > 0] = 1
        self.coronalBin = self.coronal
        self.coronalBin[self.coronalBin > 0] = 1

    def view(self, view="axial"):
        if view == "axial":
            pyplot.imshow(self.axial)
        elif view == "sagittal":
            pyplot.imshow(self.sagittal)
        elif view == "coronal":
            pyplot.imshow(self.coronal)

    def crop(self, img, cropy, cropx):
        y, x = img.shape
        x1 = min([i for i, a in enumerate(img.T.tolist()) if max(a) > 0])
        x2 = max([i for i, a in enumerate(img.T.tolist()) if max(a) > 0])
        y1 = min([i for i, a in enumerate(img.tolist()) if max(a) > 0])
        y2 = max([i for i, a in enumerate(img.tolist()) if max(a) > 0])
        startx = int((x2 + x1 - cropx) / 2)
        starty = int((y2 + y1 - cropy) / 2)
        if startx < 0:
            startx = 0
        if starty < 0:
            starty = 0

        return img[starty : starty + cropy, startx : startx + cropx]


class Model:
    def __init__(self, arrStack, keys):
        self.stack = arrStack
        self.flat = np.array([a.flatten() for a in arrStack])
        self.keys = keys

    def kMeans(self, nClusters=3):
        self.kClusters = cluster.KMeans(n_clusters=nClusters).fit(self.reduced)
        self.klabels = self.kClusters.labels_

    def kMeansPick2(self, pick2=[0, 1], nClusters=3):
        """
        Very niche function.
        Function grabs only 2 components to cluster wtih instead of all components.
        """
        self.kClusters = cluster.KMeans(n_clusters=nClusters).fit(
            [[a[pick2[0]], a[pick2[1]]] for a in self.reduced]
        )  # definitely optimizable

    def exportKMeans(self, path, imgs):
        """
        :params:
        - path: path of final folderName
        - imgs: [[k,v] for k, v in projDict.iteritems()]
        :returns:
        - labels: dictionary of {patientID: label} --> append to pd.DataFrame to record labels
        """
        labels, keys = {}, self.keys
        if len(self.stack) == len(self.kClusters.labels_):
            subDirs, tempPath = path.split("/"), ""
            for dir in subDirs:
                tempPath += dir + "/"
                try:
                    os.mkdir(tempPath)
                except Exception as e:
                    pass
            for i, n in enumerate(self.kClusters.labels_):
                try:

                    temp = np.copy(self.stack[i])
                    temp[temp == 1] = 255  # this is only for binary masks
                    misc.imsave(f"{path}/{n}_{keys [i]}.jpg", temp)
                    labels[keys[i]] = n
                except Exception as e:
                    print(e)
            return labels
        else:
            print(
                "The number of images do not match the number of data points of k-Means Clustering."
            )


class Principal(Model):
    def __init__(self, arrStack, keys, n=2):
        super().__init__(arrStack, keys)
        self.model = decomposition.PCA(n_components=n).fit(self.flat)
        self.reduced = decomposition.PCA(n_components=n).fit_transform(self.flat)
        self.scatter = self.reduced.transpose()

    def exportComponents(self, path):  # needs path fixing
        fig, ax = pyplot.subplots(
            self.model.n_components_,
            1,
            figsize=(5, 3 * self.model.n_components_),
            dpi=300,
            subplot_kw={"xticks": (), "yticks": ()},
        )
        ax = ax.ravel()
        for i in range(len(self.model.components_)):
            pixels = self.model.components_[i].reshape(-1, 200)
            ax[i].imshow(np.flipud(pixels), cmap="viridis")
            ax[i].set_title("Component - " + str(i + 1), fontsize=8)
        fig.savefig(path)
        pyplot.close()

    def reduceData(self, data):
        self.reducedData = self.model.transform(data)


class TSNE(Model):
    def __init__(self, arrStack, keys, n=2):
        super().__init__(arrStack, keys)
        self.reduced = manifold.TSNE(n_components=n).fit_transform(self.flat)
        self.scatter = self.reduced.transpose()


class UMAP(Model):
    def __init__(self, arrStack, keys, n=2):
        super().__init__(arrStack, keys)
        self.reduced = umap.UMAP(n_components=n).fit_transform(self.flat)
        self.scatter = self.reduced.transpose()
