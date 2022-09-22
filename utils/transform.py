import numpy as np
import numbers, random, skimage, warnings, cv2
from skimage.filters import threshold_otsu
from scipy.ndimage.interpolation import rotate, zoom
from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage.measurements as measure
import SimpleITK as sitk
import elasticdeform as edf

"""
Adapted from medical torch for 3D volumes of numpy arrays.
"""


class MTTransform(object):
    def __call__(self, sample):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample):
        raise NotImplementedError("You need to implement the undo_transform() method.")


class UndoCompose(object):
    def __init__(self, compose):
        self.transforms = compose.transforms

    def __call__(self, sample):
        for t in self.transforms:
            img = t.undo_transform(sample)
        return img


class UndoTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform.undo_transform(sample)


class Compose(object):

    """

    Composes several transforms together.
    Modiefied to edit both image & mask at one time.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>>     transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert mask.max() > 0
        max_ = mask.max()
        max_img = img.max()
        min_img = img.min()

        shape = mask.shape

        for i, t in enumerate(self.transforms):
            img, mask = t(img, mask)
            mask = np.round(mask)
            # make sure transformation doesn't mess up mask...
            # do the masks look like how they ought to look?
            # print(count, max_, mask.max())
            # check transformations doing right thing //
            if i < len(self.transforms)-2:
                # assert mask.max() == max_
                # assert mask.min() == 0
                try:
                    assert img.max() == max_img
                except Exception:
                    warnings.warn(f'Max value of img changed at transform {i}.')
                    max_img = img.max()
                try:
                    assert mask.max() == max_
                except Exception:
                    warnings.warn(f'Max value of mask changed at transform {i}.')
                    max_ = mask.max()
                    # max_img = img.max()
                try:
                    assert img.min() == min_img
                except Exception:
                    warnings.warn(f'Min value of img changed at transform {i}.')
                    min_img = img.min()

            else:

                try:
                    assert mask.max() == max_
                except Exception:
                    warnings.warn('Cropped out max class value.')

        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class sitkZoom3D(MTTransform):
    def __init__(self, spacing=np.arange(0.96, 1.04, 0.01)):
        self.spacing = np.random.choice(spacing)

    @staticmethod
    def resample(image, mode, new_spacing=np.array((1.0, 1.0, 3.0))):

        # originally taken from https://github.com/SimpleITK/SimpleITK/issues/561
        resample = sitk.ResampleImageFilter()
        if mode == "linear":
            resample.SetInterpolator = sitk.sitkLinear  # use linear to resample image
            image = sitk.SmoothingRecursiveGaussian(image, 2.0)
        else:
            # use sitkNearestNeighbor interpolation
            # best for masks, no gaussian smoothing required...
            resample.SetInterpolator = sitk.sitkNearestNeighbor

        orig_size = np.array(image.GetSize(), dtype=np.int)
        orig_spacing = np.array(image.GetSpacing())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        new_spacing = new_spacing
        resample.SetOutputSpacing(new_spacing)

        # new_spacing[:2] = orig_spacing[:2]
        # resample.SetOutputPixelType = sitk_image.GetPixelIDValue()
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int)  #  Image dimensions are in integers
        new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)

        # we can use this with or without gaussian smoothing, prevent analising

        image = resample.Execute(image)

        return image

    def __call__(self, img, mask=None):
        # calculate new spacing ...
        # might be better and a whole lot faster than numpy...
        zoom_by = self.spacing
        new_spacing = np.array((zoom_by, zoom_by, 3.0))
        img = self.resample(img, mode="linear", new_spacing=new_spacing)

        if mask is not None:
            mask = self.resample(mask, mode="nearest", new_spacing=new_spacing)
            return img, mask
        else:
            return img


class RandomZoom3D(MTTransform):
    def __init__(self, p=1.0, zoom_factors=np.arange(0.9, 1.1, 0.01)):

        self.zoom_by = np.random.choice(zoom_factors)
        self.p = p

    @staticmethod
    def clipped_zoom(img, zoom_factor, method="linear"):
        """
        Center zoom in/out of the given image and returning an enlarged/shrinked view of
        the image without changing dimensions
        Args:
            img : Image array
            zoom_factor : amount of zoom as a ratio (0 to Inf)
        """

        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)

        # if method == 'li':
        #     a = np.sum(img)
        #     if a < 1000:
        #         # do not downsample...
        #         # change zoom factor...
        #         zoom_factor = np.random.choice(np.arange(1.00, 1.06, 0.01))

        height, width = img.shape[:2]  # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

        ### Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1, x1, y2, x2])
        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (
            (height - resize_height) // 2,
            (width - resize_width) // 2,
        )
        pad_height2, pad_width2 = (
            (height - resize_height) - pad_height1,
            (width - resize_width) - pad_width1,
        )
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (
            img.ndim - 2
        )

        if method == "linear":
            result = cv2.resize(
                cropped_img,
                (resize_width, resize_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            result = cv2.resize(
                cropped_img,
                (resize_width, resize_height),
                interpolation=cv2.INTER_NEAREST,
            )
        result = np.pad(result, pad_spec, mode="constant")  #'constant'
        assert result.shape[0] == height and result.shape[1] == width

        if len(img.shape) == 3:
            result = result.transpose(2, 0, 1)

        return result

    def __call__(self, img, mask=None):
        flip = random.random()
        if flip < self.p:
            img = self.clipped_zoom(img.copy(), self.zoom_by, method="nearest")
            if mask is not None:
                mask = self.clipped_zoom(mask.copy(), self.zoom_by, method="nearest")
                return img, mask
            else:
                return img
        else:
            return img, mask

        # Previous clipped_zoom function...
        # """
        #     Taken from: https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
        #     """
        # much slower than open-cv
        #
        # if len(img.shape) == 3:
        #     img = img.transpose(1, 2, 0)
        #
        # if method == "linear":
        #     # bilinear interpolation
        #     mode = 1
        # else:
        #     # nearest neighbour interpolation
        #     mode = 0
        #
        # h, w = img.shape[:2]
        #
        # # For multichannel images we don't want to apply the zoom factor to the RGB
        # # dimension, so instead we create a tuple of zoom factors, one per array
        # # dimension, with 1's for any trailing dimensions after the width and height.
        # zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
        #
        # # Zooming out
        # if zoom_factor < 1:
        #
        #     # Bounding box of the zoomed-out image within the output array
        #     zh = int(np.round(h * zoom_factor))
        #     zw = int(np.round(w * zoom_factor))
        #     top = (h - zh) // 2
        #     left = (w - zw) // 2
        #
        #     # Zero-padding
        #     out = np.zeros_like(img)
        #     out[top : top + zh, left : left + zw] = zoom(img, zoom_tuple, order=mode)
        #
        # # Zooming in
        # elif zoom_factor > 1:
        #
        #     # Bounding box of the zoomed-in region within the input array
        #     zh = int(np.round(h / zoom_factor))
        #     zw = int(np.round(w / zoom_factor))
        #     top = (h - zh) // 2
        #     left = (w - zw) // 2
        #
        #     out = zoom(img[top : top + zh, left : left + zw], zoom_tuple, order=mode)
        #
        #     # `out` might still be slightly larger than `img` due to rounding, so
        #     # trim off any extra pixels at the edges
        #     trim_top = (out.shape[0] - h) // 2
        #     trim_left = (out.shape[1] - w) // 2
        #     out = out[trim_top : trim_top + h, trim_left : trim_left + w]
        #
        # # If zoom_factor == 1, just return the input array
        # else:
        #     out = img
        #
        # if len(img.shape) == 3:
        #     return out.transpose(2, 0, 1)
        # else:
        #     return out


# define function that saves
class RandomCrop3D(MTTransform):
    def __init__(self, window=5, factor=512, mode="train", data="RADCURE", crop_as="3D"):

        # can we add this to overide norm?
        # for volume just change the window...
        self.factor = factor
        self.window = window
        self.mode = mode
        self.data = data
        self.crop_as = crop_as

    @staticmethod
    def segment_head(img):
        # function to make (fake) external for center cropping of image...
        otsu = threshold_otsu(img)  # Compute the Ostu threshold
        binary_img = np.array(img > otsu, dtype=int )  # Convert image into binary mask (numbers bigger then otsu set to 1, else 0)
        fill = binary_fill_holes(binary_img)  # Fill any regions of 0s that are enclosed by 1s
        return fill

    def get_shifts(self, img):
        # Assumes there is NO external contour then use this...
        # image if '3D' is a binary or similar mask...
        # if mask is not avaliable use image with otsu thresholding...
        # slice_ = img[self.window]
        # fill = self.segment_head(img)
        # assert img.max() > 0
        # only do this if of type tensor...
        try:
            shape = img.shape
        except Exception:
            shape = img.size()
            img = img.cpu().numpy()

        if len(shape)==4:
            img = img[0,:,:,:]
            shape = img.shape
        elif len(shape)==5:
            img=img[0,0,:,:,:]
            shape = img.shape

        if self.mode == 'test':
            img = img[shape[0]//2:shape[0]-shape[0]//3]
            img = self.segment_head(img)
            com_ = measure.center_of_mass(img)
            img = img[int(com_[0])]
            self.center = [com_[0], com[0], com[1]]
        else:
            # prodcues a mask, where we take COM from...
            com = measure.center_of_mass(img)
            com = [com[0], com[1], com[2]]
            assert len(com) == 3
            self.center = com


    def get_params(self, img):
        if len(img.shape) == 3:
            self.z, self.y, self.x = img.shape
        elif len(img.shape) == 4:
            self.b, self.z, self.y, self.x = img.shape
        elif len(img.shape) == 5:
            self.b, self.c, self.z, self.y, self.x = img.shape
        else:
            self.y, self.x = img.shape
        assert self.x == self.y

    # @staticmethod
    def get_crop(self, img, mask=None):

        try:
            shape = img.shape
        except Exception:
            shape = img.size()

        if self.mode != "test":
            centerz = np.int(self.center[0]) if self.center is not None else self.z // 2
            centerx = np.int(self.center[2]) if self.center is not None else self.x // 2
            centery = np.int(self.center[1]) if self.center is not None else self.y // 2
        else:
            centerx = np.int(self.center[1]) if self.center is not None else self.x // 2
            centery = np.int(self.center[0]) if self.center is not None else self.y // 2

        startx = np.int(centerx) - (self.factor // 2)
        starty = np.int(centery) - (self.factor // 2)

        if self.mode == "train":
            assert len(self.center) == 3
            a = np.arange(-64, 64)
            startx += np.random.choice(a)
            starty += np.random.choice(a)

            try:
                assert startx > 0
            except Exception:
                warnings.warn('COM of mask < 1/4 of crop factor in x.')
                startx = 1
            try:
                assert starty > 0
            except Exception:
                warnings.warn('COM of mask < 1/4 of crop factor in y.')
                starty = 1
            try:
                assert startx < (self.x - self.factor//2 - 1)
            except Exception:
                warnings.warn('Startx needs to be changed for effective crop.')
                startx = self.x - self.factor//2 - 2
            try:
                assert starty < (self.y - self.factor//2 - 1)
            except Exception:
                warnings.warn('Starty needs to be changed for effective crop.')
                starty = self.y - self.factor//2 - 2

        else:
            try:
                assert startx > 0
            except Exception:
                warnings.warn('COM of mask < 1/4 of crop factor in x.')
                startx = 1
            try:
                assert starty > 0
            except Exception:
                warnings.warn('COM of mask < 1/4 of crop factor in y.')
                starty = 1
            try:
                assert startx < (self.x - self.factor//2 - 1)
            except Exception:
                warnings.warn('Startx needs to be changed for effective crop.')
                startx = self.x - self.factor//2 - 2
            try:
                assert starty < (self.y - self.factor//2 - 1)
            except Exception:
                warnings.warn('Starty needs to be changed for effective crop.')
                starty = self.y - self.factor//2 - 2

        # Use during training.
        # for vlidation stay cropped around GTV...
        # ie, comment out for testing...
        if self.mode != "test":
            if shape[0] > self.window*2: # 128
                warnings.warn(f'Cropping images/masks from {shape[0]} to 120.')
                # self.window = 56 # 64
                a = np.arange(-64,64)
                if self.mode == 'train':
                    centerz += np.random.choice(a)
                end = shape[0] - self.window

                if centerz < self.window - 1:
                    centerz = self.window + self.window // 2
                elif centerz > end:
                    centerz = centerz - self.window // 2

                bottom = centerz - self.window  # //2 # input 54//2
                top = centerz + self.window  # //2+1 # input 54 //2

                try:
                    assert bottom > 0
                except Exception:
                    warnings.warn('Cropping starting from z==1.')
                    bottom = 0
                    top = 0 + self.window*2
                try:
                    assert top < shape[0]
                except Exception:
                    warnings.warn(f'Cropping ending at {shape[0]}.')
                    bottom = shape[0] - 1 - self.window*2
                    top = shape[0] - 1

                if len(shape) == 3:

                    img = img[int(bottom): int(top), starty: starty + self.factor, startx: startx + self.factor]
                    if mask is not None:
                        mask = mask[int(bottom): int(top), starty: starty + self.factor, startx: startx + self.factor]

                elif len(shape) == 4:

                    img = img[:, int(bottom): int(top), starty: starty + self.factor, startx: startx + self.factor]
                    if mask is not None:
                        mask = mask[:, int(bottom): int(top), starty: starty + self.factor, startx: startx + self.factor]

                elif len(shape) == 5:

                    img = img[ :, :, int(bottom): int(top), starty: starty + self.factor, startx: startx + self.factor]
                    if mask is not None:
                        mask = mask[ :, :, int(bottom): int(top), starty: starty + self.factor, startx: startx + self.factor]
                else:

                    img = img[starty: starty + self.factor, startx: startx + self.factor]
                    if mask is not None:
                        mask = mask[starty: starty + self.factor, startx: startx + self.factor]

                if mask is not None:
                    return img, mask
                else:
                    return img

        if len(shape) == 3:
            if self.mode=='test':
                starty+=-15
            img = img[:, starty: starty + self.factor, startx: startx + self.factor]
            if mask is not None:
                mask = mask[:, starty: starty + self.factor, startx: startx + self.factor]

        elif len(shape) == 4:

            img = img[:, :, starty: starty + self.factor, startx: startx + self.factor]
            if mask is not None:
                mask = mask[:, :, starty: starty + self.factor, startx: startx + self.factor]

        elif len(shape) == 5:

            img = img[ :, :, :, starty: starty + self.factor, startx: startx + self.factor]
            if mask is not None:
                mask = mask[ :, :, :, starty: starty + self.factor, startx: startx + self.factor]
        else:
            img = img[starty: starty + self.factor, startx: startx + self.factor]
            if mask is not None:
                mask = mask[starty: starty + self.factor, startx: startx + self.factor]

        if mask is not None:
            if self.mode == 'test':
                return img, mask, (starty, startx)
            else:
                return img, mask
        else:
            if self.mode == 'test':
                return img, (starty, startx)
            else:
                return img

    def __call__(self, img, mask=None, mask2=None):

        # initiate parameters (get shifing coeff)

        if self.mode=='test':
            self.get_params(img)
            self.get_shifts(img)
        else:
            self.get_params(img)
            self.get_shifts(mask)

        if mask is not None:
            assert mask.max() > 0
            if self.mode == 'test':
                img, mask, center = self.get_crop(img, mask)
                return img, mask, center
            else:
                img, mask = self.get_crop(img, mask)
                if mask.shape != (img.shape[0] ,self.factor, self.factor):
                    warnings.warn(f'Bad mask shape...recropping. {mask.shape}')
                return img, mask
        else:
            img, c = self.get_crop(img)
            return img, c

class RandomRotation3D(MTTransform):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    :param axis: Axis of the rotation.
    """

    def __init__(self, degrees=5, axis=(1, 2), p=1.):
        if isinstance(degrees, numbers.Number):

            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")

            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.axis = axis
        self.p = p

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, mask=None):
        flip = random.random()
        if flip < self.p:
            angle = self.get_params(self.degrees)
            img = rotate(img, angle, axes=self.axis, reshape=False, order=0)
            if mask is not None:
                if len(mask.shape) == 2:
                    self.axis = (0, 1)
                mask = rotate(mask, angle, axes=self.axis, reshape=False, order=0)
                return img, mask
            else:
                return img
        else:
            return img, mask


class RandomFlip3D(MTTransform):

    """Make a symmetric inversion of the different values of each dimensions.
    (randomized)
    """

    def __init__(self, axis=2, flip_labels=True):
        self.axis = 2
        # TRUE FOR OAR SEGMENTATION...
        self.flip_labels=flip_labels

    def __call__(self, img, mask=None):

        self.coin = random.random()
        if self.coin > 0.5:
            # flip image
            img = np.flip(img, axis=self.axis).copy()
            if mask is not None:
                if len(mask.shape) == 2:
                    self.axis = 1
                elif len(mask.shape) ==4:
                    self.axis += 1
                mask = np.flip(mask, axis=self.axis).copy()
                mask_ = np.zeros(mask.shape)
                # we would also need to flip mask labels...
                if self.flip_labels is True:
                    flipped_chosen = [0,1,2,3,4,5,7,6,9,8,11,10,13,12,15,14,17,16,18,19]
                    for i, val in enumerate(flipped_chosen):
                        mask_[mask==i] = val
                else:
                    pass

                return img, mask_
            else:
                return img
        else:
            # do nothing...
            if mask is not None:
                return img, mask
            else:
                return img


class ElasticTransform3D(MTTransform):
    def __init__( self, sigma=25, points=3, axis=(1, 2), order=0, p=1.0,\
                  mode="nearest", prefilter=True):

        self.sigma = sigma
        self.pts = points
        self.ax = axis
        self.mode = mode
        self.prefilt = prefilter
        self.order = order
        self.p = p

    @staticmethod
    def choose_sig(sigma):
        a = np.arange(sigma//2, sigma+1)
        sigma = np.random.choice(a)
        return sigma

    def __call__(self, image, mask=None):
        # only run augmentation if flip below
        # certain probability...
        flip = random.random()
        if flip < self.p:
            sigma = self.choose_sig(self.sigma)
            if mask is not None:
                # can test this out...
                # self.mode = ["constant", "nearest"]
                # self.order = [1, 0]
                # want to make ure the same random deform is applied...
                if len(mask.shape) == 2:
                    self.ax = [(1, 2), (0, 1)]

                image, mask = edf.deform_random_grid(
                    [image, mask],
                    sigma=sigma,
                    points=self.pts,
                    axis=self.ax,
                    mode=self.mode,
                    order=self.order,
                    prefilter=self.prefilt,
                )

                return image, mask

            else:
                image = edf.deform_random_grid(
                    image,
                    sigma=sigma,
                    points=self.pts,
                    axis=self.ax,
                    mode=self.mode,
                    order=self.order,
                    prefilter=self.prefilt,
                )
                return image
        else:
            return image, mask


class AdditiveGaussianNoise(MTTransform):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask=None):
        # rdict = {}
        # input_data = sample['input']
        if np.random.randn() > 0.5:
            shape = img.shape
            noise = np.random.normal(self.mean, self.std, shape)
            img = img + noise

        if mask is not None:
            return img, mask
        else:
            return img


class Clahe(MTTransform):
    def __init__(self, clip_limit=3.0, kernel_size=(8, 8), return_="3D"):
        # Default values are based upon the following paper:
        # https://arxiv.org/abs/1804.09400 (3D Consistent Cardiac Segmentation)

        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def __call__(self, img, mask=None):

        if not isinstance(img, np.ndarray):
            raise TypeError("Input sample must be a numpy array.")

        input = np.copy(img)

        if len(input.shape) == 3:
            images = []
            for i, slice in enumerate(input):
                images.append(
                    skimage.exposure.equalize_adapthist(
                        slice, kernel_size=self.kernel_size, clip_limit=self.clip_limit
                    )
                )
            images = np.stack(images)
            return images

        elif len(input.shape) == 2:
            array = skimage.exposure.equalize_adapthist(
                input, kernel_size=self.kernel_size, clip_limit=self.clip_limit
            )

            return array

        else:
            raise ValueError("Input sample must be a 3D or 2D numpy array.")


class HistogramClipping(MTTransform):
    def __init__(
        self,
        percent=False,
        min_percentile=84.0,
        max_percentile=99.0,
        # old clipping messed s**t up, note -296 196 not good.
        min_hu=-500,
        max_hu=1000,
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
            return array


class NormBabe(MTTransform):

    def __init__(self, mean=False, std=False, min=-196.0, max=296.0, type="standard"):

        self.mean = np.float(mean)
        self.std = np.float(std)
        self.min = min
        self.max = max
        self.type = type

    def normalize(self, image):
        # will normalize between zero and 1
        image = (image - self.min) / (self.max - self.min)
        image[image > 1] = 1.0
        image[image < 0] = 0.0

        return image

    def __call__(self, img, mask=None):

        array = np.copy(img)
        if self.type == "standard":
            array = (array - self.mean) / self.std
        else:
            array = self.normalize(array)

        if mask is not None:

            return array, mask

        else:

            return array


class Normalize(MTTransform):

    """
    Normalize a tensor image with mean and standard deviation.
    :param mean: mean value.
    :param std: standard deviation value.
    """

    def __init__(self, mean=None, std=None, verbose=False, min=-200, max=300):

        self.mean = mean
        self.std = std
        self.verbose = verbose
        self.min = min
        self.max = max

    def __call__(self, img, mask=None):

        if self.verbose is not False:
            mean = np.mean(img)
            std = np.std(img)
            return mean, std
        else:
            if self.mean is not None and self.std is not None:
                warnings.warn("Using global normalization.")
                image = (img - self.mean) / self.std
            else:
                warnings.warn("Using image based normalization.")
                image = (img - np.mean(img)) / np.std(img)

            if mask is not None:
                return image, mask
            else:
                return image
