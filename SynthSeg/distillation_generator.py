
import keras
import os
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from ext.lab2im import utils

from ext.lab2im import edit_volumes

# from ext.neuron import models as nrn_models

class CustomDataGen(keras.utils.Sequence):
    
    def __init__(self):
        
        self.root = '/home/datasets/BraTs/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
        subjects = os.listdir(self.root)
        subjects.sort()
        
        self.path_images = []
        
        for subject in subjects:
            subject_dir = os.path.join(self.root, subject)
            subject_path = os.path.join(subject_dir, subject + '_flair.nii.gz')
            
            self.path_images.append(subject_path)
        
            
    
    def __getitem__(self, index):
        image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_image = self.path_images[index],
                                                                    n_levels = 5,
                                                                    target_res = 1,
                                                                    crop = 192,
                                                                    min_pad = None,
                                                                    path_resample = None)
        
        return image

    
    def __len__(self):
        return len(self.path_images)
    
    
def preprocess(path_image, n_levels, target_res, crop=None, min_pad=None, path_resample=None):
    
    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)

    # resample image if necessary
    if target_res is not None:
        target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
        if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
            im_res = target_res
            im, aff = edit_volumes.resample_volume(im, aff, im_res)
            if path_resample is not None:
                utils.save_volume(im, aff, h, path_resample)

    # align image
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    if n_channels == 1:
        im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)
    else:
        for i in range(im.shape[-1]):
            im[..., i] = edit_volumes.rescale_volume(im[..., i], new_min=0., new_max=1.,
                                                     min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    if min_pad is not None:  # in SynthSeg predict use crop flag and then if used do min_pad=crop else min_pad = 192
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # add batch and channel axes
    im = utils.add_axis(im) if n_channels > 1 else utils.add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx
    
    
    
def postprocess(post_patch, shape, pad_idx, crop_idx, n_dims,
                labels_segmentation, keep_biggest_component, aff, im_res, topology_classes=None):

    # get posteriors
    post_patch = np.squeeze(post_patch)
    if topology_classes is None:
        post_patch = edit_volumes.crop_volume_with_idx(post_patch, pad_idx, n_dims=3, return_copy=False)

    # keep biggest connected component
    if keep_biggest_component:
        tmp_post_patch = post_patch[..., 1:]
        post_patch_mask = np.sum(tmp_post_patch, axis=-1) > 0.25
        post_patch_mask = edit_volumes.get_largest_connected_component(post_patch_mask)
        post_patch_mask = np.stack([post_patch_mask]*tmp_post_patch.shape[-1], axis=-1)
        tmp_post_patch = edit_volumes.mask_volume(tmp_post_patch, mask=post_patch_mask, return_copy=False)
        post_patch[..., 1:] = tmp_post_patch

    # reset posteriors to zero outside the largest connected component of each topological class
    if topology_classes is not None:
        post_patch_mask = post_patch > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = edit_volumes.get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch[..., idx] *= tmp_mask
        post_patch = edit_volumes.crop_volume_with_idx(post_patch, pad_idx, n_dims=3, return_copy=False)

    # renormalise posteriors and get hard segmentation
    if keep_biggest_component | (topology_classes is not None):
        post_patch /= np.sum(post_patch, axis=-1)[..., np.newaxis]
    seg_patch = labels_segmentation[post_patch.argmax(-1).astype('int32')].astype('int32')

    # paste patches back to matrix of original image size
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype='int32')
        posteriors = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
        posteriors[..., 0] = np.ones(shape)  # place background around patch
        if n_dims == 2:
            seg[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] = seg_patch
            posteriors[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], :] = post_patch
        elif n_dims == 3:
            seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
            posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = post_patch
    else:
        seg = seg_patch
        posteriors = post_patch

    # align prediction back to first orientation
    seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=n_dims, return_copy=False)
    posteriors = edit_volumes.align_volume_to_ref(posteriors, np.eye(4), aff_ref=aff, n_dims=n_dims, return_copy=False)

    # compute volumes
    volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
    volumes = np.around(volumes * np.prod(im_res), 3)

    return seg, posteriors, volumes