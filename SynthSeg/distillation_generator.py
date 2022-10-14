
import keras
import os
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from ext.lab2im import utils
import csv

from ext.lab2im import edit_volumes
# from SynthSeg.distillation import build_teacher_model
from keras.models import Model
from ext.lab2im import layers

from ext.neuron import models as nrn_models


class CustomDataGen(keras.utils.Sequence):
    
    def __init__(self,
                path_images,
                path_segmentations,
                path_model,
                labels_segmentation,
                n_neutral_labels=None,
                names_segmentation=None,
                path_posteriors=None,
                path_resampled=None,
                path_volumes=None,
                gradients=False,
                flip=True,
                topology_classes=None,
                sigma_smoothing=0.5,
                n_levels=5,
                nb_conv_per_level=2,
                conv_size=3,
                unet_feat_count=24,
                feat_multiplier=2,
                activation='elu',
                recompute=True,):
        
        self.root = '/home/datasets/BraTs/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
        subjects = os.listdir(self.root)
        subjects.sort()
        
        self.path_images = []
        
        for subject in subjects:
            subject_dir = os.path.join(self.root, subject)
            subject_path = os.path.join(subject_dir, subject + '_flair.nii.gz')
            
            self.path_images.append(subject_path)
            
            
        # prepare input/output filepaths
        # self.path_images, path_segmentations, path_posteriors, path_resampled, path_volumes, compute, unique_vol_file = \
        #     prepare_output_files(self.path_images, path_segmentations, path_posteriors, path_resampled, path_volumes, recompute)

        # get label list
        labels_segmentation, _ = utils.get_list_labels(label_list=labels_segmentation)
        if (n_neutral_labels is not None) & flip:
            labels_segmentation, flip_indices, unique_idx = get_flip_indices(labels_segmentation, n_neutral_labels)
        else:
            labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
            flip_indices = None

        # prepare other labels list
        if names_segmentation is not None:
            names_segmentation = utils.load_array_if_path(names_segmentation)[unique_idx]
        if topology_classes is not None:
            topology_classes = utils.load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]

        # prepare volumes if necessary
        # if unique_vol_file & (path_volumes[0] is not None):
        #     write_csv(path_volumes[0], None, True, labels_segmentation, names_segmentation)
            
        
        _, _, n_dims, n_channels, _, _ = utils.get_volume_info(self.path_images[0])
        
        model_input_shape = [None] * n_dims + [n_channels]
        self.teacher_enc = build_teacher_model(path_model=path_model,
                      input_shape=model_input_shape,
                      labels_segmentation=labels_segmentation,
                      n_levels=n_levels,
                      nb_conv_per_level=nb_conv_per_level,
                      conv_size=conv_size,
                      unet_feat_count=unet_feat_count,
                      feat_multiplier=feat_multiplier,
                      activation=activation,
                      sigma_smoothing=sigma_smoothing,
                      flip_indices=flip_indices,
                      gradients=gradients)
        
        # self.teacher_enc.trainable = False    
            
    
    def __getitem__(self, index):
        image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_image = self.path_images[index],
                                                                    n_levels = 5,
                                                                    target_res = 1,
                                                                    crop = 192,
                                                                    min_pad = None,
                                                                    path_resample = None)
        
        
        target = self.teacher_enc.predict(image)
        
        return target[0], target[-1]

    
    def __len__(self):
        return len(self.path_images)
    
    

def get_flip_indices(labels_segmentation, n_neutral_labels):

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondance between labels
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique

    # get unique labels
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)

    # get indices of corresponding labels
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0]

    # build 1d vector to swap LR corresponding labels taking into account neutral labels
    flip_indices = np.zeros_like(labels_segmentation)
    for i in range(len(flip_indices)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])]
        else:
            flip_indices[i] = lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])]

    return labels_segmentation, flip_indices, unique_idx

def prepare_output_files(path_images, out_seg, out_posteriors, out_resampled, out_volumes, recompute):
    
    # check inputs
    assert path_images is not None, 'please specify an input file/folder (--i)'
    assert out_seg is not None, 'please specify an output file/folder (--o)'

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg)
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_resampled = os.path.abspath(out_resampled) if (out_resampled is not None) else out_resampled
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes

    # path_images is a text file
    if basename[-4:] == '.txt':

        # input images
        if not os.path.isfile(path_images):
            raise Exception('provided text file containing paths of input images does not exist' % path_images)
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # define helper to deal with outputs
        def text_helper(path, name):
            if path is not None:
                assert path[-4:] == '.txt', 'if path_images given as text file, so must be %s' % name
                with open(path, 'r') as ff:
                    path = [line.replace('\n', '') for line in ff.readlines() if line != '\n']
                recompute_files = [not os.path.isfile(p) for p in path]
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            unique_file = False
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = text_helper(out_seg, 'path_segmentations')
        out_posteriors, recompute_post, _ = text_helper(out_posteriors, 'path_posteriors')
        out_resampled, recompute_resampled, _ = text_helper(out_resampled, 'path_resampled')
        out_volumes, recompute_volume, unique_volume_file = text_helper(out_volumes, 'path_volume')

    # path_images is a folder
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):

        # input images
        if os.path.isfile(path_images):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = utils.list_images_in_folder(path_images)

        # define helper to deal with outputs
        def helper_dir(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    path = [path] * len(path_images)
                    recompute_files = [True] * len(path_images)
                    unique_file = True
                else:
                    if (path[-7:] == '.nii.gz') | (path[-4:] == '.nii') | (path[-4:] == '.mgz') | (path[-4:] == '.npz'):
                        raise Exception('Output FOLDER had a FILE extension' % path)
                    path = [os.path.join(path, os.path.basename(p)) for p in path_images]
                    path = [p.replace('.nii', '_%s.nii' % suffix) for p in path]
                    path = [p.replace('.mgz', '_%s.mgz' % suffix) for p in path]
                    path = [p.replace('.npz', '_%s.npz' % suffix) for p in path]
                    recompute_files = [not os.path.isfile(p) for p in path]
                utils.mkdir(os.path.dirname(path[0]))
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = helper_dir(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_dir(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, recompute_resampled, _ = helper_dir(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, recompute_volume, unique_volume_file = helper_dir(out_volumes, 'path_volumes', 'csv', '')

    # path_images is an image
    else:

        # input image
        assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                            'please make sure the path and the extension are correct' % path_images
        path_images = [path_images]

        # define helper to deal with outputs
        def helper_im(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    recompute_files = [True]
                    unique_file = True
                else:
                    if ('.nii.gz' not in path) & ('.nii' not in path) & ('.mgz' not in path) & ('.npz' not in path):
                        file_name = os.path.basename(path_images[0]).replace('.nii', '_%s.nii' % suffix)
                        file_name = file_name.replace('.mgz', '_%s.mgz' % suffix)
                        file_name = file_name.replace('.npz', '_%s.npz' % suffix)
                        path = os.path.join(path, file_name)
                    recompute_files = [not os.path.isfile(path)]
                utils.mkdir(os.path.dirname(path))
            else:
                recompute_files = [False]
            path = [path]
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_im(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, recompute_resampled, _ = helper_im(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, recompute_volume, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')

    recompute_list = [recompute | re_seg | re_post | re_res | re_vol for (re_seg, re_post, re_res, re_vol)
                      in zip(recompute_seg, recompute_post, recompute_resampled, recompute_volume)]

    return path_images, out_seg, out_posteriors, out_resampled, out_volumes, recompute_list, unique_volume_file

def write_csv(path_csv, data, unique_file, labels, names, skip_first=True, last_first=False):

    # initialisation
    utils.mkdir(os.path.dirname(path_csv))
    labels, unique_idx = np.unique(labels, return_index=True)
    if skip_first:
        labels = labels[1:]
    if names is not None:
        names = names[unique_idx].tolist()
        if skip_first:
            names = names[1:]
        header = names
    else:
        header = [str(lab) for lab in labels]
    if last_first:
        header = [header[-1]] + header[:-1]
    if (not unique_file) & (data is None):
        raise ValueError('data can only be None when initialising a unique volume file')

    # modify data
    if unique_file:
        if data is None:
            type_open = 'w'
            data = ['subject'] + header
        else:
            type_open = 'a'
        data = [data]
    else:
        type_open = 'w'
        header = [''] + header
        data = [header, data]

    # write csv
    with open(path_csv, type_open) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)

    
    
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


def build_teacher_model(path_model,
                input_shape,
                labels_segmentation,
                n_levels,
                nb_conv_per_level,
                conv_size,
                unet_feat_count,
                feat_multiplier,
                activation,
                sigma_smoothing,
                flip_indices,
                gradients):

    assert os.path.isfile(path_model), "The provided model path does not exist."

    # get labels
    n_labels_seg = len(labels_segmentation)

    if gradients:
        input_image = KL.Input(input_shape)
        last_tensor = layers.ImageGradients('sobel', True)(input_image)
        last_tensor = KL.Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x) + K.epsilon()))(last_tensor)
        net = Model(inputs=input_image, outputs=last_tensor)
    else:
        net = None

    # build UNet
    net = nrn_models.unet(input_model=net,
                          input_shape=input_shape,
                          nb_labels=n_labels_seg,
                          nb_levels=n_levels,
                          nb_conv_per_level=nb_conv_per_level,
                          conv_size=conv_size,
                          nb_features=unet_feat_count,
                          feat_mult=feat_multiplier,
                          activation=activation,
                          batch_norm=-1)
    net.load_weights(path_model, by_name=True)


    return net