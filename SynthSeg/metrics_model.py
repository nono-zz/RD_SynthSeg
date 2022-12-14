"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
# from keras.losses import CosineSimilarity

# third-party imports
from ext.lab2im import layers


def metrics_model(input_model, label_list, metrics='dice'):

    # get prediction
    last_tensor = input_model.outputs[0]
    input_shape = last_tensor.get_shape().as_list()[1:]

    # check shapes
    n_labels = input_shape[-1]
    label_list = np.unique(label_list)
    assert n_labels == len(label_list), 'label_list should be as long as the posteriors channels'

    # get GT and convert it to probabilistic values
    labels_gt = input_model.get_layer('labels_out').output
    labels_gt = layers.ConvertLabels(label_list)(labels_gt)
    labels_gt = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels_gt)
    labels_gt = KL.Reshape(input_shape)(labels_gt)

    # make sure the tensors have the right keras shape
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    labels_gt._keras_shape = tuple(labels_gt.get_shape().as_list())

    if metrics == 'dice':
        last_tensor = layers.DiceLoss()([labels_gt, last_tensor])

    elif metrics == 'wl2':
        last_tensor = layers.WeightedL2Loss(target_value=5)([labels_gt, last_tensor])

    else:
        raise Exception('metrics should either be "dice or "wl2, got {}'.format(metrics))

    # create the model and return
    model = Model(inputs=input_model.inputs, outputs=last_tensor)
    return model


# def metrics_model_distillation(teacher_model, student_model, metrics='cosine'):
    
#     # get prediction
#     last_tensor_list = student_model.outputs[2:]
#     cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
#     loss = 0
#     # check shapes
#     label_list = teacher_model.outputs[1:]
#     for i, x in enumerate(last_tensor_list):
#         assert last_tensor_list[i].shape[-1] == label_list[i].shape[-1], '{i}th label_list shape does not match!!'
        
#         last_tensor = last_tensor_list[i]
        
#         loss += cosine_loss(last_tensor_list[i], label_list[i])
        

#     # create the model and return
#     # model = Model(inputs=teacher_model.inputs, outputs=loss)
#     return loss

def metrics_model_distillation(pred, label_list, metrics='cosine'):
    
    # get prediction
    last_tensor_list = pred[2:]
    label_list = label_list[1:]
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    loss = 0
    # check shapes
    for i, x in enumerate(last_tensor_list):
        assert last_tensor_list[i].shape[-1] == label_list[i].shape[-1], '{i}th label_list shape does not match!!'
        
        last_tensor = last_tensor_list[i]
        
        loss += cosine_loss(last_tensor_list[i], label_list[i])
        
    return loss


class IdentityLoss(object):
    """Very simple loss, as the computation of the loss as been directly implemented in the model."""
    def __init__(self, keepdims=True):
        self.keepdims = keepdims

    def loss(self, y_true, y_predicted):
        """Because the metrics is already calculated in the model, we simply return y_predicted.
           We still need to put y_true in the inputs, as it's expected by keras."""
        loss = y_predicted

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss
