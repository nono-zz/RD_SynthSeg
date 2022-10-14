import tensorflow as tf
import keras




class CustomAccuracy(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):
        last_tensor_list = y_pred[2:]
        label_list = y_true[1:]
        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        loss = 0
        # check shapes
        for i, x in enumerate(last_tensor_list):
            assert last_tensor_list[i].shape[-1] == label_list[i].shape[-1], '{i}th label_list shape does not match!!'
            
            last_tensor = last_tensor_list[i]
        
        loss += cosine_loss(last_tensor_list[i], label_list[i])
        
        return loss
    
class loss_layer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    def __call__(self, y_true, y_pred):
        last_tensor_list = y_pred[2:]
        label_list = y_true[1:]

        loss = 0
        # check shapes
        for i, x in enumerate(last_tensor_list):
            assert last_tensor_list[i].shape[-1] == label_list[i].shape[-1], '{i}th label_list shape does not match!!'
            
            last_tensor = last_tensor_list[i]
        
        loss += self.cosine_loss(last_tensor_list[i], label_list[i])
        
        return loss
    
if __name__ == '__main__':
    loss_compute = loss_layer()
    x = tf.constant([3,3,3,3])
    y = loss_compute(x,x)