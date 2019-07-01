from keras import backend as K
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import MaxPooling2D
class MaxPoolingMask2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(MaxPoolingMask2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        pooled = K.pool2d(inputs, pool_size, strides, border_mode,
                 dim_ordering, pool_mode='max')
        upsampled = UpSampling2D(size=pool_size)(pooled)
        indexMask = K.tf.equal(inputs, upsampled)
        assert indexMask.get_shape().as_list() == inputs.get_shape().as_list()
        return indexMask
    
    def get_output_shape_for(self, input_shape):
        return input_shape


def unpooling(inputs):
    '''
    do unpooling with indices, move this to separate layer if it works
    1. do naive upsampling (repeat elements)
    2. keep only values in mask (stored indices) and set the rest to zeros
    '''
    x = inputs[0]
    mask = inputs[1]
    mask_shape = mask.get_shape().as_list()
    x_shape = x.get_shape().as_list()
    pool_size = (mask_shape[1] / x_shape[1], mask_shape[2] / x_shape[2])
    on_success = UpSampling2D(size=pool_size)(x)
    on_fail = K.zeros_like(on_success)
    return K.tf.where(mask, on_success, on_fail)


def unpooling_output_shape(input_shape):
    return input_shape[1]
