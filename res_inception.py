# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import keras
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv3D, AveragePooling3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import add

def _bn_relu(input):
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = input._keras_shape[1] \
        // residual._keras_shape[1]
    stride_dim2 = input._keras_shape[2] \
        // residual._keras_shape[2]
    stride_dim3 = input._keras_shape[3] \
        // residual._keras_shape[3]
    equal_channels = residual._keras_shape[4] \
        == input._keras_shape[4]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=residual._keras_shape[4],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
            )(input)
    return add([shortcut, residual])



def _inception3d(input, reg_factor=1e-4):
    b0 = _conv_bn_relu3D(filters=64, kernel_size=(1, 1, 1),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(input)
    b1 = _conv_bn_relu3D(filters=96, kernel_size=(1, 1, 1),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(input)
    b1 = _conv_bn_relu3D(filters=128, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(b1)
    b2 = _conv_bn_relu3D(filters=16, kernel_size=(1, 1, 1),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(input)
    b2 = _conv_bn_relu3D(filters=32, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(b2)
    b3 = MaxPooling3D(pool_size=(3, 3, 3), 
                      strides=(1, 1, 1), padding='same'
                      )(input) 
    b3 = _conv_bn_relu3D(filters=32, kernel_size=(1, 1, 1),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(b3)
    residual = keras.layers.concatenate([b0, b1, b2, b3], axis=-1)
    return _shortcut3d(input, residual)

class I3Dbuilder(object):
    
    @staticmethod
    def build(input_shape, num_outputs, reg_factor):
        input = Input(shape=input_shape)
        x = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                                strides=(2, 2, 2),
                                kernel_regularizer=l2(reg_factor)
                                )(input)
        x = MaxPooling3D(pool_size=(1, 3, 3), 
                      strides=(1, 2, 2), padding='same'
                      )(x)                                
        x = _conv_bn_relu3D(filters=64, kernel_size=(1, 1, 1),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(x)                        
        x = _conv_bn_relu3D(filters=192, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(x)   
        x = MaxPooling3D(pool_size=(1, 3, 3), 
                      strides=(1, 2, 2), padding='same'
                      )(x) 
        x = _inception3d(x)
        x = _inception3d(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), 
                      strides=(2, 2, 2), padding='same'
                      )(x)
        x = _inception3d(x)
        x = _inception3d(x)       
        x = _inception3d(x)
        x = _inception3d(x)
        x = _inception3d(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), 
                      strides=(2, 2, 2), padding='same'
                      )(x)
        x = _inception3d(x)
        x = _inception3d(x)                      
        x = AveragePooling3D(pool_size=(2, 4, 4), strides=(1, 1, 1), padding='valid')(x)
#        x = _conv_bn_relu3D(filters=192, kernel_size=1, 1, 1),
#                                strides=(1, 1, 1),
#                                kernel_regularizer=l2(reg_factor)
 #                               )(x)
        flatten1 = Flatten()(x)
        out = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="sigmoid",
                          kernel_regularizer=l2(reg_factor))(flatten1)
        model = Model(inputs=input, outputs=out)
        return model


    @staticmethod
    def build_i3d(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 18."""
        return I3Dbuilder.build(input_shape, num_outputs, reg_factor=reg_factor)
    
    
    
    
    
    
    
    
    
    