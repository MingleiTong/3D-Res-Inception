# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import keras as K
import keras.layers as L
import numpy as np
import os 
import time
import h5py
import argparse
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from res_inception import I3Dbuilder
from keras.optimizers import SGD, Adam, RMSprop


def rgb_branch():    
    model = I3Dbuilder.build_i3d((16, 128, 128, 3), 8)
    optimizer = Adam(lr=1e-4, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model
def flow_branch():    
    model = I3Dbuilder.build_i3d((16, 128, 128, 3), 8)
    optimizer = Adam(lr=1e-4, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model
    

def finetune_Net(rgb_weight=None, flow_weight=None, trainable=False):
    """
    fine tune from the trained weights without update 
    in order to 
    """
    model_h = rgb_branch()
    model_l = flow_branch()
    if not rgb_weight is None: 
        model_h.load_weights(rgb_weight)
    if not flow_weight is None:
        model_l.load_weights(flow_weight)
    for i in xrange(1):
        model_h.layers.pop()
        model_l.layers.pop()
    if not trainable:
        model_h.trainable = False
        model_l.trainable = False
    rgb_in= model_h.input
    rgb_out = model_h.layers[-1].output
    flow_in = model_l.input
    flow_out = model_l.layers[-1].output

    merge = L.concatenate([rgb_out, flow_out], axis=-1)
    merge = L.BatchNormalization(axis=-1)(merge)
    merge = L.Dropout(0.25)(merge)
    merge = L.Dense(128)(merge)
    merge = L.advanced_activations.LeakyReLU(alpha=0.2)(merge)
    logits = L.Dense(8, activation='softmax')(merge)
    model = K.models.Model([rgb_in, flow_in], logits)
    if not rgb_weight is None or flow_weight is None:
        optm = K.optimizers.SGD(lr=0.005,momentum=1e-6)
        # optm=K.optimizers.Adam(lr=0.0005)
    else:
        optm=K.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optm,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model    
