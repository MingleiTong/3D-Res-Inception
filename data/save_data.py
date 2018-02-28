#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import os
from input_data import read_clip_and_label
import numpy as np

path = './data/'
flow_train_images, flow_train_labels, _, _, _ = read_clip_and_label(
    filename='./label/flow_train.list',
    batch_size=380,
    num_frames_per_clip=16,
    crop_size=128,
    shuffle=True)
flow_test_images, flow_test_labels, _, _, _ = read_clip_and_label(
    filename='./label/flow_test.list',
    batch_size=94,
    num_frames_per_clip=16,
    crop_size=128,
    shuffle=True)
 
np.save(os.path.join(path, 'flow_train.npy'), flow_train_images)
np.save(os.path.join(path, 'flow_test.npy'), flow_test_images)
np.save(os.path.join(path, 'flow_train_label.npy'), flow_train_labels)
np.save(os.path.join(path, 'flow_test_label.npy'), flow_test_labels)


train_images, train_labels, _, _, _ = read_clip_and_label(
    filename='./label/rgb_train.list',
    batch_size=380,
    num_frames_per_clip=16,
    crop_size=128,
    shuffle=True)
test_images, test_labels, _, _, _ = read_clip_and_label(
    filename='./label/rgb_test.list',
    batch_size=94,
    num_frames_per_clip=16,
    crop_size=128,
    shuffle=True)
 
np.save(os.path.join(path, 'rgb_train.npy'), train_images)
np.save(os.path.join(path, 'rgb_test.npy'), test_images)
np.save(os.path.join(path, 'rgb_train_label.npy'), train_labels)
np.save(os.path.join(path, 'rgb_test_label.npy'), test_labels)



