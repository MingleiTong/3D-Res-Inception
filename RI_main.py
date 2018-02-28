# -*- coding: utf-8 -*-
import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse 
from RI_models import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

# trained weight
_rgb_weights = "res_inception/weights/rgb_weights.h5"
_flow_weights = "res_inception/weights/flow_weights.h5"
_full_weights = "res_inception/weights/weights_finetune.h5"
# save weights
_weights_h = "res_inception/weights/rgb_weights.h5"
_weights_l = "res_inception/weights/flow_weights.h5"
_weights = "res_inception/weights/weights.h5"

_TFBooard = 'res_inception/events/'



parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    type=str,
                    default='finetune',
                    help='rgb,flow,finetune')
parser.add_argument('--test',
                    type=str,
                    default='finetune',
                    help='rgb,flow,finetune')
parser.add_argument('--modelname', type=str,
                    default='res_inception/weights/finetune_models.h5', help='final model save name')
parser.add_argument('--epochs',type=int,
                    default=100,help='number of epochs')
parser.add_argument('--BATCH_SIZE',type=int,
                    default=10,help='number of batch')
args = parser.parse_args()

if not os.path.exists('res_inception/weights/'):
    os.makedirs('res_inception/weights/')

if not os.path.exists(_TFBooard):
    os.mkdir(_TFBooard)

def train_rgb(model):


    Xh_train = np.load('./data/rgb_train.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('./data/rgb_train_label.npy'))

    Xh_val = np.load('./data/rgb_test.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('./data/rgb_test_label.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights_h, verbose=1, save_best_only=True)

    model.fit(Xh_train, Y_train, batch_size=args.BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=(Xh_val, Y_val))
    scores = model.evaluate(
        Xh_val, Y_val, batch_size=10)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)

def train_flow(model):



    Xl_train = np.load('./data/flow_train.npy')
    # Xh_train = np.load('../file/train_Xh.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('./data/flow_train_label.npy'))

    Xl_val = np.load('./data/flow_test.npy')
    # Xh_val = np.load('../file/val_Xh.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('./data/flow_test_label.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights_l, verbose=1, save_best_only=True)
    
    # if you need TTensorboard while training phase just uncomment 
    # TFBoard = TensorBoard(
    #     log_dir=_TFBooard, write_graph=True, write_images=False)
    # model.fit([Xl_train], Y_train, batch_size=BATCH_SIZE, class_weight=cls_weights, epochs=args.epochs,
    #           callbacks=[model_ckt, TFBoard], validation_data=([Xl_val], Y_val))
    
    model.fit([Xl_train], Y_train, batch_size=args.BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=([Xl_val], Y_val))
    scores = model.evaluate([Xl_val], Y_val,batch_size=10)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)

def train_full(model):

    Xl_train = np.load('./data/flow_train.npy')
    Xh_train = np.load('./data/rgb_train.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('./data/rgb_train_label.npy'))

    Xl_val = np.load('./data/flow_test.npy')
    Xh_val = np.load('./data/rgb_test.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('./data/rgb_test_label.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights, verbose=1, save_best_only=True)
   
    model.fit([Xh_train, Xl_train], Y_train, batch_size=args.BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_split=0.2)
    scores = model.evaluate([Xh_val,Xl_val], Y_val,batch_size=10)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)


def test(network):
    if network =='flow':
        model = flow_branch()
        model.load_weights(_weights_l)
        Xl = np.load('./data/flow_test.npy')
        Y = K.utils.np_utils.to_categorical(np.load('./data/flow_test_label.npy'))
        scores = model.evaluate(
        Xl, Y, batch_size=10)
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])
    if network == 'rgb':
        model = rgb_branch()
        model.load_weights(_weights_h)
        Xh = np.load('./data/rgb_test.npy')
        Y = K.utils.np_utils.to_categorical(np.load('./data/rgb_test_label.npy'))
        #pred = model.predict([Xh])
        scores = model.evaluate(
        Xh, Y, batch_size=10)
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])
    if network == 'finetune':
        model =finetune_Net()
        model.load_weights(_weights)
        Xl = np.load('./data/flow_test.npy')
        Xh = np.load('./data/rgb_test.npy')
        Y = K.utils.np_utils.to_categorical(np.load('./data/rgb_test_label.npy'))
        scores = model.evaluate([Xh, Xl], Y, batch_size=10)
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])

def visual_model(model,imgname):
    from keras.utils import plot_model
    plot_model(model, to_file=imgname, show_shapes=True)

def main():
    if args.train == 'flow':
        model = flow_branch()
        imgname = 'flow_model.png'
        visual_model(model, imgname)
        train_flow(model)
    if args.train == 'rgb':
        model = rgb_branch()
        imgname = 'rgb_model.png'
        visual_model(model, imgname)
        train_rgb(model)
    if args.train == 'finetune':
        model = finetune_Net(rgb_weight=None,
                             flow_weight=None, 
                             trainable=False)
        imgname = 'model.png'
        visual_model(model, imgname)
        train_full(model)
    #test phase
    if args.test == 'flow':
        start = time.time()
        test('flow')
        print('elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'rgb':
        start = time.time()
        test('rgb')
        print('elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'finetune':
        start = time.time()
        test('finetune')
        print('elapsed time:{:.2f}s'.format(time.time() - start))

if __name__ == '__main__':
    main()
    
