# 3D-Res-Inception
Introduction

The scripts are uesd for Keras(Tensorflow backend).
Inspired by I3D and shortcut, 3D Res-Inception is a 3D convolutional neural network for crowd behavior recognition in video.

Dataset 

We used CUHK crowd dataset which consists of 474 video clips from 215 crowd scene.It was divided into eight catogories based on crowd behavior.  
The dataset can be downloaded at 
http://www.ee.cuhk.edu.hk/~jshao/projects/CUHKcrowd_files/cuhk_crowd_dataset.htm

Content

res_inception.py  
The script is to build a 3D Res-Inception network.

RI_models.py  
The script is to build a two branch 3D Res-Inception network.One branch input rgb sequences, the other input optical-flow sequences.

RI_main.py  
The main script set the parameters(batch_size, number of epochs...), path of data and saved model.
 
