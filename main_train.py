# MIT License

# Copyright (c) 2019 De Leo Lab- Hopkins Marine Station, Stanford University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""
Main script for training the U-Net model.

See our method paper:
Liu ZY-C, Chamberlin AJ, Lamore LL, Bauer J, Jones IJ, Van Eck P, Ngo T, Sokolow SH, 
Lambin EF, De Leo GA. Deep learning segmentation of satellite imagery identifies 
aquatic vegetation associated with schistosomiasis snail hosts in Senegal, Africa. 
Remote Sensing. 2019 (in prep.).

Utility script:
unet_model.py: U-Net architecture, model building
data_generator.py: data augmentation

After running the main script, the accuracy will be printed. Model weights will be saved.

For questions, email: zacqoo@gmail.com 
"""
# ------------------------------------------------------------------------------ 
# import keras libraries
from keras.callbacks import ModelCheckpoint
#from keras.preprocessing.image import ImageDataGenerator
from unet_model import unet
from data_generator import trainGenerator 
# ------------------------------------------------------------------------------ 
# Set up file path
path = 'data/training_set'
path_image = 'images'
path_mask = 'floating'
# define parameters
in_s = 256 # input image size 
img_width, img_height = in_s, in_s  
# number of epochs to train top model  
epochs = 20 
# number of steps per epoch
steps_per_epoch = 150
# learning rate
learning_rate = 1e-4
# pre trained weight, default = None
pretrained_weights = None
# ------------------------------------------------------------------------------  
#data augmentation
data_gen_args = dict(rescale= 1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,path,path_image,path_mask,data_gen_args, save_to_dir = None)
# ------------------------------------------------------------------------------  
# training
model = unet(pretrained_weights, img_width, img_height, learning_rate)
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[model_checkpoint])

# save weights to drive
model.save_weights('unet_1.hdf5')

