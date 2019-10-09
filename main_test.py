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
Main script for running the trained U-Net model for validation dataset; no trainings performed.

See our method paper:
Liu ZY-C, Chamberlin AJ, Lamore LL, Bauer J, Jones IJ, Van Eck P, Ngo T, Sokolow SH, 
Lambin EF, De Leo GA. Deep learning segmentation of satellite imagery identifies 
aquatic vegetation associated with schistosomiasis snail hosts in Senegal, Africa. 
Remote Sensing. 2019 (in prep.).

Utility script:
unet_model.py: U-Net architecture, model building
data_generator.py: data augmentation

After running the main script, segmentation maps will be created from test images.

For questions, email: zacqoo@gmail.com 
""" 
# ------------------------------------------------------------------------------ 
# import keras libraries
#from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from unet_model import unet
from data_generator import trainGenerator 
from data_generator import saveResult

from PIL import Image
import numpy as np
import glob
import os
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
# number of test images
num_test_img = 4
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
# load weights
model = unet(pretrained_weights, img_width, img_height, learning_rate)
model.load_weights('unet_keras_flow_cera.hdf5')
# ------------------------------------------------------------------------------
# Process test images, overlay prediction masks, change white to red
datagen = ImageDataGenerator(rescale= 1./255) 
img = datagen.flow_from_directory(('data/test_set/'),target_size = (256,256),batch_size=1,shuffle = False, color_mode = "rgb") 
results = model.predict_generator(num_test_img,4,verbose=1)
# save predictions
saveResult('data/test_set/pred_masks/', results)

# load test images
image_list = []
image_filename = []
for filename in sorted(glob.glob('data/test_set/images/*.png')):
    im=Image.open(filename)
    image_list.append(im)
    image_filename.append(filename)
# print list of test image filenames
image_filename

# load predicted mask images
mask_list = []
mask_filename = []
for filename in sorted(glob.glob('data/test_set/pred_masks/*.png')):
    mask=Image.open(filename)
    mask_list.append(mask)
    mask_filename.append(filename)
# print list of predicted mask image filenames
mask_filename

# Process test images, overlay prediction masks, change white to red
for (img, mask, filename) in zip(image_list, mask_list, image_filename):
  # resize mask to match drone image size
  new_width  = img.size[0]
  new_height = img.size[1]
  mask_resize = mask.resize((new_width, new_height), Image.ANTIALIAS)
  # change white color to red, in mask
  mask_resize = mask_resize.convert('RGBA')
  data = np.array(mask_resize)   # "data" is a height x width x 4 numpy array
  red, green, blue, alpha = data.T # Temporarily unpack the bands for readability
  # Replace white with red... (leaves alpha values alone...)
  white_areas = (red == 255) & (blue == 255) & (green == 255)
  data[..., :-1][white_areas.T] = (255, 0, 0) # Transpose back needed
  mask_resize_red = Image.fromarray(data)
  # overlay mask on drone image, add transparency
  background = mask_resize_red.convert("RGBA")
  overlay = img.convert("RGBA")
  new_img = Image.blend(background, overlay, 0.85)
  file_name = filename.split('.')
  fullpath = os.path.join(file_name[0] + '_pred' + '.' + "png")
  print('saving ', filename)
  new_img.save(fullpath)
