# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
import os
from PIL import Image
from skimage.color import rgb2gray
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Read a region from the slide
# Return a numpy RBG array
def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im


# Find non-tissue areas by looking for all gray regions.
def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return list(zip(indices[0], indices[1]))


def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked

def check_center_region(patch, center):
    center_w = center[0]
    center_h = center[1]
    return np.sum(patch[center_h-64: center_h+64, center_w-64, center_w + 64])>=1
    
    

# get windows on
def get_windows(slide_path, tumor_mask_path, levels, stride=150, window_len=299, threshold=0.5):
    slide_windows1 = []
    slide_windows2 = []
    window_labels = []

    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)
    level1, level2 = levels[0], levels[1]

    window_width = slide.level_dimensions[level1][0] - window_len + 1 
    window_height = slide.level_dimensions[level1][1] - window_len + 1
    downsample_factor1 = slide.level_downsamples[level1]
    downsample_factor2 = slilde.level_downsamples[level2]
    
    # sliding window
    for w in range(0, window_height, stride):
        for h in range(0, window_width, stride):
        curr_coord = (int(w*downsample_factor1), int(h*downsample_factor1))
        center_coord = (int((w+window_len//2)*downsample_factor1), int((h+window_len//2)*downsample_factor1))

        slide_image = read_slide(slide,
                                 x=curr_coord[0],
                                 y=curr_coord[1],
                                 level=level1,
                                 width=window_len,
                                 height=window_len)

        tumor_mask_image = read_slide(tumor_mask,
                                      x=curr_coord[0],
                                      y=curr_coord[1],
                                      level=level1,
                                      width=window_len,
                                      height=window_len)

        # calculate the percentage of tissue
        tissue_pixels = find_tissue_pixels(slide_image)
        #print(tissue_pixels)
        percent_tissue = len(tissue_pixels) / float(curr_slide.shape[0] * curr_slide.shape[0] ) 
        #print("curr_slide shape", curr_slide.shape[0])) 
        #print("curr_slide shape", curr_slide.shape[0])

        if percent_tissue >= 0.5 and np.mean(curr_slide) > 0.2 and check_center_region(slide_image, center_coord):
            slide_windows1.append(slide_image)
            
            slide_windows2.append(read_slide(slide,
                                             x = center_coord[0] //(2**(level2-level1)) - (window_size//2)*downsample_factor2, 
                                             y = center_coord[1] //(2**(level2-level1)) - (window_size//2)*downsample_factor2, 
                                             level = level2, 
                                             width = window_size, 
                                             height = window_size
                                            ))
            
            if np.sum(tumor_mask_image[:,:,0]) / (window_len * window_len) >= threshold:
                window_labels.append(1)
            else:
                window_labels.append(0)

    return slide_windows1, slide_windows2, window_labels    


def display_windows(images, row, col, x, y):
  fig = plt.figure(figsize=(x,y))
  shuffle = np.arange(len(images))
  np.random.shuffle(shuffle)
  for i in range(row):
    for j in range(col):
      idx = i*row+j
      ax = fig.add_subplot(col, row, idx+1)
      plt.imshow(images[shuffle[idx]])
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
    

# def get_windows(slide_paths, tumor_mask_paths, level, stride = 150, window_len=299, threshold = 0.5):
def generate_raw_patch(slide_paths, level, stride=150, window_len=299):
    slide_windows = []
    slide_coord = []
    slide_level0_coord = []
    #   contains_cancer = []

    #   for slide_path, tumor_mask_path in zip(slide_paths, tumor_mask_paths):
    for slide_path in slide_paths:
        slide = open_slide(slide_path)
        #         tumor_mask = open_slide(tumor_mask_path)

        window_width = slide.level_dimensions[level][0] - window_len + 1
        window_height = slide.level_dimensions[level][1] - window_len + 1
        downsample_factor = slide.level_downsamples[level]

        # sliding window
        for w in range(0, window_width, stride):
            for h in range(0, window_height, stride):
                level0_coord = (int(w * downsample_factor), int(h * downsample_factor))
                curr_coord = (w, h)

                curr_slide = read_slide(slide,
                                        x=curr_coord[0],
                                        y=curr_coord[1],
                                        level=level,
                                        width=window_len,
                                        height=window_len)
                slide_windows.append(curr_slide)
                slide_coord.append(curr_coord)
                slide_level0_coord.append(level0_coord)

    return slide_windows, slide_coord, slide_level0_coord


def make_prediction(model, slide_windows, slide_coord, wid, height, window_len = 299,
                    stride = 150):
    #maybe stride not needed?
    #todo: do we need level0 coord? 
    
    final_output = np.zeros([wid, height])#todo: which dimension? 
    #turn into batchsize 
        
    fill_1 = np.ones([window_len, window_len])
    fill_0 = np.zeros([window_len, window_len])
    
    for i in range(0, len(slide_windows), 100):
        temp = slide_windows[i:i+100]
        temp_coord = slide_coord[i:i+100]
        temp = np.array(temp)
    
        pred = model.predict(temp)
        pred = np.argmax(pred, axis=1) 
        for i in range(len(temp_coord)):
            x_left = slide_coord[i][0] #to do :check is it correct or need to swap? 
            y_left = slide_coord[i][1]
            if pred[i] == 1:
                final_output[x_left: x_left + window_len, y_left : y_left + window_len] = fill_1

            #print("current coord is ",slide_coord[i] )
#         else:
#              final_output[x_left: x_left + window_len, y_left : y_left + window_len] = fill_0
    return final_output

