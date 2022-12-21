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
# As mentioned in class, we can improve efficiency by ignoring non-tissue areas 
# of the slide. We'll find these by looking for all gray regions.
def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return list(zip(indices[0], indices[1]))
def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked


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


def get_patches(slide_path, tumor_mask_path, 
                    lvl1, lvl2, window_size, 
                    tumor_sampled_limit, healthy_sampled_limit):
    """
    Return the patchs for two levels and their labels

    slide_path: Path to the slide.
    tumor_mask_path: Path to the tumor mask slide
    lvl1 and lvl2: Levels of the slide
    window_size: Sliding Window size
    tumor_sampled_limit: Number of Tumorous patches to return per slide per level
    healthy_sampled_limit: Number of Healthy patches to return per slide per level
    """
        
  patch_images_1 = []
  patch_images_2 = []
  
  patch_labels = []
  
  num_cancer = 0
  num_health = 0
 
  reference_lvl = 4

  slide = open_slide(slide_path)
  print ("Read WSI from %s with width: %d, height: %d" % (slide_path, slide.level_dimensions[0][0], slide.level_dimensions[0][1]))

  tumor_mask = open_slide(tumor_mask_path)
  print ("Read tumor mask from %s" % (tumor_mask_path))
  
  slide_image = read_slide(slide, 
                         x=0, 
                         y=0, 
                         level=reference_lvl, 
                         width=slide.level_dimensions[reference_lvl][0], 
                         height=slide.level_dimensions[reference_lvl][1])
  
  tumor_mask_image = read_slide(tumor_mask, 
                         x=0, 
                         y=0, 
                         level=reference_lvl, 
                         width=slide.level_dimensions[reference_lvl][0], 
                         height=slide.level_dimensions[reference_lvl][1])
  
  tumor_mask_image = tumor_mask_image[:,:,0]
  
  #Get a list of tumor pixels at reference level
  list_tumor_mask_pixels = np.nonzero(tumor_mask_image)
  
  #Construct a healthy tissue mask by subtracting tumor mask from tissue mask
  tissue_pixels = find_tissue_pixels(slide_image)
  tissue_regions = apply_mask(slide_image, tissue_pixels)

  healthy_mask_image = tissue_regions[:,:,0] - tumor_mask_image
  healthy_mask_image = healthy_mask_image > 0
  healthy_mask_image = healthy_mask_image.astype('int')

  #Get a list of healthy tissue pixels at reference level
  list_healthy_mask_pixels = np.nonzero(healthy_mask_image)
  
  #Collect tumor patches
  tumor_pixels = random.sample(list(zip(list_tumor_mask_pixels[1], list_tumor_mask_pixels[0])), tumor_sampled_limit * 10)
  
  count = 0
  for pixel in tumor_pixels:
    if count >= tumor_sampled_limit:
      break
      
    (x_ref, y_ref) = pixel

    #Convert reference_lvl coordinates to level 0 coordinates
    x0 = x_ref*(2**reference_lvl)
    y0 = y_ref*(2**reference_lvl)
    
    downsample_factor = 2**lvl1
    
    patch = read_slide(slide,
                       x = x0-(window_size//2)*downsample_factor,
                       y = y0-(window_size//2)*downsample_factor, 
                       level = lvl1,
                       width = window_size,
                       height = window_size)
    
    tumor_mask_patch = read_slide(tumor_mask,
                       x = x0-(window_size//2)*downsample_factor,
                       y = y0-(window_size//2)*downsample_factor, 
                       level = lvl1,
                       width = window_size,
                       height = window_size)
    
    tumor_mask_patch = tumor_mask_patch[:,:,0]
    
    tissue_pixels = find_tissue_pixels(patch)
    tissue_pixels = list(tissue_pixels)
    percent_tissue = len(tissue_pixels) / float(patch.shape[0] * patch.shape[0]) * 100

    if percent_tissue > 50 and check_patch_centre(tumor_mask_patch, 128):
        patch_images_1.append(patch)
        patch_images_2.append(read_slide(slide, x = x0-(window_size//2)*downsample_factor, y = y0-(window_size//2)*downsample_factor, level = lvl2, width = window_size, height = window_size))

        patch_labels.append(1)
        count += 1
        

        
  #Collect healthy patches
  healthy_pixels = random.sample(list(zip(list_healthy_mask_pixels[1], list_healthy_mask_pixels[0])), healthy_sampled_limit * 20)
  
  count = 0
  for pixel in healthy_pixels:
    if count >= healthy_sampled_limit:
      break
      
    (x_ref, y_ref) = pixel

    #Convert reference_lvl coordinates to level 0 coordinates
    x0 = x_ref*(2**reference_lvl)
    y0 = y_ref*(2**reference_lvl)
    
    downsample_factor = 2**lvl1
    
    patch = read_slide(slide,
                       x = x0-(window_size//2)*downsample_factor,
                       y = y0-(window_size//2)*downsample_factor, 
                       level = lvl1,
                       width = window_size,
                       height = window_size)
    
    tumor_mask_patch = read_slide(tumor_mask,
                       x = x0-(window_size//2)*downsample_factor,
                       y = y0-(window_size//2)*downsample_factor, 
                       level = lvl1,
                       width = window_size,
                       height = window_size)
    
    tumor_mask_patch = tumor_mask_patch[:,:,0]
    
    tissue_pixels = find_tissue_pixels(patch)
    tissue_pixels = list(tissue_pixels)
    percent_tissue = len(tissue_pixels) / float(patch.shape[0] * patch.shape[0]) * 100

    if percent_tissue > 50 and (not check_patch_centre(tumor_mask_patch, 128)):
        patch_images_1.append(patch)
        patch_images_2.append(read_slide(slide, x = x0-(window_size//2)*downsample_factor, y = y0-(window_size//2)*downsample_factor, level = lvl2, width = window_size, height = window_size))
        patch_labels.append(0)
        count += 1

  return patch_images_1, patch_images_2, patch_labels


# get all non-tissue windows test version 
def get_windows(slide_paths, tumor_mask_paths, level, level2,  stride = 150, window_len=299):
  slide_windows = []
  contains_cancer = []

  for slide_path, tumor_mask_path in zip(slide_paths, tumor_mask_paths):
    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)

    window_width = slide.level_dimensions[level][0] - window_len + 1 
    window_height = slide.level_dimensions[level][1] - window_len + 1
    downsample_factor = slide.level_downsamples[level]
    
    # sliding window
    for w in range(0, window_width, stride):
      for h in range(0, window_height, stride):
        curr_coord = (int(w*downsample_factor), int(h*downsample_factor))

        curr_slide = read_slide(slide,
                                 x=curr_coord[0],
                                 y=curr_coord[1],
                                 level=level,
                                 width=window_len,
                                 height=window_len)
        
        curr_tumor_mask = read_slide(tumor_mask,
                                      x=curr_coord[0],
                                      y=curr_coord[1],
                                      level=level,
                                      width=window_len,
                                      height=window_len)
        
        # calculate the percentage of tissue
        tissue_pixels = find_tissue_pixels(curr_slide)
        #print(tissue_pixels)
        percent_tissue = len(tissue_pixels) / float(curr_slide.shape[0] * curr_slide.shape[0] ) 
        #print("curr_slide shape", curr_slide.shape[0])) 
        #print("curr_slide shape", curr_slide.shape[0])

        if percent_tissue >= 0.2 and np.mean(curr_slide) > 0.2:
          slide_windows.append(curr_slide)
          if np.sum(curr_tumor_mask[:,:,0]) / (window_len * window_len)  >= threshold:
            contains_cancer.append(1)
          else:
            contains_cancer.append(0)

  return slide_windows, contains_cancer


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



