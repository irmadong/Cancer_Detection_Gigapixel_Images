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
from keras.models import Model, load_model, Sequential
from keras.layers import Dense,Flatten, Dropout, Conv2D, BatchNormalization, Activation, MaxPooling2D, concatenate
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet import ResNet50
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
# import tensorflow.keras.applications.resnet50
# from keras.applications.resnet50 import ResNet50


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

# +


# Find non-tissue areas by looking for all gray regions.
def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return list(zip(indices[0], indices[1]))


# -

def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked

def check_center_region(patch, center_size = 128, threshold = 0.0):
    start = int(299//2)
    mid = int(center_size //2)
    percent = (np.sum(patch[start-mid: start + mid, start-mid: start + mid]))/(center_size*center_size)
    center_region = np.array(patch[start-mid: start + mid, start-mid: start + mid])
#     print(percent)
    return percent > threshold

    

# get windows on
def get_windows(slide_path, tumor_mask_path, levels, stride=150, window_len=299):
    slide_windows1 = []
    slide_windows2 = []
    window_labels = []


    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)
    level1, level2 = levels[0], levels[1]

    window_width = slide.level_dimensions[level1][0] - window_len + 1 
    window_height = slide.level_dimensions[level1][1] - window_len + 1
    downsample_factor1 = slide.level_downsamples[level1]
    downsample_factor2 = slide.level_downsamples[level2]
    
    # sliding window
    for w in range(0, window_width, stride):
        for h in range(0, window_height, stride):
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
            
            tumor_mask_image = tumor_mask_image[:,:,0]

            # calculate the percentage of tissue
            tissue_pixels = find_tissue_pixels(slide_image)
            #print(tissue_pixels)
            percent_tissue = len(tissue_pixels) / float(slide_image.shape[0] * slide_image.shape[0]) 



            if percent_tissue >= 0.5 and np.mean(slide_image) > 0.2:
                slide_windows1.append(slide_image)

                slide_windows2.append(read_slide(slide,
                                                 x = int(center_coord[0] //(2**(level2-level1)) - (window_len//2)*downsample_factor2), 
                                                 y = int(center_coord[1] //(2**(level2-level1)) - (window_len//2)*downsample_factor2), 
                                                 level = level2, 
                                                 width = window_len, 
                                                 height = window_len
                                                ))
                if check_center_region(tumor_mask_image, 128):
                    #if the center has tumor --> 1 
                    window_labels.append(1)
                else:
                    window_labels.append(0)


    return slide_windows1, slide_windows2, window_labels    


# get windows on
def get_test_windows(slide_path, tumor_mask_path, levels, stride=299, window_len=299):
    slide_windows1 = []
    slide_windows2 = []
    window_labels = []
    coord_1 = [] #todo: which coord? to level0 or itself? currently itself 
    coord_2 = []


    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)
    level1, level2 = levels[0], levels[1]

    window_width = slide.level_dimensions[level1][0] - window_len + 1 
    window_height = slide.level_dimensions[level1][1] - window_len + 1
    downsample_factor1 = slide.level_downsamples[level1]
    downsample_factor2 = slide.level_downsamples[level2]
    
    # sliding window
    for w in range(0, window_width, stride):
        for h in range(0, window_height, stride):
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
            
            tumor_mask_image = tumor_mask_image[:,:,0]

            # calculate the percentage of tissue
            tissue_pixels = find_tissue_pixels(slide_image)
            #print(tissue_pixels)
            percent_tissue = len(tissue_pixels) / float(slide_image.shape[0] * slide_image.shape[0]) 

            if percent_tissue >= 0.5 and np.mean(slide_image) > 0.2:
                slide_windows1.append(slide_image)

                slide_windows2.append(read_slide(slide,
                                                 x = int(center_coord[0] //(2**(level2-level1)) - (window_len//2)*downsample_factor2), 
                                                 y = int(center_coord[1] //(2**(level2-level1)) - (window_len//2)*downsample_factor2), 
                                                 level = level2, 
                                                 width = window_len, 
                                                 height = window_len
                                                ))
                coord_1.append((int(w), int(h)))
                coord_2.append((int(center_coord[0] //(2**(level2-level1))//downsample_factor2 -(window_len//2)),
                                    int(center_coord[1] //(2**(level2-level1))//downsample_factor2 -(window_len//2))))
                if check_center_region(tumor_mask_image, 128, 0.0):
     
                    window_labels.append(1)
                else:
                    window_labels.append(0)


    return slide_windows1, slide_windows2, window_labels, coord_1, coord_2   


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


# +
# # def get_windows(slide_paths, tumor_mask_paths, level, stride = 150, window_len=299, threshold = 0.5):
# def generate_raw_patch(slide_paths, level, stride=150, window_len=299):
#     slide_windows = []
#     slide_coord = []
#     slide_level0_coord = []
#     #   contains_cancer = []

#     #   for slide_path, tumor_mask_path in zip(slide_paths, tumor_mask_paths):
#     for slide_path in slide_paths:
#         slide = open_slide(slide_path)
#         #         tumor_mask = open_slide(tumor_mask_path)

#         window_width = slide.level_dimensions[level][0] - window_len + 1
#         window_height = slide.level_dimensions[level][1] - window_len + 1
#         downsample_factor = slide.level_downsamples[level]

#         # sliding window
#         for w in range(0, window_width, stride):
#             for h in range(0, window_height, stride):
#                 level0_coord = (int(w * downsample_factor), int(h * downsample_factor))
#                 curr_coord = (w, h)

#                 curr_slide = read_slide(slide,
#                                         x=curr_coord[0],
#                                         y=curr_coord[1],
#                                         level=level,
#                                         width=window_len,
#                                         height=window_len)
#                 slide_windows.append(curr_slide)
#                 slide_coord.append(curr_coord)
#                 slide_level0_coord.append(level0_coord)

#     return slide_windows, slide_coord, slide_level0_coord

# +
# def make_prediction(model, slide_windows, slide_coord, wid, height, window_len = 299,
#                     stride = 150):
#     #maybe stride not needed?
#     #todo: do we need level0 coord? 
    
#     final_output = np.zeros([wid, height])#todo: which dimension? 
#     #turn into batchsize 
        
#     fill_1 = np.ones([window_len, window_len])
#     fill_0 = np.zeros([window_len, window_len])
    
#     for i in range(0, len(slide_windows), 100):
#         temp = slide_windows[i:i+100]
#         temp_coord = slide_coord[i:i+100]
#         temp = np.array(temp)
    
#         pred = model.predict(temp)
#         pred = np.argmax(pred, axis=1) 
#         for i in range(len(temp_coord)):
#             x_left = slide_coord[i][0] #to do :check is it correct or need to swap? 
#             y_left = slide_coord[i][1]
#             if pred[i] == 1:
#                 final_output[x_left: x_left + window_len, y_left : y_left + window_len] = fill_1

#             #print("current coord is ",slide_coord[i] )
# #         else:
# #              final_output[x_left: x_left + window_len, y_left : y_left + window_len] = fill_0
#     return final_output
# -

def shuffle_data(arr,arr2, label):
    w_tumor = [arr[i] for i in range(len(arr)) if label[i]==1]
    w_tumor2 = [arr2[i] for i in range(len(arr2)) if label[i]==1]
    wo_tumor = [arr[i] for i in range(len(arr)) if label[i]==0]
    wo_tumor2 = [arr2[i] for i in range(len(arr2)) if label[i]==0]
    
    
    shuffle_idx = np.arange(len(wo_tumor))
    np.random.shuffle(shuffle_idx)
    
    wo_tumor = [wo_tumor[shuffle_idx[i]] for i in shuffle_idx[:len(w_tumor)]]
    wo_tumor2 = [wo_tumor2[shuffle_idx[i]] for i in shuffle_idx[:len(w_tumor)]]
    
    return w_tumor, w_tumor2, wo_tumor, wo_tumor2


def cut_data(arr, arr2, label, size = 1250):
    #seperate the data 
    w_tumor = [arr[i] for i in range(len(arr)) if label[i]==1]
    w_tumor2 = [arr2[i] for i in range(len(arr2)) if label[i]==1]
    wo_tumor = [arr[i] for i in range(len(arr)) if label[i]==0]
    wo_tumor2 = [arr2[i] for i in range(len(arr2)) if label[i]==0]
    
    #shuffle index  
    shuffle_idx = np.arange(len(wo_tumor))
    np.random.shuffle(shuffle_idx)
    
    wo_tumor = [wo_tumor[shuffle_idx[i]] for i in shuffle_idx[:size]]
    wo_tumor2 = [wo_tumor2[shuffle_idx[i]] for i in shuffle_idx[:size]]
    
    w_tumor = [w_tumor[shuffle_idx[i]] for i in shuffle_idx[:size]]
    w_tumor2 = [w_tumor2[shuffle_idx[i]] for i in shuffle_idx[:size]]
    
    
    
    
    return w_tumor, w_tumor2, wo_tumor, wo_tumor2


# +
# def data_reduction(train1, train2,  test1, test2, train_limit=2000, test_limit=500):
#   len1 = len(test1)
#   if len1 > train_limit:
#     train1, test1 = shuffle(train1, test1, train_limit)
  
#   len2 = len(test2)
#   if len2 > test_limit:
#     train2, test2 = shuffle(train2, test2, test_limit)
  
#   return train1, train2, test1, test2

# def shuffle(arr, label, limit):
#   arr_pos = np.array([arr[i] for i in len(label) if label[i] == 1][:limit//2])
#   arr_neg = np.array([arr[i] for i in len(label) if label[i] == 0][:limit - limit//2])
#   arr = np.concatenate(arr_pos, arr_neg)
#   label = np.concatenate(np.ones(len(arr_pos)), np.zeros(len(arr_neg)))
#   shuffle_idx = np.arrange(len(arr))
#   np.random.shuffle(shuffle_idx)
#   arr = [arr[shuffle_idx[i]] for i in shuffle_idx]
#   label = [label[shuffle_idx[i] for i in shuffle_idx]]

#   return arr, label
# -

def preprocess(x):

    x = x.astype("float32")
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def create_tf_dataset(X_train1, X_train2, y_train1):
    
    train_ds_1 = tf.data.Dataset.from_tensor_slices(X_train1)
    #image_ds_1 = train_ds_1.map(preprocess)

    train_ds_2 = tf.data.Dataset.from_tensor_slices(X_train2)
    #image_ds_2 = train_ds_2.map(preprocess)
    
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train1, tf.int64))

    image_ds = tf.data.Dataset.zip(((train_ds_1,train_ds_2), label_ds))


    BATCH_SIZE = 16
    final_ds = image_ds.repeat()
    
    #final_ds = final_ds.shuffle(buffer_size=3000)
    final_ds = final_ds.batch(BATCH_SIZE)
    
    final_ds = final_ds.prefetch(1)

    return final_ds


## Imagenet bases using model subclassing
class multi_level_inception(tf.keras.Model):

    def __init__(self):
        super(multi_level_inception, self).__init__(name='multi_level_inception')

        #conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        #conv_base.trainable = True

        self.model1 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        
            
        self.model1.trainable = True
        self.flatten1  = Flatten()

        self.model2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        self.model2.trainable = True
        for layer in self.model2.layers:
            layer._name += str("_2")
        self.flatten2 = Flatten()

        self.concate_layer = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.last = Dense(1, activation='sigmoid')#todo check 1 or 2?

    def call(self, x):
        x1, x2= x[0], x[1]

        x1 = self.model1(x1)
        x1 = self.flatten1 (x1)

        x2 = self.model2(x2)
        x2 = self.flatten2(x2)
    

        x = tf.concat([x1, x2], 1)
        x = self.concate_layer(x)
        x = self.dropout(x)
        x = self.last(x)

        return x


def create_test_tfdataset(X_test1, X_test2):
    #without label 
    test_ds_1 = tf.data.Dataset.from_tensor_slices(X_test1)
    #image_ds_1 = train_ds_1.map(preprocess)

    test_ds_2 = tf.data.Dataset.from_tensor_slices(X_test2)
    #image_ds_2 = train_ds_2.map(preprocess)
    
  

    image_ds = tf.data.Dataset.zip(((test_ds_1,test_ds_2), ))


    BATCH_SIZE = 4
    #final_ds = image_ds.repeat()
    
    final_ds = image_ds.batch(BATCH_SIZE)
    
    final_ds = final_ds.prefetch(1)

    return final_ds


def make_prediction(model, slide_windows1, slide_windows2, window_labels, coord_1, heatmap):
    #get label for the patch 
    preds = []
    for i in range(len(slide_windows1)):
        testds = create_test_tfdataset(slide_windows1, slide_windows2)
        print()
        #pred = model.predict([preprocess(np.array(slide_windows1[i])), preprocess(np.array(slide_windows2[i]))])
        pred = model.predict(testds)
        #check prob or class maybe chance last dense layer into 1 
        if i < 10:
            print('before', pred)
        pred = pred > 0.5
        if i<10:
            print("after", pred)
        preds.append(pred)
        w,h = coord_1[i]
        if pred == 1:
            midh = h + 299//2
            midw = w+299//2
            heatmap[int(midh - 64):(midh+64), (midw - 64): (midw + 64)] = 1
        
            
    
    return preds, heatmap

# +
# def make_prediction_batch(model, test_ds):
#     #get label for the patch 
#     preds = []
#     for i in range(len(slide_windows1)):
#         testds = create_test_tfdataset(slide_windows1, slide_windows2)
#         print()
#         #pred = model.predict([preprocess(np.array(slide_windows1[i])), preprocess(np.array(slide_windows2[i]))])
#         pred = model.predict(testds)
#         #check prob or class maybe chance last dense layer into 1 
#         if i < 10:
#             print('before', pred)
#         pred = pred > 0.5
#         if i<10:
#             print("after", pred)
#         preds.append(pred)
#         w,h = coord_1[i]
#         if pred == 1:
#             midh = h + 299//2
#             midw = w+299//2
#             heatmap[int(midh - 64):(midh+64), (midw - 64): (midw + 64)] = 1
        
            
    
#     return preds, heatmap
# -

def generate_heat_map(model, pred, coord, heatmap, threshold = 0.5):
    class_pred = []
    for i in range(len(coord)):
        
        w,h = coord[i]
#         midh = h + 299//2
#         midw = w+299//2
        if pred[i] > threshold: 
            #heatmap[int(midh - 64):(midh+64), (midw - 64): (midw + 64)] = 1 #check 
            heatmap[int(h):int(h+299), int(w):int(w+299)] = 1 
            class_pred.append(1)
        else:
            #heatmap[int(midh - 64):(midh+64), (midw - 64): (midw + 64)] = 0 #check
            class_pred.append(0)
    return class_pred


def evaluate(label, class_pred, label_name):
    fpr, tpr, thresholds = metrics.roc_curve(label, class_pred)
    #print(metrics.auc(fpr, tpr))
    acc = metrics.accuracy_score(label, class_pred)
    print("accuracy:", acc)
    print('AUC for ROC Curve :%s'%(metrics.auc(fpr, tpr)))
    plt.title('ROC curve ' + label_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr)
    plt.show()
    print(metrics.classification_report(label, class_pred))

    precision, recall, thresholds = metrics.precision_recall_curve(label, class_pred)
    plt.title('Precision Recall curve ' + label_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.show()
    print("AUC for Precision Recall Curve :%s "%(metrics.auc(recall, precision)) )



def draw_overlap(mask_image, heatmap, slide_image):
    plt.figure(figsize=(10,10), dpi=100)
    plt.grid(False)
    plt.imshow(heatmap)
    plt.show()
    plt.figure(figsize=(10,10), dpi=100)
    plt.imshow(mask_image)
    plt.imshow(heatmap, cmap='binary', alpha = 0.5) 
    plt.show()
    plt.figure(figsize=(10,10), dpi=100)
    plt.imshow(slide_image)
    plt.imshow(heatmap, cmap = "Reds", alpha = 0.5) 
    plt.show()


def create_model():

    model1 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299, 3))
    in1 = model1.input
    out1 = model1.output

    for layer in model1.layers:
        layer.trainable = True

    model2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299, 3))

    for layer in model2.layers:
        layer._name +=layer.name + str("_2")

    in2 = model2.input
    out2 = model2.output

    for layer in model2.layers:
        layer.trainable = True

    concate= concatenate([out1, out2])

    flat = Flatten()(concate)
    dropout1 = Dropout(.5)(flat)
    dense1 = Dense(128, activation='relu')(dropout1)
    final_layer = Dense(1, activation='sigmoid')(dense1)
    model = Model([in1, in2], final_layer)
   
    return model 
