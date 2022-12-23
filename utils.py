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
    '''
    read the region from the slide 
    slide: the slide 
    x: the top left corner x
    y: the top left corner y
    level: the zoom level 
    width: the width of the region
    height: the height of the region 
    as_float: the pixel as floar or not 
    
    
    '''
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im


# +

def find_tissue_pixels(image, intensity=0.8):
    '''
    Find non-tissue areas by looking for all gray regions.
    image: the slide image 
    intensity: the instensity of the image 
    
    
    '''
    
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return list(zip(indices[0], indices[1]))


# -

def apply_mask(im, mask, color=(255,0,0)):
    '''
    apply mask on the image 
    im: the image 
    mask: the mask
    color: the color 
    
    '''
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked

def check_center_region(patch, center_size = 128, threshold = 0.0):
    '''
    check the center region whether has tumor or not
    patch: the patch of the slide 
    center_size: the dimension of the center 
    threshold: the threshold of the tumor 
    
    '''
    start = int(299//2)
    mid = int(center_size //2)
    percent = (np.sum(patch[start-mid: start + mid, start-mid: start + mid]))/(center_size*center_size)
    center_region = np.array(patch[start-mid: start + mid, start-mid: start + mid])
    return percent > threshold


# +

def get_windows(slide_path, tumor_mask_path, levels, stride=150, window_len=299):
    '''
    get sliding windows on the lide 
    
    slide_path: the path of the slide 
    tumor_mask_path: the path of the tumor mask 
    levels: the zoom level 
    stride: the stride
    window_len: the window size 
    
    
    '''
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


# -

# get windows on
def get_test_windows(slide_path, tumor_mask_path, levels, stride=299, window_len=299):
    '''
    similar to the get windows but this time also records the coordinate of the patches 
    
    slide_path: the path of the slide 
    tumor_mask_path: the path of the tumor mask 
    levels: the zoom level 
    stride: the stride
    window_len: the window size 
    
    
    
    '''
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
    """
    display the patches randomly
    images: the image 
    row: the row number
    col: the column number
    x: the top left x 
    y: the top left y 
    
    
    """
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


def shuffle_data(arr,arr2, label):
    """
    shuffle the patches 
    arr: the first level list of patches 
    arr2: the second level list of patches 
    label: the lable of each patch
    
    
    """
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
    '''
    Cut off the data from training and test set 
    
    arr: the first level of sliding windows 
    arr2: the second level of sliding windows 
    label: the label of the patches 
    size: the target size of reduced dataset 
    
    '''
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


def preprocess(x):
    """
    the preprocess step for inception V3 model 
    x: the np array 
    """

    x = x.astype("float32")
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def create_tf_dataset(X_train1, X_train2, y_train1):
    """
    create the tensorflow batch dataset 
    
    X_train1: the first selected level of X training dataset
    X_train2: the second selected level of X training dataset 
    y_train1: the label of the training dataset 
    
    """
    
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


def create_test_tfdataset(X_test1, X_test2):
    '''
    create the tensorflow batch dataset without label
    X_test1: the first selected level test dataset
    X_test2: the second selected level test dataset 
    
    '''
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


def generate_heat_map(model, pred, coord, heatmap, threshold = 0.5):
    '''
    generate the heapmap 
    
    model: the trained model
    pred: the prediction
    coord: the coordination
    threshold: the prediction probability
    '''
    class_pred = []
    for i in range(len(coord)):
        
        w,h = coord[i]

        if pred[i] > threshold: 
            
            heatmap[int(h):int(h+299), int(w):int(w+299)] = 1 
            class_pred.append(1)
        else:
            class_pred.append(0)
    return class_pred


def evaluate(label, class_pred, label_name):
    """
    evaluate the model performance as the binary classification problem 
    label: the ground truth label
    class_pred: the predicted class
    label_name: the name of the slide for the title 
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, class_pred)
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
    """
    draw the overlap between the ground truth heatmap and the predicted heatmap 
    & the overlap between the slide image and the predicted heatmap 
    mask_image: the ground truth heatmap 
    heatmap: the predicted heatmap
    slide_image: the slide image 
    """
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
    """
    create the multi parallel inception v3 model with concatenated input of two zoom levels
    """

    model1 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299, 3))
    in1 = model1.input
    out1 = model1.output

    for layer in model1.layers:
        layer.trainable = True

    model2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299, 3))
    for layer in model2.layers:
        layer.trainable = True
    for layer in model2.layers:
        layer._name +=layer.name + str("_2")

    in2 = model2.input
    out2 = model2.output


    concate= concatenate([out1, out2])
    flat = Flatten()(concate)
    dropout1 = Dropout(.5)(flat)
    dense1 = Dense(128, activation='relu')(dropout1)
    final_layer = Dense(1, activation='sigmoid')(dense1)
    model = Model([in1, in2], final_layer)
   
    return model 
