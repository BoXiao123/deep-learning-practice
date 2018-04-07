from keras.models import Model,Sequential
from keras.layers import Reshape,Activation,Conv2D,Input,MaxPooling2D,BatchNormalization,Flatten,Dense,Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import cv2
from tools import sapce_to_deth_x2
#from  preprocessing import parse_annotation,Batchgeneration
#from utils import WeightReader, decode_netout, draw_boxes

#assign GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#Hyperparameters
LABELS=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
IMAGE_H,IMAGE_W=416,416
GRID_H,GRID_W=13,13
BOX=5
CLASS=len(LABELS)
CLASS_WEIGHTS=np.ones(CLASS,dtype='float32')
OBJ_THRESHOLD=0.3
NMS_THRESHOD=0.3
ANCHORS=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NO_OBJECT_SCALE=1.0
OBJECT_SCALE=5.0
COORD_SCALE=1.0
CLASS_SCALE=1.0
BATCH_SIZE=16
WARM_UP_BATCHES=0
TRUE_BOX_BUFFER=50
#weights and image path
weights_path='yolo.path'
train_image_folder='path/to/your/tarin/images'
train_annotation_folder='path/to/your/tarin/anotations'
valid_image_folder='path/to/your/valid/images'
valid_annotation_folder='path/to/your/valid/annotations'
#resize input
input_image=Input(shape=(IMAGE_H,IMAGE_W,3))
true_boxes=Input(shape=(1,1,1,TRUE_BOX_BUFFER,4))
#def yolo networks
#layer1
model=Conv2D(32,(3,3),strides=(1,1),padding='same',name='conv_1',use_bias=False)(input_image)
model=BatchNormalization(name='norm_1')(model)
model=LeakyReLU(alpha=0.1)(model)
model=MaxPooling2D(pool_size=(2,2))(model)
#layer2
model=Conv2D(64,(3,3),strides=(1,1),padding='same',name='conv2',use_bias=False)(model)
model=BatchNormalization(name='norm_2')(model)
model=LeakyReLU(alpha=0.1)(model)
model=MaxPooling2D(pool_size=(2,2))(model)
#layer3
model=Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv_3',use_bias=False)(model)
model=BatchNormalization(name='norm_3')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer4
model=Conv2D(64,(1,1),strides=(1,1),padding='same',name='conv_4',use_bias=False)(model)
model=BatchNormalization(name='norm_4')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer5
model=Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv_5',use_bias=False)(model)
model=BatchNormalization(name='norm_5')(model)
model=LeakyReLU(alpha=0.1)(model)
model=MaxPooling2D(pool_size=(2,2))(model)
#layer6
model=Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv_6',use_bias=False)(model)
model=BatchNormalization(name='norm_6')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer7
model=Conv2D(128,(1,1),strides=(1,1),padding='same',name='conv_7',use_bias=False)(model)
model=BatchNormalization(name='norm_7')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer8
model=Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv_8',use_bias=False)(model)
model=BatchNormalization(name='norm_8')(model)
model=LeakyReLU(alpha=0.1)(model)
model=MaxPooling2D(pool_size=(2,2))(model)
#layer9
model=Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_9',use_bias=False)(model)
model=BatchNormalization(name='norm_9')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer10
model=Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv_10',use_bias=False)(model)
model=BatchNormalization(name='norm_10')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer11
model=Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_11',use_bias=False)(model)
model=BatchNormalization(name='norm_11')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer12
model=Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv_12',use_bias=False)(model)
model=BatchNormalization(name='norm_12')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer13
model=Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_13',use_bias=False)(model)
model=BatchNormalization(name='norm_13')(model)
model=LeakyReLU(alpha=0.1)(model)
#skip_connection
skip_connection=model

model=MaxPooling2D(pool_size=(2,2))(model)
#layer14
model=Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_14',use_bias=False)(model)
model=BatchNormalization(name='norm_14')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer15
model=Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv_15',use_bias=False)(model)
model=BatchNormalization(name='norm_15')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer16
model=Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_16',use_bias=False)(model)
model=BatchNormalization(name='norm_16')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer17
model=Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv_17',use_bias=False)(model)
model=BatchNormalization(name='norm_17')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer18
model=Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv18',use_bias=False)(model)
model=BatchNormalization(name='norm_18')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer19
model=Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_19',use_bias=False)(model)
model=BatchNormalization(name='norm_19')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer20
model=Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_20',use_bias=False)(model)
model=BatchNormalization(name='norm_20')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer21
skip_connection=Conv2D(64,(1,1),strides=(1,1),padding='same',name='conv21',use_bias=False)(skip_connection)
skip_connection=BatchNormalization(name='norm_21')(skip_connection)
skip_connection=LeakyReLU(alpha=0.1)(skip_connection)
skip_connection=Lambda(sapce_to_deth_x2)(skip_connection)
model=concatenate([skip_connection,model])
#layer22
model=Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv22',use_bias=False)(model)
model=BatchNormalization(name='norm_22')(model)
model=LeakyReLU(alpha=0.1)(model)
#layer23
model=Conv2D(BOX*(CLASS+5),(1,1),strides=(1,1),padding='same',name='conv_23')(model)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(model)
output = Lambda(lambda args: args[0])([output, true_boxes])
yolo_model=Model([input_image,true_boxes],output)
plot_model(yolo_model, to_file='model1.png',show_shapes=True)
print (yolo_model.summary())
