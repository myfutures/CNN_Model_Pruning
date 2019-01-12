import numpy as np
import tensorflow as tf
#training params
num_epochs=10                       #times for the use of all training data
batch_size=32                       #number of images for one batch
resize_side_min=256                 #size of image after being randomly resized
resize_side_max=350                 #google给出的数值为512
default_image_size=224              #size of the input image
learning_rate=0.01
learning_rate_decay_factor=0.94     #decay factor for exponential decay
num_epochs_per_decay=2.5
dropout_rate=0.5

#dataset processing params
num_readers=4


#dataset params
#imagenet
imagenet=dict()
imagenet['num_class']=1001                                          #number of the classes
imagenet['label_offset']=1                                          #offset of the label
imagenet['mean']=np.array([104., 117., 124.], dtype=np.float32)     #google [123.68,116.78,103.94]
imagenet['train_set_size']=1271167
imagenet['validation_set_size']=50000
imagenet['label_path']='/home/victorfang/Desktop/imagenet_tfrecord/labels.txt'
imagenet['dataset_path']='/home/victorfang/Desktop/imagenet_tfrecord'



#file path
dataset_dir='/home/victorfang/Desktop/imagenet_tfrecord'
label_dir="/home/victorfang/Desktop/imagenet_tfrecord/labels.txt"

#model saving params
#how often to write summary and checkpoint
log_step = 100
checkpoint_step=10000

# Path for tf.summary.FileWriter and to store model checkpoints
summary_path = "/home/victorfang/Desktop/resnet_model_saved/tensorboard"
checkpoint_path = "/home/victorfang/Desktop/resnet_model_saved/checkpoint_online"
highest_accuracy_path='/home/victorfang/Desktop/resnet_model_saved/accuracy.txt'