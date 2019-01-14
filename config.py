import numpy as np
import tensorflow as tf
#training params
num_epochs=10                       #times for the use of all training data
batch_size=32                       #number of images for one batch
resize_side_min=256                 #size of image after being randomly resized
resize_side_max=350                 #google给出的数值为512
learning_rate=0.003
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


#model saving params
#how often to write summary and checkpoint
log_step = 1
checkpoint_step=10000

# Path for tf.summary.FileWriter and to store model checkpoints
root_path='/home/victorfang/Desktop/'
summary_path = '_model_saved/tensorboard'
checkpoint_path = "_model_saved/checkpoint_online"
highest_accuracy_path='_model_saved/accuracy.txt'