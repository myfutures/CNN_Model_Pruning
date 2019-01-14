import tensorflow as tf
import AlexNet
import numpy as np
from datetime import datetime
from generate_data_from_tfrecord import read_tfrecord
import config as conf
import os
from tensorflow.contrib.model_pruning.python import pruning
from ResNet import inference

def train_alexnet(dataset_name='imagenet',
                  prune=False,
                  prune_params='',
                  learning_rate=conf.learning_rate,
                  num_epochs=conf.num_epochs,
                  batch_size=conf.batch_size,
                  learning_rate_decay_factor=conf.learning_rate_decay_factor,
                  num_epochs_per_decay=conf.num_epochs_per_decay,
                  dropout_rate=conf.dropout_rate,
                  log_step=conf.log_step,
                  checkpoint_step=conf.checkpoint_step,
                  summary_path=conf.root_path+'alexnet'+conf.summary_path,
                  checkpoint_path=conf.root_path+'alexnet'+conf.checkpoint_path,
                  highest_accuracy_path=conf.root_path+'alexnet'+conf.highest_accuracy_path,
                  default_image_size=227,                                                                       #224 in the paper
                  ):
    """prune_params: Comma separated list of pruning-related hyperparameters
       ex:'begin_pruning_step=10000,end_pruning_step=100000,target_sparsity=0.9,sparsity_function_begin_step=10000,sparsity_function_end_step=100000'
    """
    if dataset_name is 'imagenet':
        num_class=conf.imagenet['num_class']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_size=conf.imagenet['validation_set_size']
        label_offset=conf.imagenet['label_offset']
        label_path=conf.imagenet['label_path']
        dataset_path=conf.imagenet['dataset_path']        

        x = tf.placeholder(tf.float32, [batch_size, default_image_size, default_image_size, 3])
        y = tf.placeholder(tf.float32, [batch_size, num_class-label_offset])
        keep_prob=tf.placeholder(tf.float32)                                        #placeholder for dropout rate
        # prepare to train the model
        model = AlexNet.AlexNet(x, keep_prob, num_class-label_offset, [],prune=prune)
        # Link variable to model output
        score = model.fc8

        # List of trainable variables of the layers we want to train
        var_list = [v for v in tf.trainable_variables()]

        # Op for calculating the loss
        with tf.name_scope("cross_ent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                          labels=y))

        global_step = tf.Variable(0, False)
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            decay_steps = int(train_set_size / batch_size * num_epochs_per_decay)
            learning_rate = tf.train.exponential_decay(
                learning_rate,
                global_step,
                decay_steps,
                learning_rate_decay_factor,
                staircase=True)
            # Create optimizer and apply gradient descent to the trainable variables
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        if prune:
            # Parse pruning hyperparameters
            prune_params=pruning.get_pruning_hparams().parse(prune_params)
            # Create a pruning object using the pruning specification
            p=pruning.Pruning(prune_params,global_step=global_step)
            # Add conditional mask update op. Executing this op will update all
            # the masks in the graph if the current global step is in the range
            # [begin_pruning_step, end_pruning_step] as specified by the pruning spec
            mask_update_op = p.conditional_mask_update_op()
            # Add summaries to keep track of the sparsity in different layers during training
            p.add_pruning_summaries()




        # Add the variables we train to the summary
        for var in var_list:
            tf.summary.histogram(var.name, var)
        # Add the loss to summary
        tf.summary.scalar('cross_entropy', loss)
        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)
        # Merge all summaries together
        merged_summary = tf.summary.merge_all()
        # Initialize the FileWriter
        writer = tf.summary.FileWriter(summary_path)


        # prepare the data
        img_train, label_train, labels_text_train = read_tfrecord('train', dataset_path,default_image_size=default_image_size)
        img_validation, label_validation, labels_text_validation = read_tfrecord('validation',dataset_path,default_image_size=default_image_size)
        coord = tf.train.Coordinator()
        


        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver()
        
        with tf.Session() as sess:

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)

            # Load the pretrained weights into the non-trainable layer
            model.load_initial_weights(sess)
            
            #start the input pipeline queue
            threads = tf.train.start_queue_runners(sess, coord=coord)

            # load the weights from checkpoint if there exists one
            model_saved = tf.train.get_checkpoint_state(checkpoint_path)
            if model_saved and model_saved.model_checkpoint_path:
                saver.restore(sess, model_saved.model_checkpoint_path)
                print('load model from ' + model_saved.model_checkpoint_path)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              summary_path))

            # Loop over number of epochs
            for epoch in range(num_epochs):
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

                highest_accuracy = 0                                                                       #highest accuracy by far
                if os.path.exists(highest_accuracy_path):
                    f = open(highest_accuracy_path, 'r')
                    highest_accuracy = float(f.read())
                    f.close()
                    print('highest accuracy from previous training is %f' % highest_accuracy)
                    
                train_batches_per_epoch = int(np.floor(train_set_size / batch_size))
                for step in range(train_batches_per_epoch):
                    # train the model
                    img, l, l_text = sess.run([img_train, label_train, labels_text_train])
                    _, sc, gl_step, lr = sess.run([train_op, score, global_step, learning_rate],
                                                  feed_dict={x:img,y:l,keep_prob:dropout_rate})
                    if prune:
                        # Update the masks by running the mask_update_op
                        sess.run(mask_update_op)

                    # Generate summary with the current batch of data and write to file
                    if step%log_step ==0 :
                        s ,aq= sess.run([merged_summary,accuracy], feed_dict={x:img,y:l,keep_prob: 1.})
                        writer.add_summary(s, epoch * train_batches_per_epoch + step)
                        print("global_step:" + str(gl_step) + ';learning_rate:' + str(lr)+';accuracy:',aq)


                    #validate the model and write checkpoint if the accuracy is higher
                    if step % checkpoint_step == 0 and step!=0:
                        val_batches_per_epoch = int(np.floor(validation_set_size / batch_size))
                        print("{} Start validation".format(datetime.now()))
                        test_acc = 0.
                        test_count = 0
                        for _ in range(val_batches_per_epoch):  # val_batches_per_epoch
                            #validate the model
                            img,l,l_text=sess.run([img_validation, label_validation, labels_text_validation])
                            acc = sess.run(accuracy,feed_dict={x:img,y:l,keep_prob:1.})
                            test_acc += acc
                            test_count += 1
                        test_acc /= test_count
                        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
                        # save the model if it is better than the previous best model
                        if test_acc > highest_accuracy:
                            print("{} Saving checkpoint of model...".format(datetime.now()))
                            highest_accuracy = test_acc
                            # save checkpoint of the model
                            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + '.ckpt')
                            # save_path = saver.save(sess, checkpoint_name, global_step=global_step)
                            f = open(highest_accuracy_path, 'w')
                            f.write(str(highest_accuracy))
                            f.close()
                            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                                           checkpoint_name))
            coord.request_stop()
            coord.join(threads)

def train_resnet_50(dataset_name='imagenet',
                  prune=False,
                  prune_params='',
                  learning_rate=conf.learning_rate,
                  num_epochs=conf.num_epochs,
                  batch_size=conf.batch_size,
                  learning_rate_decay_factor=conf.learning_rate_decay_factor,
                  num_epochs_per_decay=conf.num_epochs_per_decay,
                  log_step=conf.log_step,
                  checkpoint_step=conf.checkpoint_step,
                  summary_path=conf.root_path+'resnet'+conf.summary_path,
                  checkpoint_path=conf.root_path+'resnet'+conf.checkpoint_path,
                  highest_accuracy_path=conf.root_path+'resnet'+conf.highest_accuracy_path,
                  default_image_size=224,                                                                       #224 in the paper
                  ):
    if dataset_name is 'imagenet':
        num_class=conf.imagenet['num_class']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_size=conf.imagenet['validation_set_size']
        label_offset=conf.imagenet['label_offset']
        label_path=conf.imagenet['label_path']
        dataset_path=conf.imagenet['dataset_path']   
    
    x = tf.placeholder(tf.float32, [batch_size, default_image_size, default_image_size, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_class-label_offset])
    logits = inference(x,
                       num_classes=num_class-label_offset,
                       is_training=True,
                       )
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([cross_entropy_mean] + regularization_losses)

    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        global_step = tf.Variable(0, False)
        decay_steps = int(train_set_size / batch_size * num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)
        # Create optimizer and apply gradient descent to the trainable variables
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables()]

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('loss', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(summary_path)

    # prepare the input data
    img_train, label_train, labels_text_train = read_tfrecord('train', dataset_path)
    img_validation, label_validation, labels_text_validation = read_tfrecord('validation', dataset_path)
    coord = tf.train.Coordinator()

    graph_weight = 'resnet_v1_50.ckpt'
    ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver_res = tf.train.Saver(ref_vars)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        #load model from checkpoint supplied by google
        saver_res.restore(sess, graph_weight)

        # load the weights from checkpoint if there exists one
        #model_saved = tf.train.get_checkpoint_state(checkpoint_path)
        # if model_saved and model_saved.model_checkpoint_path:
        #     saver.restore(sess, model_saved.model_checkpoint_path)
        #     print('load model from '+model_saved.model_checkpoint_path)
        # else:
        #     print('no model restored')

        # saver.restore(sess,checkpoint_path)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          summary_path))
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            highest_accuracy= 0
            if os.path.exists(highest_accuracy_path):
                f = open(highest_accuracy_path, 'r')
                highest_accuracy = float(f.read())
                f.close()
                print('highest accuracy from previous training is %f' % highest_accuracy)

            # Get the number of training/validation steps per epoch
            train_batches_per_epoch = int(np.floor(train_set_size / batch_size))
            for step in range(train_batches_per_epoch):
                # get next batch of data
                img, l, l_text = sess.run([img_train, label_train, labels_text_train])

                # And run the training op
                _, gl_step, lr, aq = sess.run([train_op, global_step, learning_rate, accuracy],
                                              feed_dict={x: img, y: l})

                # Generate summary with the current batch of data and write to file
                if step % log_step == 0:
                    s, aq = sess.run([merged_summary, accuracy], feed_dict={x: img, y: l})
                    writer.add_summary(s, epoch * train_batches_per_epoch + step)
                    print("global_step:" + str(gl_step) + ';learning_rate:' + str(lr) + ';accuracy:', aq)

                if step % checkpoint_step == 0 and step!=0:
                    print("{} Start validation".format(datetime.now()))
                    test_acc = 0.
                    test_count = 0
                    val_batches_per_epoch = int(np.floor(validation_set_size / batch_size))
                    for _ in range(val_batches_per_epoch):  # val_batches_per_epoch
                        img_batch, label_batch, label_text_batch = sess.run([img_validation, label_validation, labels_text_validation])
                        acc = sess.run(accuracy, feed_dict={x: img_batch,y: label_batch,})
                        test_acc += acc
                        test_count += 1
                    test_acc /= test_count
                    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
                    # save the model if it is better than the previous best model
                    if test_acc > highest_accuracy:
                        print("{} Saving checkpoint of model...".format(datetime.now()))
                        highest_accuracy = test_acc
                        # save checkpoint of the model
                        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + '.ckpt')
                        # save_path = saver.save(sess, checkpoint_name, global_step=global_step)
                        f = open(highest_accuracy_path, 'w')
                        f.write(str(highest_accuracy))
                        f.close()
                        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                                       checkpoint_name))
        coord.request_stop()
        coord.join(threads)


def top_k_accuracy(predictions, labels, k, batch_size=conf.batch_size):
    a=tf.nn.in_top_k(predictions, labels, k=k)
    in_top1 = tf.to_float(a)
    num_correct = tf.reduce_sum(in_top1)
    return num_correct / batch_size


def read_label(dataset_name):
    if dataset_name is 'imagenet':
        s=conf.imagenet['label_path']
        label_offset=conf.imagenet['label_offset']
    f = open(s, "r")  # 设置文件对象
    data = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
    f.close()  # 关闭文件
    data=data[label_offset:]
    return data


if __name__ == '__main__':
    train_resnet_50()
    # train_alexnet(prune=False,
                  # prune_params='begin_pruning_step=1,end_pruning_step=100000,initial_sparsity=0.5,target_sparsity=0.7,sparsity_function_begin_step=1,sparsity_function_end_step=100000')