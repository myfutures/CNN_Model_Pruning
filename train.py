import tensorflow as tf
import AlexNet
import numpy as np
from datetime import datetime
from generate_data_from_tfrecord import read_tfrecord
import config as conf
import os

def train_alexnet(dataset_name='imagenet',
                  learning_rate=conf.learning_rate,
                  num_epochs=conf.num_epochs,
                  batch_size=conf.batch_size,
                  learning_rate_decay_factor=conf.learning_rate_decay_factor,
                  num_epochs_per_decay=conf.num_epochs_per_decay,
                  dropout_rate=conf.dropout_rate,
                  log_step=conf.log_step,
                  checkpoint_step=conf.checkpoint_step,
                  summary_path=conf.summary_path,
                  checkpoint_path=conf.checkpoint_step,
                  highest_accuracy_path=conf.highest_accuracy_path
                  ):
    if dataset_name is 'imagenet':
        num_classes=conf.imagenet['num_classes']
        train_set_size=conf.imagenet['train_set_size']
        validation_set_size=conf.imagenet['validation_set_size']
        label_offset=conf.imagenet['label_offset']
        mean=conf.imagenet['mean']
        label_path=conf.imagenet['label_path']
        dataset_path=conf.imagenet['dataset_path']
        
        
        # prepare the data
        img_train, label_train, labels_text_train = read_tfrecord('train', dataset_path)
        coord = tf.train.Coordinator()

        keep_prob=tf.placeholder(tf.float32)                                        #placeholder for dropout rate
        # prepare to train the model
        model_train = AlexNet.AlexNet(img_train, keep_prob, num_classes-label_offset, [])
        # Link variable to model_train output
        score_train = model_train.fc8

        # List of trainable variables of the layers we want to train
        var_list = [v for v in tf.trainable_variables()]

        # Op for calculating the loss
        with tf.name_scope("cross_ent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_train,
                                                                          labels=label_train))

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

        # Evaluation op: Accuracy of the model_train
        with tf.name_scope("accuracy_train"):
            correct_pred = tf.equal(tf.argmax(score_train, 1), tf.argmax(label_train, 1))
            accuracy_train = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
        # Add the variables we train to the summary
        for var in var_list:
            tf.summary.histogram(var.name, var)
        # Add the loss to summary
        tf.summary.scalar('cross_entropy', loss)
        # Add the accuracy_train to the summary
        tf.summary.scalar('accuracy_train', accuracy_train)
        # Merge all summaries together
        merged_summary = tf.summary.merge_all()
        # Initialize the FileWriter
        writer = tf.summary.FileWriter(summary_path)

        #model validation
        # Validate the model_train on the entire validation set
        img_validation, label_validation, labels_text_validation = read_tfrecord('validation',dataset_path)
        model_validation=AlexNet.AlexNet(img_train, tf.constant(1), num_classes-label_offset, [])
        # Link variable to model_train output
        score_validation = model_validation.fc8
        with tf.name_scope("accuracy_validation"):
            correct_pred = tf.equal(tf.argmax(score_validation, 1), tf.argmax(label_validation, 1))
            accuracy_validation = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize an saver for store model_train checkpoints
        saver = tf.train.Saver()
        
        with tf.Session() as sess:

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Add the model_train graph to TensorBoard
            writer.add_graph(sess.graph)

            # Load the pretrained weights into the non-trainable layer
            model_train.load_initial_weights(sess)
            
            #start the input pipeline queue
            threads = tf.train.start_queue_runners(sess, coord=coord)

            # load the weights from checkpoint if there exists one
            model_saved = tf.train.get_checkpoint_state(checkpoint_path)
            if model_saved and model_saved.model_checkpoint_path:
                saver.restore(sess, model_saved.model_checkpoint_path)
                print('load model_train from ' + model_saved.model_checkpoint_path)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              summary_path))

            # Loop over number of epochs
            for epoch in range(num_epochs):
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

                highest_accuracy = 0                                                                       #highest accuracy_train by far
                if os.path.exists(highest_accuracy_path):
                    f = open(highest_accuracy_path, 'r')
                    highest_accuracy = float(f.read())
                    f.close()
                    print('highest accuracy_train from previous training is %f' % highest_accuracy)
                    
                train_batches_per_epoch = int(np.floor(train_set_size / batch_size))
                for step in range(train_batches_per_epoch):
                    # train the model
                    _, sc, gl_step, lr = sess.run([train_op, score_train, global_step, learning_rate],feed_dict={keep_prob:dropout_rate})

                    # Generate summary with the current batch of data and write to file
                    if step%log_step ==0 :
                        print("global_step:" + str(gl_step) + ';learning_rate:' + str(lr))
                        s = sess.run(merged_summary, feed_dict={keep_prob: 1.})
                        writer.add_summary(s, epoch * train_batches_per_epoch + step)

                    #validate the model and write checkpoint if the accuracy is higher
                    if step % checkpoint_step == 0 and step!=0:
                        val_batches_per_epoch = int(np.floor(validation_set_size / batch_size))
                        print("{} Start validation".format(datetime.now()))
                        test_acc = 0.
                        test_count = 0
                        for _ in range(val_batches_per_epoch):  # val_batches_per_epoch
                            #validate the model
                            acc = sess.run(accuracy_validation)
                            test_acc += acc
                            test_count += 1
                        test_acc /= test_count
                        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
                        # save the model_train if it is better than the previous best model_train
                        if test_acc > highest_accuracy:
                            print("{} Saving checkpoint of model_train...".format(datetime.now()))
                            highest_accuracy = test_acc
                            # save checkpoint of the model_train
                            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + '.ckpt')
                            save_path = saver.save(sess, checkpoint_name, global_step=global_step)
                            f = open(highest_accuracy_path, 'w')
                            f.write(str(highest_accuracy))
                            f.close()
                            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                                           checkpoint_name))
            coord.request_stop()
            coord.join(threads)


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
    train_alexnet()