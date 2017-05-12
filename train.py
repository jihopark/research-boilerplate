# code for tensorflow training


#!/usr/bin/env python
""" Training script for all the models"""
import time
import os

import tensorflow as tf
import numpy as np

# Training parameters
tf.flags.DEFINE_string("model_name", "", "Which model to train")
tf.flags.DEFINE_integer("batch_size", 32, "Number of batch size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 1000,
                        "Evaluate model on test set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning Rate of the model")
tf.flags.DEFINE_string("dataset_name", "","Which dataset to train")

# Misc Parameters
tf.flags.DEFINE_integer("memory_usage_percentage", 90, "Set Memory usage percentage (default:90)")
tf.flags.DEFINE_string("experiment_name", "", "Experiment name determines log dir")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
tf.logging.set_verbosity(tf.logging.ERROR)


def evaluate(model, sess, x_eval, y_eval):

def train_batch(model, sess, train_batch_gen):

def calculate_metrics(truth, prediction, writer=None, steps=0):

def train(model, train_set, valid_set, sess, train_iter):
    if not sess:
        return None
    sess.run(tf.global_variables_initializer())

    # create a summary writter for tensorboard visualization
    log_path = "%s/logs/%s/%s" % (os.path.dirname(os.path.abspath(__file__)), FLAGS.dataset_name, FLAGS.experiment_name)

    train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
    eval_writer = tf.summary.FileWriter(log_path + '/eval')

    # create a saver object to save and restore variables
    # https://www.tensorflow.org/programmers_guide/variables#checkpoint_files
    saver = tf.train.Saver()
    os.makedirs(log_path + "/ckpt")
    ckpt_path =  log_path + "/ckpt"

    print("Training Started with model: %s, log_path=%s" % (model.name, log_path))
	print("Dataset Name=%s, Experiment Name=%s" % (FLAGS.dataset_name, FLAGS.experiment_name))

    for i in range(train_iter):
        try:
            feed_dict = train_batch(model, sess, train_set)
            if i % 100 == 0:
                summary, cost, pred = sess.run([model.merge_summary,
                                           model.cost,
                                           model.prediction], feed_dict)
                train_precision, train_recall, train_f1 = calculate_metrics(feed_dict[model.labels], pred, train_writer, i)
                print("Iteration %s: mini-batch cost=%.4f" % (i, cost))
                print("Precision=%.4f, Recall=%.4f, F1=%.4f" % (train_precision,
                                                                train_recall,
                                                                train_f1
                                                               ))
                train_writer.add_summary(summary, i)
            if i % FLAGS.evaluate_every == 0:
                summary, cost, pred = eval(model, sess, valid_set["x"], valid_set["y"])
                valid_precision, valid_recall, valid_f1 = calculate_metrics(valid_set["y"],
                                                                            pred,
                                                                            eval_writer, i)
                print("\n**Validation set cost=%.4f" % cost)
                print("Precision=%.4f, Recall=%.4f, F1=%.4f\n" % (valid_precision,
                                                                  valid_recall,
                                                                  valid_f1))
                eval_writer.add_summary(summary, i)
            if i % FLAGS.checkpoint_every == 0:
                saver.save(sess, ckpt_path + ("/model-%s.ckpt" % i))
        except KeyboardInterrupt:
            print('Interrupted by user at iteration{}'.format(i))
            break

	saver.save(sess, ckpt_path + "/model-final.ckpt")
	saver.export_meta_graph(log_path + '/final.meta')

    train_writer.close()
    eval_writer.close()
    sess.close()
"""

"""
if __name__ == '__main__':
    name = FLAGS.dataset_name
	# load the data

	# create your model

	# create session for training
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage_percentage/100)
    session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

	# create your batch generator
    train_batch_generator = balanced_batch_gen(x_train,
                                               y_train,
                                               FLAGS.batch_size)

    with tf.Session(config=session_conf) as sess:
        train(model, train_batch_generator, {"x": x_test, "y": y_test}, sess, FLAGS.num_steps)
