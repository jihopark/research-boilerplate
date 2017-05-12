import tensorflow as tf


class Model(object):
    def __init__(self, name):
        self.name = name

        with tf.name_scope("input"):
            self.X = tf.placeholder(
                tf.int32, [None, sequence_length], name="X")
            self.labels = tf.placeholder(
                tf.int64, [None, 1], name="labels")

        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Embedding layer

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
                with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, n_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.prediction = tf.argmax(self.logits, 1, name="prediction")

        # CalculateMean cross-entropy loss
        with tf.name_scope("training"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels_one_hot)
            self.cost = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            tf.summary.scalar("cost", self.cost)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.cost)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        self.merge_summary = tf.summary.merge_all()
