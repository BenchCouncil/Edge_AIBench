# coding=utf8
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

# for MNIST class_num = 10
class_num = 10
batch_size = 128
hidden_size = 256
layer_num = 2
lr = 0.001
keep_prob = 0.5
feature_num = 13
epochs = 50
display_step = 30

model_file_prefix = ''


def load_inputs():
    inputs = []
    labels = []

    return inputs, labels


def sample(data, label, ratio):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    length = len(data)
    for i in range(length):
        if random.random() > ratio:
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
    return train_data, train_label, test_data, test_label


def pad_batch(x_batch):
    # 取x_batch中长度最大的值
    max_length = max([len(x) for x in x_batch])
    # 对x_batch中长度不够的补pad
    return [x + [[0] * feature_num] * (max_length - len(x)) for x in x_batch]


def one_hot_label(y_label):
    y_labels = []
    for label in y_label:
        tp = [0] * class_num
        tp[label] = 1
        y_labels.append(tp)
    return y_labels


def get_batches(xx, yy):
    # 定义生成器，用来获取batch
    for batch_i in range(0, len(xx) / batch_size):
        start_i = batch_i * batch_size
        x_batch = xx[start_i:start_i + batch_size]
        y_batch = yy[start_i:start_i + batch_size]
        # 补全序列
        pad_x_batch = np.array(pad_batch(x_batch))
        y_batch = np.array(one_hot_label(y_batch))

        # 记录每条记录的长度
        pad_x_lengths = []
        for x in x_batch:
            pad_x_lengths.append(len(x))

        yield pad_x_batch, pad_x_lengths, y_batch


def get_test(xx, yy):
    pad_x_batch = np.array(pad_batch(xx))
    y_batch = np.array(one_hot_label(yy))

    pad_x_lengths = []
    for x in xx:
        pad_x_lengths.append(len(x))

    yield pad_x_batch, pad_x_lengths, y_batch


def lstm_cell(keep_prob_):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=keep_prob_)
    return cell


def basic_lstm(X_, y_, x_length_, batch_size_, keep_prob_):
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(keep_prob_) for _ in range(layer_num)],
                                             state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size_, dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X_, sequence_length=x_length_, initial_state=init_state,
                                       time_major=False)
    lasth = outputs[:, -1, :]  # 或者 lasth = state[-1][1]

    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(lasth, W) + bias)
    cross_entropy = -tf.reduce_mean(y_ * tf.log(y_pre))
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return train_op, accuracy


def MNIST_task():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print(mnist.train.images.shape)

    _X = tf.placeholder(tf.float32, [None, 784])
    # shape = (batch_size, timestep_size, input_size)
    X_ = tf.reshape(_X, [-1, 28, 28])
    x_length_ = tf.placeholder(tf.int32, (None,), name='x_length')
    y_ = tf.placeholder(tf.float32, [None, class_num])
    batch_size_ = tf.placeholder(shape=[], dtype=tf.int32, name='batch_size')
    keep_prob_ = tf.placeholder(shape=[], dtype=tf.float32, name='keep_prob')

    train_op, accuracy = basic_lstm(X_, y_, x_length_, batch_size_, keep_prob_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = mnist.train.next_batch(batch_size)
            if (i + 1) % 200 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    _X: batch[0], x_length_: [28] * batch_size, y_: batch[1], keep_prob_: 1.0, batch_size_: batch_size})
                # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
                print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))
            sess.run(train_op,
                     feed_dict={_X: batch[0], x_length_: [28] * batch_size, y_: batch[1], keep_prob_: keep_prob,
                                batch_size_: batch_size})
        print("test accuracy %g" % sess.run(accuracy, feed_dict={
            _X: mnist.test.images, y_: mnist.test.labels, x_length_: [28] * mnist.test.images.shape[0], keep_prob_: 1.0,
            batch_size_: mnist.test.images.shape[0]}))


def other_task():
    inputs, labels = load_inputs()
    train_data, train_label, test_data, test_label = sample(inputs, labels, 0.1)

    # batch sequence embedding
    X_ = tf.placeholder(tf.float32, [None, None, feature_num], name='X')
    x_length_ = tf.placeholder(tf.int32, (None,), name='x_length')
    y_ = tf.placeholder(tf.float32, [None, class_num], name='y')
    batch_size_ = tf.placeholder(shape=[], dtype=tf.int32, name='batch_size')
    keep_prob_ = tf.placeholder(shape=[], dtype=tf.float32, name='keep_prob')

    train_op, accuracy = basic_lstm(X_, y_, x_length_, batch_size_, keep_prob_)

    checkpoint = model_file_prefix + "/trained_model.ckpt"
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epochs):
            count = 0
            for x_batch, x_length, y_batch in get_batches(train_data, train_label):
                sess.run(train_op,
                         feed_dict={X_: x_batch, x_length_: x_length, y_: y_batch, keep_prob_: keep_prob,
                                    batch_size_: batch_size})
                count += 1
                if count % display_step == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        X_: x_batch, x_length_: x_length, y_: y_batch, keep_prob_: 1.0, batch_size_: batch_size})
                    print("epch %d, step %d / %d, training accuracy %f" % (
                        epoch_i, count, len(train_data) / batch_size, train_accuracy))

        x_batch, x_length, y_batch = get_test(test_data, test_label)
        test_accuracy = sess.run(accuracy, feed_dict={
            X_: x_batch, x_length_: x_length, y_: y_batch, keep_prob_: 1.0, batch_size_: len(test_data)})
        print("test accuracy %f" % test_accuracy)
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')


def main():
    MNIST_task()


if __name__ == '__main__':
    main()