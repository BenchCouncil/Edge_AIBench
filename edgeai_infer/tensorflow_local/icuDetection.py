import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import time


if __name__ == '__main__':
    input_data = np.ones([32,48,76])
    sess = tf.Session()
    with gfile.FastGFile('./pretrained_model/icu_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='haha')
    sess.run(tf.global_variables_initializer())
    input_x = sess.graph.get_tensor_by_name("haha/X:0")
    op = sess.graph.get_tensor_by_name("haha/Dense_out/Sigmoid:0")
    for i in range(1000):
        ret = sess.run(op, feed_dict={input_x: input_data})
    for i in range(1000):
        ret = sess.run(op, feed_dict={input_x: input_data})
    start_1 = time.time()
    for i in range(1000):
        ret = sess.run(op, feed_dict={input_x: input_data})
    end_1 = time.time()
    print("Run model time: ", (end_1 - start_1)/1000)
