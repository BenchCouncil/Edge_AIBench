import time
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from xmlrpc.server import SimpleXMLRPCServer


if __name__ == '__main__':
    input_data = np.ones([4,224,224,3])
    sess = tf.Session()
    with gfile.FastGFile('./pretrained_model/actionRecog.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())
    input_x = sess.graph.get_tensor_by_name("input_tensor:0")
    op = sess.graph.get_tensor_by_name("fc8u/BiasAdd:0")
    for i in range(1000):
        ret = sess.run(op, feed_dict={input_x: input_data})
    for i in range(1000):
        ret = sess.run(op, feed_dict={input_x: input_data})
    start = time.time()
    for i in range(1000):
        ret = sess.run(op, feed_dict={input_x: input_data})
    end = time.time()
    print("Run model time: ", (end - start)/1000)
