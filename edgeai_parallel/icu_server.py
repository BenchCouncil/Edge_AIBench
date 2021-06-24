import grpc
import time
from concurrent import futures
from example import data_pb2, data_pb2_grpc
import base64
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '7777'


class FormatData(data_pb2_grpc.FormatDataServicer):
    input_data = tf.placeholder(shape=[32, 48, 76], dtype=float)
    sess = tf.Session()
    with gfile.FastGFile('./pretrained_model/icu_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='haha')
    sess.run(tf.global_variables_initializer())
    input_x = sess.graph.get_tensor_by_name("haha/X:0")
    op = sess.graph.get_tensor_by_name("haha/Dense_out/Sigmoid:0")

    def DoFormat(self, request, context):
        str = request.text
        print("get request")
        decode_img = base64.b64decode(str)
        img = np.fromstring(decode_img, dtype=np.float)
        img = np.reshape(img, (32, 48, 76))
        ret = self.sess.run(self.op, feed_dict={self.input_x: img})
        strings = ["%.8f" % number for number in ret]
        ret = ' '.join(item for item in strings)
        return data_pb2.actionresponse(text=ret)


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    data_pb2_grpc.add_FormatDataServicer_to_server(FormatData(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            print('start===============>')
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()

