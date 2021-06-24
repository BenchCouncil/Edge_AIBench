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
    input_data = tf.placeholder(shape=[4,224,224,3], dtype=float)
    sess = tf.Session()
    with gfile.FastGFile('./pretrained_model/actionRecog.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())
    input_x = sess.graph.get_tensor_by_name("input_tensor:0")
    op = sess.graph.get_tensor_by_name("fc8u/BiasAdd:0")


    def DoFormat(self, request, context):
        str = request.text
        print("get request")
        decode_img = base64.b64decode(str)
        # decode_img = str
        img = np.fromstring(decode_img, dtype=np.float)
        img = np.reshape(img, [4,224,224,3])
        ret = self.sess.run(self.op, feed_dict={self.input_x: img})
        ret = ret.flatten()
        strings = ["%.8f" % number for number in ret]
        ret = ' '.join(item for item in strings)
        print("-------------")
        return data_pb2.actionresponse(text=ret)


def serve():
    MAX_MESSAGE_LENGTH = 6422533
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4),options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
               ])  # 创建一个服务器
    data_pb2_grpc.add_FormatDataServicer_to_server(FormatData(), grpcServer)  # 在服务器中添加派生的接口服务（自己实现了处理函数）
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)  # 添加监听端口
    grpcServer.start()  # 启动服务器
    try:
        while True:
            print('start===============>')
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)  # 关闭服务器


if __name__ == '__main__':
    serve()
