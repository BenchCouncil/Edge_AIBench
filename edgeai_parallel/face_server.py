import grpc
import time
from concurrent import futures
from example import data_pb2, data_pb2_grpc
import base64
import numpy as np
import caffe

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '7777'

caffe.set_mode_cpu()
class FormatData(data_pb2_grpc.FormatDataServicer):
    model_def = './pretrained_model/sphereface_model.prototxt'
    model_weights = './pretrained_model/sphereface.caffemodel'

    def DoFormat(self, request, context):
        str = request.text
        print("get request")
        decode_img = base64.b64decode(str)
        # decode_img = str
        img = np.fromstring(decode_img, dtype=np.float)
        img = np.reshape(img, [1,112,96,3])
        net = caffe.Classifier(self.model_def, self.model_weights)
        prediction = net.predict(img, oversample=False)
        ret = prediction.flatten()
        strings = ["%.8f" % number for number in ret]
        ret = ' '.join(item for item in strings)
        print("-------------")
        return data_pb2.actionresponse(text=ret)


def serve():
    MAX_MESSAGE_LENGTH = 6422533
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4),options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
               ])
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
