import grpc
from example import data_pb2, data_pb2_grpc
import base64
import numpy as np
import time

_HOST = '10.26.10.40'
_PORT = '8021'


def run():

    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    client = data_pb2_grpc.FormatDataStub(channel=conn)
    str = base64.b64encode(np.ones([4,224,224,3]).tostring())
    # str = np.ones([4, 224, 224, 3]).tostring()
    # print(str)
    for i in range(100):
        response = client.DoFormat(data_pb2.actionrequest(text=str))

    timetable = []
    for i in range(100):
        start = time.time()
        response = client.DoFormat(data_pb2.actionrequest(text=str))
        end = time.time()
        print("------------------------------")
        print("Time consuming is : %.5f" % (end - start))
        print("------------------------------")
        # print("received: " + response.text)
        timetable.append((end - start))
    print("Average time is : ", np.mean(timetable))


if __name__ == '__main__':
    run()
