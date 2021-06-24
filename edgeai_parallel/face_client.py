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
    str = base64.b64encode(np.ones([1,112,96,3]).tostring())
    for i in range(20):
        response = client.DoFormat(data_pb2.actionrequest(text=str))
    timetable = []
    for i in range(20):    
        start = time.time()
        response = client.DoFormat(data_pb2.actionrequest(text=str))
        end = time.time()
        print("Time consuming is : %.5f" % (end-start))
        timetable.append(end - start)
    print("------------------------------")
    print("Time consuming is : %.5f" % np.mean(timetable))
    print("------------------------------")


if __name__ == '__main__':
    run()
