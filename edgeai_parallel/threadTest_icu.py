import threading
import time
_HOST = '10.26.10.91'
_PORT = '42068'  # icu / action:42068  lane: face:33938
import grpc
from example import data_pb2, data_pb2_grpc
import base64
import numpy as np
from matplotlib import pyplot as plt
timetable = []


class MyThread(threading.Thread):
    def __init__(self,arg):
        super(MyThread, self).__init__()
        self.arg=arg

    def run(self):#定义每个线程要运行的函数
        conn = grpc.insecure_channel(_HOST + ':' + _PORT)
        client = data_pb2_grpc.FormatDataStub(channel=conn)
        #  icu:[32,48,76] action:[4,224,224,3] lane:[1,288,800,3] face:[1,112,96,3]
        str = base64.b64encode(np.ones([1,288,800,3]).tostring())
        response = client.DoFormat(data_pb2.actionrequest(text=str))
        print("------------------",self.arg)
        timetable.append(time.time()-start)
        print(timetable)
        np.save("./records/lane_gpu.npy", timetable)

# t = MyThread(0)  # lane 需要取消这部分注释代码
# t.start()
# time.sleep(8)

start = time.time()
concurrent_num = list(range(1,30,1))
for num in concurrent_num:
    for i in range(num):
        t =MyThread(i)
        t.start()
        time.sleep(0.001)
    time.sleep(8)  # caffe需要sleep, 否则资源加载会失败 我设为了8， tensorflow直接注释掉