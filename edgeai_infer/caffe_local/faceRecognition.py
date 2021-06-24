import caffe
import numpy as np
import time
from PIL import Image
import matplotlib.image as mpimg


if __name__ == '__main__':
    model_def = './pretrained_model/sphereface_model.prototxt'
    model_weights = './pretrained_model/sphereface.caffemodel'
    # caffe.set_mode_cpu()
    net = caffe.Classifier(model_def, model_weights)
    input_data = np.ones([1,112,96,3])
    for i in range(1000):
        prediction = net.predict(input_data, oversample=False)
    for i in range(1000):
         prediction = net.predict(input_data, oversample=False)
    start = time.time()
    for i in range(1000):
        img = mpimg.imread('/EdgeAI/edgeAIbench/facerecognition2020/dataset/lfw_112_96/Al_Leiter/Al_Leiter_0001.jpg')
        input_data = img.reshape([1,112,96,3])
        prediction = net.predict(input_data, oversample=False)
    end = time.time()
    print("Run Model time is ", (end - start)/1000)
