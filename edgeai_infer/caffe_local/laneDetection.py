import caffe
import numpy as np
import time

if __name__ == '__main__':
    model_def = './pretrained_model/lane.prototxt'
    model_weights = './pretrained_model/lane.caffemodel'
    input_data = np.ones([1, 288, 800, 3])
    # caffe.set_mode_cpu()
    net = caffe.Classifier(model_def, model_weights)
    start = time.time()
    for i in range(100):
        prediction = net.predict(input_data, oversample=False)
    end = time.time()
    print("Run Model time is ", (end-start)/100)
