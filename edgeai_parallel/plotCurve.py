from matplotlib import pyplot as plt
import numpy as np

def plot_cpu_gpu_compare(cpupath, gpupath, bias, title):
    data_1 = np.load(cpupath)
    data_2 = np.load(gpupath)
    b1 = []
    b2 = []
    it = 0
    for i in range(30):
        if it >= len(data_1) or it >= len(data_2):
            break
        b1.append(data_1[it]-bias*i)  # -5*i
        b2.append(data_2[it]-bias*i)
        it += i + 2
    l1, = plt.plot(b1)
    l2, = plt.plot(b2)
    plt.legend(handles = [l1, l2,], labels = ['CPU', 'GPU'], loc = 'best')
    plt.title(title)
    plt.show()
    name = "./records/" + title+".jpg"
    plt.savefig(name)

timeshift = 0  # icu/action:0  lane/face:8
name = "action"
path_1 = "./records/" + name + "_cpu.npy"
path_2 = "./records/" + name + "_gpu.npy"
plot_cpu_gpu_compare(path_1, path_2, timeshift, name)
