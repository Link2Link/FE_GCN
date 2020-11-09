import matplotlib.pyplot as plt
import re
import os
import numpy as np
log_dir = "../example/semseg/log"
log_name = "20201104-001819"
log_file = os.path.join(log_dir, log_name, log_name+'.log')


if __name__ == '__main__':
    with open(log_file, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        print(data)
    pattern = re.compile("class 0: 0.\d*")
    result = re.findall(pattern, data)
    mIOU = np.zeros((len(result),5))
    for i in range(5):
        pattern = re.compile("class {c}: 0.\d*".format(c=i))
        result = re.findall(pattern, data)
        for epoch in range(len(result)):
            IOU = result[epoch]
            IOU = float(IOU[9:])
            mIOU[epoch, i] = IOU
    plt.figure()
    for i in range(5):
        mIOU_i = mIOU[:, i]
        plt.plot(mIOU_i, label='class %d'%(i))
        plt.text(len(mIOU_i)-1, mIOU_i[-1], r'(epoch %d, %f)'%(len(mIOU_i)-1,mIOU[-1,i]), fontsize=10)
        best = np.argmax(mIOU_i)
        plt.text(best - 2, mIOU_i[best]-0.05, r'best:(epoch %d, %f)' % (best, mIOU_i[best]), fontsize=10)

        print("class {0}, best at epoch {1} {2}".format(i, best, mIOU_i[best]))
    plt.legend()
    plt.show()

