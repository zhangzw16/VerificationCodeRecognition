import h5py
from PIL import Image
import numpy as np

def createTrainSet():
    trainList = []
    f=h5py.File("./datasets/codes.hdf5","r+")
    dset_code = f['codename']
    # print(dset.shape)
    for i in dset_code.value:
        # print(i.decode())
        name = i.decode()
        img = Image.open("./images/{}.jpg".format(name))
        imgMatrix = np.array(img)
        trainList.append(imgMatrix)
    # print(trainList[1].shape)
    f.create_dataset('images', data=trainList)


if __name__ == '__main__':
    createTrainSet()