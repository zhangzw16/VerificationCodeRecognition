import h5py

def createTrain():
    f=h5py.File("./datasets/codes.hdf5","r")
    dset = f['codename']
    # print(dset.shape)
    for i in dset.value:
        print(i.decode())
        name = i.decode()
        



if __name__ == '__main__':
    createTrain()