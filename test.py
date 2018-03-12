import h5py
from PIL import Image

f = h5py.File("./datasets/codes.hdf5","r")
dset_code = f['codename']
for i in dset_code.value:
    # print(i.decode())
    name = i.decode()
    print(name)


dset_image = f['images']
img = dset_image.value[0]
img = Image.fromarray(img)
img.show()
