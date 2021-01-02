import cv2
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt
from robot import Robot


# Start Camera
cam = cv2.VideoCapture(0)


rbt = Robot()


data = {"imgs":[], "locs":[]}
X = [-80, 80]
Y = [240, 300]
Z = [-40, 0]

inc = 10

for x in tqdm(range(X[0], X[1]+inc, inc)):
    for y in range(Y[0], Y[1]+inc, inc):
        for z in range(Z[0], Z[1]+inc, inc):
            rbt.move(x,y,z)
            val, img = cam.read()
            if val:
                data["imgs"].append(img)
                data["locs"].append([x,y,z])


print("Collected: ", len(data["imgs"]))


f = h5py.File('training.hdf5', 'w')
f.create_dataset("imgs", data=data["imgs"])
f.create_dataset("locs", data=data["locs"])
f.close()


#img = data["imgs"][100][:,:,::-1]
#plt.imshow(img)
