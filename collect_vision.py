import cv2, os
from tqdm import tqdm
from robot import Robot


# Start Camera
cam = cv2.VideoCapture(0)


rbt = Robot()

os.makedirs("vision_training")
os.makedirs("vision_training/imgs")

labels = open("vision_training/labels.txt",'wt')
X = [-40, 40]
Y = [240, 300]
Z = [-40, 0]

inc = 10

datum = 0
for x in tqdm(range(X[0], X[1]+inc, inc), ncols=50):
    for y in tqdm(range(Y[0], Y[1]+inc, inc), ncols=50):
        for z in range(Z[0], Z[1]+inc, inc):
            rbt.move(x,y,z)
            val, img = cam.read()
            if val:
                cv2.imwrite('vision_training/imgs/{}.png'.format(datum),img)
                labels.write("{} {:4.2f} {:4.2f} {:4.2f}\n".format(datum, x,y,z))
                datum += 1

labels.close()
print("Collected: {}".format(datum))

#img = data["imgs"][100][:,:,::-1]
#plt.imshow(img)
