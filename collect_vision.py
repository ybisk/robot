import cv2, os
import random as r
from tqdm import tqdm
from robot import Robot

def gstreamer_pipeline(capture_width=1280, capture_height=720, 
                       display_width=1280, display_height=720,
                       framerate=60, flip_method=0):
  return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

nano = True
if nano:
  cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
else:
  # Start Camera
  cam = cv2.VideoCapture(0)


def capture(Xr = [-40, 40], Yr = [240, 300], Zr = [-40, 0], inc = 5, grid = True):
  if grid:
    X = list(range(Xr[0], Xr[1]+inc, inc))
    Y = list(range(Yr[0], Yr[1]+inc, inc))
    Z = list(range(Zr[0], Zr[1]+2*inc, 2*inc))
  else:
    X = [round(r.random()*(Xr[1]-Xr[0]) + Xr[0],1) \
       for _ in range(int((Xr[1]-Xr[0])/inc))]
    Y = [round(r.random()*(Yr[1]-Yr[0]) + Yr[0],1) \
       for _ in range(int((Yr[1]-Yr[0])/inc))]
    Z = [round(r.random()*(Zr[1]-Zr[0]) + Zr[0],1) \
       for _ in range(int((Zr[1]-Zr[0])/inc))]
    X.sort()
    Y.sort()
    Z.sort()


  if cam.isOpened():
    rbt = Robot()

    dir_name = "vision_training_{}".format("grid" if grid else "rand")
    os.makedirs(dir_name)
    os.makedirs("{}/imgs".format(dir_name))
    labels = open("{}/labels.txt".format(dir_name),'wt')

    datum = 0
    for x in tqdm(X, ncols=50):
      for y in tqdm(Y, ncols=50):
        for z in Z:
          rbt.move(x,y,z)
          val, img = cam.read()
          if val:
            cv2.imwrite(dir_name + '/imgs/{0:04d}.png'.format(datum),img)
            labels.write("{:0:04d} {:4.2f} {:4.2f} {:4.2f}\n".format(datum, x,y,z))
            datum += 1

    labels.close()
    print("Collected: {}".format(datum))


if __name__ == "__main__":
  capture()
