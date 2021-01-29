import cv2, os
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


def capture(X = [-40, 40], Y = [240, 300], Z = [-40, 0], inc = 10):
  if cam.isOpened():
    rbt = Robot()

    os.makedirs("vision_training")
    os.makedirs("vision_training/imgs")

    labels = open("vision_training/labels.txt",'wt')

    datum = 0
    for x in tqdm(range(X[0], X[1]+inc, inc), ncols=50):
      for y in tqdm(range(Y[0], Y[1]+inc, inc), ncols=50):
        for z in range(Z[0], Z[1]+inc, inc):
          rbt.move(x,y,z)
          val, img = cam.read()
          if val:
            cv2.imwrite('vision_training/imgs/{0:04d}.png'.format(datum),img)
            labels.write(f"{datum:04d} {x:4.2f} {y:4.2f} {z:4.2f}\n")
            datum += 1

    labels.close()
    print("Collected: {}".format(datum))


if __name__ == "__main__":
  capture()
