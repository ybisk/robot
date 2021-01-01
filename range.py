import os, sys
from PIL import Image, ImageDraw
import scipy.misc
import numpy as np

fname = sys.argv[1].split(".gcode")[0]
stats = open("{}.txt".format(fname), 'wt')
lines = [line.strip().split() for line in open(sys.argv[1])]
vals = {"X":[], "Y":[], "Z":[]}
Z = 0
for line in lines:
  if len(line) > 0 and (line[0] == "G0" or line[0] == "G1"):
    for coord in line[1:]:
      d = coord[0]
      v = float(coord[1:])
      if d in vals:
        if d == "Z" and line[-1] == "F150":
            Z += v
            vals[d].append(Z)
        else:
          vals[d].append(v)

print("Ranges:")
stats.write("Ranges:\n")
for d in vals:
  print("{:3} {:4} {:4}".format(d, min(vals[d]), max(vals[d])))
  stats.write("{:3} {:4} {:4}\n".format(d, min(vals[d]), max(vals[d])))

def ctr(vs):
  return (max(vs) + min(vs))/2

print("Center:\n {:4.2f} {:4.2f} {:4.2f}".format(ctr(vals["X"]), ctr(vals["Y"]), ctr(vals["Z"])))
stats.write("Center:\n {:4.2f} {:4.2f} {:4.2f}\n".format(ctr(vals["X"]), ctr(vals["Y"]), ctr(vals["Z"])))


BL = int(10*min(vals["X"])), int(10*min(vals["Y"]))
TR = int(10*max(vals["X"])), int(10*max(vals["Y"]))

img = np.ones((TR[1] - BL[1] + 1, TR[0] - BL[0] + 1, 3), dtype=np.int32)
img = 255*img

start = [0,0,0]
for line in lines:
  if len(line) > 0 and line[0] == "G0":
    for coord in line[1:]:
      d = coord[0]
      v = int(10*float(coord[1:]))
      if d == "X":
        start[0] = v
      elif d == "Y":
        start[1] = v
      elif d == "Z":
        if line[-1] == "F150":
          start[2] += v
  elif len(line) > 0 and line[0] == "G1":
    end = [0,0]
    for coord in line[1:]:
      d = coord[0]
      v = int(10*float(coord[1:]))
      if d == "X":
        end[0] = v
      elif d == "Y":
        end[1] = v

    # TODO: THIS IS WRONG BUT CURRENT GCODE IS JUST HORIZONTAL LINES
    color = 0 if start[2] < -10 else 1 if start[2] < 0 else 2
    for i in range(start[0] - BL[0], end[0] - BL[0]):
      img[start[1] - BL[1]][i][color] = 0

im = Image.fromarray(np.flipud(img.astype(np.uint8)))
im.save("{}.jpg".format(fname))
stats.close()
