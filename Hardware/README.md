The following are materials and setup instructions for a very simple educational "arm farm" based on a Rotrics DexArm and a Raspberry Pi 4 or Jetson Nano 4GB.


## Arm Farm Hardware Setup
**Table**: 
- $11.52: [2ft x 2ft block of wood](https://www.homedepot.com/p/Sanded-Plywood-Common-23-32-in-x-2-ft-x-2-ft-Actual-0-703-in-x-23-75-in-x-23-75-in-300950/202093835)
- $3.68 * 2: [1in x 36in Wood dowel](https://www.homedepot.com/p/8316U-1-in-x-36-in-Hardwood-Square-Dowel-10001818/203334085)
- $2.18: [1in hinge](https://www.homedepot.com/p/Everbilt-1-in-Zinc-Plated-Non-Removable-Pin-Narrow-Utility-Hinges-2-Pack-15161/202034166)
- $8.47: [1/4in magnets](https://www.homedepot.com/p/Master-Magnet-1-4-in-Dia-Neodymium-Rare-Earth-Magnet-Discs-with-Foam-Adhesive-12-Pack-97584/206503481)
- *Misc*: 1 1/4 to 1 1/2 in screws
- *Misc*: 4 3/4 in screws
- *Misc*: Superglue or equivalent

*Construction*
1. Cut 1ft off a dowel and then reconnect with the hinge
2. Cut 1ft off the second dowel.
3. Attach 1ft dowels from bottom at opposite edges (x-axis) of the board approximately in the middle (~1ft) from edges (y-axis) using the longer screws.
4. *optional*: Drill a small divet in center of the top of dowel that doesn't have a hinge and glue in a magent
5. *optional*: Mark where the hinged dowel lands on top of the magnet and drill out a small indent to place second magnet (this helps w/ alignment)
6. Remove rubber pads from Rotrics robot and screw into the edge of the board (shorter screw) -- again centered

*Cost*: 
$25.85 or ~$30 with glue and screws and <1hr of labor

*Grid*: 
Execute [Wide Grid GCode](gcodes/WideGrid.gcode) or import two adjacent copies of [Grid svg](gcodes/WideGrid_half.svg) to burn a 400mm wide by 160mm tall grid onto the table. Note: Use 90% of laser or lines will be very light/spotty.

<img src="images/BasicSetup.jpg" width="300">

**Robot is [Rotrics DexArm](https://www.rotrics.com/products/dexarm)**:
- $999 USD for Luxury Kit (incl pneumatic gripper and laser) or 
- $659 USD for Base kit + $259 USD for [Pneumatic Gripper](https://www.rotrics.com/products/pneumatic-kit)

**Blocks**
Small blocks can me made by cutting up the remaining Dowel or via the 3D printer attachment

**Overhead Light**
Recommend some overhead or bar lamp lighting (minimal shadows) to evenly light the surface

## Arm Farm Compute
**Computer**: 
- $119.95 for a [Raspberry Pi 4](https://www.canakit.com/raspberry-pi-4-4gb.html) Kit with 4GB of RAM

**Jetson Nano Option**:
- $90.00 for [Jeton Nano](https://www.nvidia.com/en-us/autonomous-machines/jetson-store/)
- $14.99 for [5V 6A 30W Power Supply ](https://smile.amazon.com/gp/product/B07QH3PL1Z/)
- $7.50 -- $18.99 for [Micro SD](http://smile.amazon.com/gp/product/B0887GP791)

**Camera and Accessories**: 
- $29.95 for [8mp Raspberry Pi Camera](https://www.canakit.com/raspberry-pi-noir-camera-v2-8mp.html)
- $5.40 for [1meter camera cable](https://smile.amazon.com/gp/product/B07J57LQQS)
- *Misc* $26.99 for [USB/Bluetooth Keyboard & Mouse](https://smile.amazon.com/gp/product/B07LH6TZSZ)
- *Misc* $84.99 for [Small touchscreen](https://smile.amazon.com/gp/product/B07L6WT77H) is completely unnecessary

*Cost*: $150ish or ... more if you buy more stuff

## Camera Connect
- *Cheap*: Tape or Zip-ties
- *A little more*: 3D printed clips see [CAD Directory](CAD) for files.  These can be easily printed and and tiled around the provided 3D printer platform if DexArm attachment was purchased:

<img src="images/DexArmWithClips.jpg" width="300">

`sudo usermod -a -G dialout ybisk`

Torchvision `git clone https://github.com/pytorch/vision.git -b v0.8.2; cd vision; sudo python3 setup.py install`

## LCD Displays

<img src="images/LCD.png">

- $11.99 for 2-Pack [16x2 LCD Display](https://smile.amazon.com/gp/product/B07S7PJYM6/ref=ppx_yo_dt_b_asin_title_o01_s00?ie=UTF8&psc=1)

Install drivers and see demo code:
[LCD from RPi Guy](https://github.com/the-raspberry-pi-guy/lcd) and `sudo apt install python3-smbus`

Might want to try `sudo apt-get install i2c-tools`  and `sudo i2cdetect -r -y 1` to see i2c setup

Notes for Jetson Nano (WIP):
In general things don't line up directly [NVIDIA Jetson Nano J41 Header Pinout](https://www.jetsonhacks.com/nvidia-jetson-nano-j41-header-pinout/).  Here we just have to re-orient:
- GND = 6
- VCC = 4
- SDA = 3
- SCL = 5
- Code:
  - `sudo i2cdetect -r -y 1` shows 27? Awesome.
  - Set the `BUS_NUMBER=1` [LCD Driver](https://github.com/the-raspberry-pi-guy/lcd/blob/master/drivers/i2c_dev.py#L10) 
  - Remove the RPi import. This won't work even if you install the library.
