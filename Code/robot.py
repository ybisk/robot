import serial, time, sys

class Robot:
  def __init__(self):
    self.ser = serial.Serial(port='/dev/ttyACM0', 
                        baudrate = 115200, 
                        parity=serial.PARITY_NONE, 
                        stopbits=serial.STOPBITS_ONE, 
                        bytesize=serial.EIGHTBITS, 
                        timeout=1)
    self.reset()

  def wait_ok(self):
      x = ''
      time_now = time.time()
      
      wait_time = 0.1
      while 'busy' in x or time.time() - time_now < wait_time:
          x=bytes.decode(self.ser.readline().strip(), errors="ignore")
          if len(x) > 0:
              if x == "o":
                  x = "ok"
              if 'ok' not in x and \
                 'wait' not in x and \
                 x[:1] != 'M':
                  print(len(x.strip()), x)
          time.sleep(0.01)
      
  def serial_write(self, cmd):
      self.ser.write(str.encode("{cmd}\n\r".format(cmd=cmd)))
      self.wait_ok()

  def send(self, cmd):
      cmd = cmd.split(";")[0]
      if len(cmd.strip()) < 1:
          return
      self.serial_write(cmd)

  def reset(self):
    self.send('M1112') # Go to home position
    self.send('G90') # Absolute positioning
    # G92.1 reset the working height. 
    self.send('G92.1') # Machine coordinate system
    self.send('G0 X0 Y200 Z0') # 0 200 0

  def clean(self):
    self.send('M400') # All GCode processing to pause and wait in a loop until all moves in the planner are completed.
    self.send('M5') # Done
    self.send('M888 P15') # turn off laser
    self.send('M1112') # Go to home position

  def move(self, X=0, Y=240, Z=0):
    self.send("G0 X{} Y{} Z{}".format(X,Y,Z))
