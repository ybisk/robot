import serial, time, sys
from tqdm import tqdm, trange

# https://manual.rotrics.com/get-start/rotary-module#2-g-code-commands-for-rotary-module

ser = serial.Serial(port='/dev/ttyACM0', 
                    baudrate = 115200, 
                    parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, 
                    bytesize=serial.EIGHTBITS, 
                    timeout=1)

def wait_ok():
    x = ''
    time_now = time.time()
    
    wait_time = 0.5
    while 'busy' in x or time.time() - time_now < wait_time:
        x=bytes.decode(ser.readline().strip(), errors="ignore")
        if len(x) > 0:
            if x == "o":
                x = "ok"
            elif 'ok' not in x and \
                 'wait' not in x and \
                 x[:1] != 'M':
                print(len(x.strip()), x)
        time.sleep(0.01)
    
def serial_write(cmd):
    ser.write(str.encode(f"{cmd}\n\r"))
    wait_ok()

def send(cmd):
    cmd = cmd.split(";")[0]
    if len(cmd.strip()) < 1:
        return
    serial_write(cmd)


send('M1112') # Go to home position

send('G90') # Absolute positioning

# G92.1 reset the working height. 
send('G92.1') # Machine coordinate system

send('G0 X0 Y200 Z0') # 0 200 0


try: 
  while True:
      a = input("location: ")
      if "laser" in a:
        a = a.split()
        if len(a) == 2:
          send('M3 S{}'.format(a[1])) # set laser strength
        else:
          send('M3 S5') # very light laser
      elif "off" in a:
        send('M888 P15') # turn off laser
      
      elif "rotary" in a:
        send('M888 P6') # Set the current end effector as a rotary module and initialize it
        send("M2100")   # Initialize the rotary module every time DexArm restarts
      elif "pos" in a:
        send("M2101")   # Read the current rotary position
      elif "R" in a:
        send(f"M2101 {a}") # Rotate n degrees in the clockwise direction
      elif "P" in a:
        send(f"M2101 {a}") # Rotate to position n degree
      elif "pick" in a:
        send("M1000") # air pump box to pump in
      elif "place" in a:
        send("M1002") # air pump box to release air
      elif "blow" in a:
        send("M1001") # air pump box to pump out
      elif "stop" in a:
        send("M1003") # stop air pump box
      else:
        send(a.strip())
finally:
  print("Cleaning up!")
  send('M400') # All GCode processing to pause and wait in a loop until all moves in the planner are completed.
  send('M5') # Done
  send('M888 P15') # turn off laser
  send('M1112') # Go to home position

exit()
