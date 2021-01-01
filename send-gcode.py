import serial, time, sys
from tqdm import tqdm, trange
# First, initialize the arm:
# M1112         Home
# G92 X0 Y0 Z0  Set Work Origin
# Then move it to a convenient real work origin:
# G0 X-100 Y-100
# and again:
# G92 X0 Y0 Z0  Set Work Origin
# The working area is then between:
# G0 X0 Y0
# G0 X0 Y200
# G0 X200 Y200
# G0 X200 Y0
# All this assuming Z=0 (home position)

# Mikey.exe 06/27/2020
gcode_file = sys.argv[1].strip()
file_read = open(gcode_file, 'r')
gcode = file_read.readlines()

line_counter = 0

ser = serial.Serial(port='/dev/ttyACM0', 
                    baudrate = 115200, 
                    parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, 
                    bytesize=serial.EIGHTBITS, 
                    timeout=1)

def wait_ok(ctype):
    x = ''
    time_now = time.time()
    
    wait_time = 0.1
    if ctype == "setup": # Give the first few commands longer to setup
        wait_time = 5.0
    elif ctype == 'toggle': # If turning on laser, don't wait
        wait_time = 0.00
    
    while "busy" in x or time.time() - time_now < wait_time:
        x=bytes.decode(ser.readline().strip(), errors="ignore")
        if len(x) > 0:
            if x == "o":
                x = "ok"
            elif 'ok' not in x and \
                 'busy' not in x and \
                 'wait' not in x and \
                 x[:1] != 'M':
                print(len(x.strip()), x)
        time.sleep(0.01)
    
started = -1
def serial_write(cmd, pbar):
    global line_counter, started
    line_counter += 1
    pbar.set_description('N{:4d} {:36s}'.format(line_counter, cmd))
    ser.write(str.encode("{cmd}\n\r".format(cmd=cmd)))

    ctype = "setup"
    if cmd == "G0 Z0":  
      # Final setup command
      started = 0
    elif cmd[0] == "M":
      # Turning laser on/off
      ctype = "toggle"
    elif cmd[0:2] == "G0" and started == 0:
      # First move towards burn start
      started = 1
    elif cmd[0] == "G" and started == 1:
      ctype = "move"  

    wait_ok(ctype)

def send(cmd, pbar):
    cmd = cmd.split(";")[0]
    if len(cmd.strip()) < 1:
        return
    serial_write(cmd, pbar)


pbar = tqdm(gcode, desc='{:42s}'.format('Command'), leave=True, ncols=100)
    
send('M1111', pbar) # Reset to origin position
send('M1112', pbar) # Go to home position

send('G90', pbar) # Absolute positioning
send('G92.1', pbar) # Machine coordinate system

try: 
  for line in pbar:
      send(line.strip(), pbar)
finally:
  print("Cleaning up!")
  send('M888 P15', pbar) # turn off laser
  send('M5', pbar) # custom: turn off cover fan
  send('M1112', pbar) # Go to home position

exit()
