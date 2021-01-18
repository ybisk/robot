# Rotrics Code

`python manual.py` for issuing individual test commands

`python send-gcode.py` iterates through a gcode file (doesn't wait properly)

`python range.py` calculate statistics about a gcode file for helping with placement and draw image/path


# Arm Farm Setup
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

**Robot is [Rotrics DexArm](https://www.rotrics.com/products/dexarm)**:
- $999 USD for Luxury Kit (incl pneumatic gripper and laser) or 
- $659 USD for Base kit + $259 USD for [Pnuumatic Gripper](https://www.rotrics.com/products/pneumatic-kit)

**Blocks**
Small blocks can me made by cutting up the remaining Dowel 
