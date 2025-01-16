import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# Define the SVG path data
svg_path_data = "M5.32826 2.44919C6.17856 2.03131 7.11811 1.80739 8.07844 1.80247C9.54615 1.79495 10.9705 2.29944 12.1063 3.22907C13.2421 4.15869 14.0182 5.45525 14.3009 6.89549C14.5837 8.33572 14.3554 9.82946 13.6554 11.1195C12.9553 12.4095 11.8274 13.4151 10.4658 13.963C9.10418 14.5109 7.59412 14.5669 6.19567 14.1214C4.79721 13.6758 3.59789 12.7566 2.80423 11.5219C2.01058 10.2873 1.67226 8.81455 1.84755 7.35733C1.907 6.86318 1.55459 6.4144 1.06044 6.35496C0.566287 6.29552 0.117511 6.64792 0.0580663 7.14207C-0.167306 9.01564 0.26767 10.9092 1.28808 12.4965C2.3085 14.0839 3.85049 15.2658 5.6485 15.8387C7.44651 16.4116 9.38801 16.3396 11.1387 15.6351C12.8893 14.9306 14.3395 13.6377 15.2395 11.9791C16.1396 10.3205 16.4331 8.39999 16.0696 6.54827C15.706 4.69655 14.7082 3.02953 13.2479 1.8343C11.7876 0.639072 9.95626 -0.00955937 8.06921 0.000106488C6.78118 0.00670406 5.52223 0.319717 4.39229 0.902597L4.06937 0.368993C3.85454 0.0140161 3.32578 0.0600553 3.17553 0.446817L2.17369 3.02579C2.05488 3.33164 2.25789 3.6671 2.58396 3.70372L5.33341 4.01249C5.74573 4.05879 6.0318 3.61171 5.81697 3.25674L5.32826 2.44919ZPB0 0 14 14QB"

# Parse the SVG path data
path_data = []
commands = {'M': Path.MOVETO, 'L': Path.LINETO, 'C': Path.CURVE4, 'Z': Path.CLOSEPOLY}
parts = svg_path_data.split(' ')
i = 0
while i < len(parts):
    cmd = parts[i][0]
    if cmd in commands:
        path_data.append((commands[cmd], [float(x) for x in parts[i][1:].split()]))
    elif cmd == 'P':
        path_data.append((Path.MOVETO, [float(x) for x in parts[i][1:].split()]))
    elif cmd == 'Q':
        path_data.append((Path.CLOSEPOLY, [0, 0]))
    i += 1

# Create the Path object
codes, verts = zip(*path_data)
path = Path(verts, codes)

# Create a PathPatch object
patch = patches.PathPatch(path, facecolor='none', edgecolor='black')

# Plotting the SVG path
fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(patch)
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 15)
ax.set_aspect('equal', adjustable='datalim')
ax.axis('off')  # Hide axes for visualization

plt.title("Visualization of SVG Path")
plt.show()
