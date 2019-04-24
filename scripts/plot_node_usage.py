# NOTE: This script is generally to be called by plot_node_usage.sh

# Import relevant modules
import numpy as np
import sys
import matplotlib.pyplot as plt

# Read in the arguments to this script: the temporary text file that holds the timing information, the figure size to create, and the directory in which to create the new figure
data = np.loadtxt(sys.argv[1], dtype='str')
myfigsize = eval(sys.argv[2])
figdir = sys.argv[3]

# Extract the data from the data array
times = np.array(data[:,[0,3]], dtype='uint32')
times = ( times - np.min(times) ).astype('uint16')
nodes = data[:,1]
unodes = np.unique(nodes)
hpsets = np.array(range(data.shape[0]), dtype='uint8')
hpsets2 = data[:,2]

# Populate the array that will hold the timing data in plotting form
on_times = np.ones((len(unodes), np.max(times)+1), dtype='int8') * -1
for iunode, unode in enumerate(unodes):
    wh = np.where(nodes==unode)[0]
    for iwh in range(len(wh)):
        ind = times[wh[iwh],:]
        on_times[iunode,ind[0]:ind[1]+1] = hpsets[wh[iwh]]

# Create a figure and axes
fig = plt.figure(figsize=myfigsize)
ax = fig.gca()

# Plot the timing data
for ihpset in hpsets:
    node_ind, time_ind = np.where(on_times==ihpset)
    colors = ['C'+str(ihpset) for x in range(len(node_ind))]
    ax.scatter(time_ind/60, node_ind, c=colors)

# Set the figure properties
ax.set_xlabel('Elapsed time (min)')
ax.set_yticks(range(len(unodes)))
ax.set_yticklabels(unodes)
ax.set_ylabel('Compute node')
ax.set_title('Compute node usage')
fig.legend(hpsets2, loc='right')

# Save the figure
plt.savefig(figdir+'/'+'compute_node_usage.png')