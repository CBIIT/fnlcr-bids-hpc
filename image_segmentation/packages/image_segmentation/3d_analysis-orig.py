def get_chunk_rand(x,Y):
    x_shp = np.array(x.shape)
    Y_shp = np.array(Y.shape)
    poss_starts = Y_shp - x_shp + 1
    starts = [random.randint(0,poss_starts[0]),random.randint(0,poss_starts[1]),random.randint(0,poss_starts[2])]
    ends = starts + x_shp
    return(Y[starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]])


def get_chunk_seq(x,Y):
    x_shp = np.array(x.shape)
    Y_shp = np.array(Y.shape)
    poss_starts = Y_shp - x_shp + 1
    mylist = []
    for i in range(poss_starts[0]):
        for j in range(poss_starts[1]):
            for k in range(poss_starts[2]):
                starts = [i,j,k]
                ends = starts + x_shp
                mylist = mylist + [Y[starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]]]
    return(mylist)


import numpy as np
from numpy import random

roi1_file = '/Users/weismanal/links/1-pre-processing/roi1/1-not_padded/roi1_images_gray_unpadded.npy'
#fov_z_file = '/Users/weismanal/links/1-pre-processing/field_of_view/field_of_view-normalized_for_unet-z_first.npy'
fov_z_file = '/Users/weismanal/links/1-pre-processing/field_of_view/field_of_view.npy'
spacing = 100

roi1 = np.load(roi1_file)
fovz = np.load(fov_z_file)

target = get_chunk_rand(roi1,fovz)

#roi1_shp = np.array(roi1.shape)
#fovz_shp = np.array(fovz.shape)
#tmp = fovz_shp - roi1_shp + 1
#starts = [random.randint(0,tmp[0]),random.randint(0,tmp[1]),random.randint(0,tmp[2])]
#ends = starts + roi1_shp
#target = fovz[starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]]

x = target[::spacing,::spacing,::spacing]
y = fovz[::spacing,::spacing,::spacing]

print(x.shape)
print(y.shape)

chunks = get_chunk_seq(x,y)
for chunk in chunks:
    if (x==chunk).all():
        print(chunk)

#print(chunks)


#ind0,ind1,ind2 = np.where(y==x[0,0,0])

#for (i0,i1,i2) in zip(ind0,ind1,ind2):
#    test = 
#    print(y[i0,i1,i2])


#print(len(ind0))
#print(len(ind1))
#print(len(ind2))
#print(roi1.shape)
#print(ends)

#print(np.histogram(roi1))
#print(np.histogram(fovz))

# ------------------------------------

def overlay_mask(images,masks,alpha,colors):
    import numpy as np
    myshape = images.shape # save its shape
    N = myshape[0] # save the stack size
    H = myshape[1] # save the image height
    W = myshape[2] # save the image width
    masks = masks.reshape((N,H*W)) # load the .npy file of the mask
    colors = colors.reshape((N,H*W)) # load the .npy file of the colors
    both = images.reshape((N,H*W,3)) # copy the images as "both"
    for i in range(N): # for each image in the stack...
        ind = np.nonzero(masks[i,:]!=0) # find where the mask is
        #both[i,ind,0] = both[i,ind,0]*alpha + 0*(1-alpha) # in the both array, make the mask location red
        #both[i,ind,1] = both[i,ind,1]*alpha + 0*(1-alpha)
        #both[i,ind,2] = both[i,ind,2]*alpha + 255*(1-alpha)
        both[i,ind,0] = both[i,ind,0]*alpha + colors[i,ind]*(1-alpha) # in the both array, make the mask location red
        both[i,ind,1] = both[i,ind,1]*alpha + colors[i,ind]*(1-alpha)
        #both[i,ind,2] = both[i,ind,2]*alpha + 255*(1-alpha)
        both[i,ind,2] = both[i,ind,2]*alpha + colors[i,ind]*(1-alpha)
    return(both.reshape((N,H,W,3))) # make the both array a reasonable shape

def randomize_labels(labels):
    import random
    import numpy as np
    random.seed(1)
    ind_labels = np.nonzero(labels!=0) # goes into labels; get the indices of labels that aren't background (=ind above)
    labels_nonzero = labels[ind_labels] # get the foreground labels
    labels2 = np.copy(labels) # duplicate the original labels
    y = np.unique(labels_nonzero) # same as above
    nx = y.size + 1 # same as above
    z = y + 10*nx # same as above
    random.shuffle(z) # shuffle the shifted unique labels (same as above)    
    #for ilabel in y: # for each foreground label... (ilabel=iy above)
    #print(nx)
    for i in np.arange(0,nx-1): # for indices of the unique labels excluding 0...
        #print(i)
        ilabel = y[i]
        iz = z[i]
        ind_labels_nonzero = np.nonzero(labels_nonzero==ilabel) # goes into labels_nonzero and ind_labels[X]; determine where the foreground labels equal the current unique foreground label
        ind0 = ind_labels[0][ind_labels_nonzero] # get the "x" indices of labels that are the current foreground value
        ind1 = ind_labels[1][ind_labels_nonzero] # get the "y" indices of labels that are the current foreground value
        ind2 = ind_labels[2][ind_labels_nonzero] # get the "z" indices of labels that are the current foreground value
        labels2[ind0,ind1,ind2] = iz
        #print(labels2[ind0,ind1,ind2]) # print the shifted and randomized foreground values that are the current unshifted, unrandomized foreground value
    labels2[ind_labels] = labels2[ind_labels]  - 10*nx    
    return(labels2)

def read_or_create_npy_file(fn,cmd='print("Warning: No command string entered")'):
    from pathlib import Path
    import numpy as np
    my_file = Path(fn)
    if my_file.exists():
        print('File ' + fn + ' exists; loading it')
        return(np.load(fn))
    else:
        print('File ' + fn + ' does not exist; creating it from "' + cmd + '"')
        arr = eval(cmd)
        np.save(fn,arr)
        return(arr)


# Import relevant modules
import numpy as np
from skimage import measure

# Parameters
#img_npy_file = '/Users/weismanal/notebook/2018-11-15/roi1_images_gray_unpadded.npy'
img_npy_file = '/Users/weismanal/notebook/2018-11-03/inferences/field_of_view.npy'
#msk_npy_file = '/Users/weismanal/notebook/2018-11-15/roi1_masks_unpadded.npy'
#focus = 'intersection'
focus = 'intersection'
#msk_npy_file = 'good_union_masks.npy'
#alpha = 0.8
alpha = 0
write_tif_file = False

# Variables
msk_npy_file = 'good_' + focus + '_masks.npy'
lbl_npy_file = 'good_' + focus + '_labels.npy'

# Load the data
img = np.load(img_npy_file).astype('uint8')
msk = np.load(msk_npy_file).astype('uint8')

# Make the image RGB and blue
img = np.tile(img.reshape(img.shape+(1,)),(1,1,1,3))
img[:,:,:,0] = 255

# Determine 3D labels/colors
#labels = measure.label(msk).astype('uint64')
#np.save('good_intersection_labels.npy',labels)
labels = read_or_create_npy_file(lbl_npy_file,"measure.label(msk).astype('uint64')") # if in the end e.g. max(labels)<2^64, do np.save('good_intersection_labels.npy',labels.astype('uint16'))
labels = randomize_labels(labels)
colors = np.copy(labels)
colors[labels!=0] -= 1
colors = colors / np.max(colors) * (2**8-1)
colors = colors.astype('uint8')

# Overlay the masks on the images
#overlay = overlay_mask(img,labels0,alpha,colors0)
overlay = overlay_mask(img,labels,alpha,colors)

if write_tif_file:
    import my_module as mm
    mm.numpy_array_to_tif_file(overlay,'good_'+focus+'_overlay.tif')

# ------------------------------------

# Plot the results

def make_axes_equal(ax,X,Y,Z):
    import numpy as np
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


#%matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np

myindex = 112
fig_width = 10 # note DPI seems to be 100
spacing = 100 # the lower the spacing, the larger the file, most likely
#angle_spacing = 10 # the lower the angle_spacing, the larger the movie
angle_spacing = 15

if True:
    # Plot the result
    fig1 = plt.figure(figsize=(fig_width,fig_width))
    ax1 = fig1.add_subplot(111)
    #plt.imshow(overlay[myindex,:,:,:])
    #plt.imshow(overlay[:,myindex,:,:])
    ax1.imshow(overlay[:,myindex,:,:])
    plt.show()

if False:
    z_axis = (0,1,2)
    x_axis = (1,2,0)
    y_axis = (2,0,1)
    ax_choice = y_axis
    
    fig2 = plt.figure(figsize=(fig_width,fig_width))
    ax2 = fig2.gca(projection='3d')
    ax2.set_aspect('equal')
    #fig2.axis('equal')
    ix,iy,iz = np.nonzero(labels.transpose(ax_choice)!=0)
    ix = ix[0::spacing]
    iy = iy[0::spacing]
    iz = iz[0::spacing]
    c = labels.transpose(ax_choice)[ix,iy,iz]
    ax2.scatter(ix,iy,iz,c=c-1,marker='.')
    make_axes_equal(ax2,ix,iy,iz)
    az = np.arange(360)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    for iaz in az[0::angle_spacing]:
        ax2.view_init(30,iaz)
        iaz_str = '{:03d}'.format(iaz)
        fn = 'movie/frame_'+iaz_str+'.png'
        plt.savefig(fn,dpi='figure')
        #plt.show()
    #plt.savefig('test.png',dpi='figure')

# -------------------------------------------

# Create good masks (both intersection and union) by "filling out" the regions that we know are true mitochondria based on the intersection filtering out the false positives
# First run this script (3)
# Then run the labeling script at the top of the Jupyter file (1) --> this takes a long time if the labels need to be created on the FOV
# Then run the plotting section (2)


# NEXT:
#   Examine the properties (e.g., volume) of the resulting mitochondria (3D)
#   Also look at the 2D overlays in Slicer


def fill_out_msk(dir_msk,int_msk):
    
    import numpy as np
    from skimage import measure

    dir_msk_good = np.zeros(dir_msk.shape).astype('uint8')
    
    for iplane in np.arange(dir_msk.shape[0]): # each plane is a 2D mask
        
        # Define the current 2D masks
        msk0 = int_msk[iplane,:,:] # this is the "sparse" mask
        msk1 = dir_msk[iplane,:,:] # this is the mask that should be pretty filled in

        # Get the patches in the "good" mask using skimage.measure.label
        labels = measure.label(msk1) # this assigns a nonzero patch ID to every foreground pixel in the good mask
        # Note that this will certainly include pancake pixels and mito-looking clouds

        # For each foreground pixel in the SPARSE mask, which should EXCLUDE pancake pixels and mito-looking clouds...
        px1,px2 = np.nonzero(msk0!=0)
        nnz = px1.size
        good_patches = []
        for inz in range(nnz): # each pixel will not overlap with pancakes/clouds and must correspond to a patch in "labels"
            ipx1 = px1[inz]
            ipx2 = px2[inz]
            good_patches.append(labels[ipx1,ipx2]) # record the corresponding patch in labels
        good_patches = np.unique(good_patches) # in general this should be a subset of all the patches in labels, check this!
        
        # Copy just the valid patches over to the current result
        for patch in good_patches:
            px1,px2 = np.nonzero(labels==patch) # get the pixels corresponding to the current patch
            dir_msk_good[iplane,px1,px2] = 1 # label the good patches all with 1
            
    return(dir_msk_good)


import numpy as np

# Parameter
num = 1

# Inputs (inferences on the FOV)
#x_msk, y_msk, z_msk
z_msk = np.load(str(num)+'-z_mask.npy')
nz = z_msk.shape[0]
nx = z_msk.shape[1]
ny = z_msk.shape[2]
x_msk = np.load(str(num)+'-x_mask.npy').transpose((1,2,0))[0:nz,:,:]
y_msk = np.load(str(num)+'-y_mask.npy').transpose((2,0,1))[0:nz,:,:]

# Take intersection to eliminate pancake structures and regions that look like mitochondria in the current direction
int_msk = x_msk & y_msk & z_msk # this will be somewhat sparse but its centers should indicate where real mitochondria are
    
# Go down z axis plane by plane, and repeat going down the x and y axes
z_msk_good = fill_out_msk(z_msk,int_msk)
x_msk_good = fill_out_msk(x_msk.transpose((1,2,0)),int_msk.transpose((1,2,0))).transpose((2,0,1))
y_msk_good = fill_out_msk(y_msk.transpose((2,0,1)),int_msk.transpose((2,0,1))).transpose((1,2,0))

# Explore what the resulting intersection and union look like; these are our results
#int_good = x_msk_good & y_msk_good & z_msk_good
#union_good = x_msk_good | y_msk_good | z_msk_good

# Save the results
#np.save('good_intersection_masks.npy',int_good)
int_good = read_or_create_npy_file('good_intersection_masks.npy',"x_msk_good & y_msk_good & z_msk_good")
#np.save('good_union_masks.npy',union_good)
union_good = read_or_create_npy_file('good_union_masks.npy',"x_msk_good | y_msk_good | z_msk_good")

# ---------------------------------------

def read_or_create_npy_file(fn,cmd='print("Warning: No command string entered")'):
    from pathlib import Path
    import numpy as np
    my_file = Path(fn)
    if my_file.exists():
        print('File ' + fn + ' exists; loading it')
        return(np.load(fn))
    else:
        print('File ' + fn + ' does not exist; creating it from "' + cmd + '"')
        arr = eval(cmd)
        np.save(fn,arr)
        return(arr)

arr = read_or_create_npy_file('dude6.npy')
#arr.shape