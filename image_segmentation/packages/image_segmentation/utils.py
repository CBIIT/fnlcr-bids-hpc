# This module should contain relatively simple functions that can be used for many image segmentation-related purposes
# "Compound" functions that call the functions in this script (among other things) should be part of other modules in the image_segmentation package

def arr_info(arr):
    # Print information about a numpy array
    import numpy as np
    print('Shape: ',arr.shape)
    print('dtype: ',arr.dtype)
    print('Min: ',np.min(arr))
    print('Max: ',np.max(arr))
    hist,edges = np.histogram(arr)
    print('Hist: ',hist)
    print('Edges: ',edges)

def calculate_metrics(msk0,msk1,twoD_stack_dim=-1):
    # Calculate how well the image segmentation task performed by comparing the inferred masks (msk1) to the known masks (msk0)

    # Import relevant modules
    import numpy as np

    # Arrays required for calculations of metrics
    target = msk0.astype('bool')
    guess = msk1.astype('bool')
    overlap_fg = target & guess
    overlap_bg = (~target) & (~guess)

    # Process the arrays for the true number of foreground and background pixels and those the model gets correct
    nfg = count(target,twoD_stack_dim=twoD_stack_dim)
    nbg = count(~target,twoD_stack_dim=twoD_stack_dim)
    noverlap_fg = count(overlap_fg,twoD_stack_dim=twoD_stack_dim)
    noverlap_bg = count(overlap_bg,twoD_stack_dim=twoD_stack_dim)

    # Convert to true/false positives/negatives
    npos = nfg # purple + blue
    nneg = nbg # gray + red
    ntruepos = noverlap_fg # --> should be number of purples
    ntrueneg = noverlap_bg # --> should be number of grays
    nfalsepos = nneg - ntrueneg # --> should be number of reds (reason this makes sense: a false positive is really a negative that's not a true negative)
    nfalseneg = npos - ntruepos # --> should be number of blues (reason this makes sense: a false negative is really a positive that's not a true positive)

    # Metrics that should all be large
    tpr = ntruepos / (ntruepos + nfalseneg) # = ntruepos / npos # i.e., how good you are at detecting mito (sensitivity = recall = true positive rate)
    tnr = ntrueneg / (ntrueneg + nfalsepos) # = ntrueneg / nneg # i.e., how good you are at detecting NOT mito (specificity = selectivity = true negative rate)
    ppv = ntruepos / (ntruepos + nfalsepos) # positive predictive value = precision (i.e., of all the red that you see, how much of it is correct?)
    bacc = (tpr+tnr) / 2 # Overall accuracy (balanced accuracy)
    f1 = 2 / ( (1/tpr) + (1/ppv) )
    
    # Return a numpy array of the metrics
    return(np.array([tpr,tnr,ppv,bacc,f1]))
    
def count(arr,twoD_stack_dim=-1):
    # Sum an array over particular dimensions (over all dimensions by default)
    import numpy as np
    dims = [0,1,2]
    if twoD_stack_dim == -1:
        return(np.sum(arr))
    else:        
        dims.remove(twoD_stack_dim)
        return(np.sum(arr,axis=tuple(dims)))

def gray2rgba(img,A=255,mycolor=[1,1,1],makeBGTransp=False):
    # By default, make the image not transparent at all, make the transparency color white, and don't make the 0-pixels transparent
    import numpy as np
    mycolor = np.array(mycolor,dtype='float32')
    tmp = np.expand_dims(img,3)
    tmp = np.tile(tmp,(1,1,1,4))
    tmp[:,:,:,0] = (tmp[:,:,:,0]*mycolor[0]).astype('uint8')
    tmp[:,:,:,1] = (tmp[:,:,:,1]*mycolor[1]).astype('uint8')
    tmp[:,:,:,2] = (tmp[:,:,:,2]*mycolor[2]).astype('uint8')
    tmp[:,:,:,3] = A
    if makeBGTransp:
        bg0,bg1,bg2 = np.where(img==0)
        tmp[bg0,bg1,bg2,3] = 0
    return(tmp)

def normalize_images(images,idtype):
    # Automatically detect the pixel range and scale the images to a new range specified by idtype
    import numpy as np
    possible_range_maxs = [1,2**8-1,2**16-1]
    possible_range_dtypes = ['float32','uint8','uint16']
    median = np.median(images)
    found = False
    range_min = 0
    for range_max in possible_range_maxs:
        if (median>=range_min) & (median<=range_max):
            found = True
            break
        range_min = range_max
    if not found:
        print('No range found')
        print(median)
        return(-1)
    else:
        print('Range found; median is ',median,' and range max is ',range_max)
        images = ( images.astype('float32') / range_max * possible_range_maxs[idtype] ).astype(possible_range_dtypes[idtype])
        arr_info(images)
        return(images)
        
def pad_images(images, nlayers):
    """
    In Unet, every layer the dimension gets divided by 2
    in the encoder path. Therefore the image size should be divisible by 2^nlayers.
    """
    import math
    import numpy as np
    divisor = 2**nlayers
    nlayers, x, y = images.shape # essentially setting nlayers to z direction so return is z, x, y
    x_pad = int((math.ceil(x / float(divisor)) * divisor) - x)
    y_pad = int((math.ceil(y / float(divisor)) * divisor) - y)
    padded_image = np.pad(images, ((0,0),(0, x_pad), (0, y_pad)), 'constant', constant_values=(0, 0))
    return padded_image

def transpose_stack(images):
    # Split the images into all three dimensions, making the other two dimensions of the images accessible
    # Input images should be three dimensions (e.g., a stack)
    x_first = images.transpose((1,2,0)) # (x,y,z)
    y_first = images.transpose((2,0,1)) # (y,z,x)
    #z_first = images.transpose((0,1,2)) # (z,x,y)
    z_first = images
    return(x_first,y_first,z_first)