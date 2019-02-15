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
    nlayers, x, y = images.shape
    #z, x, y = images.shape
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