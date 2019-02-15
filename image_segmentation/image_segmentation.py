def preprocess_inference_images(images_npy_file, idtype, nlayers_max, prefix):

    # Load relevant modules
    import numpy as np
    import image_segmentation_utils as imseg_utils

    # Load the images
    images = np.load(images_npy_file)

    # Automatically scale the images
    images = imseg_utils.normalize_images(images,idtype)

    # Make the other two dimensions of the images accessible
    x_first,y_first,z_first = imseg_utils.transpose_stack(images)

    # Pad the each of 2D planes of the images
    x_first = imseg_utils.pad_images(x_first,nlayers_max)
    y_first = imseg_utils.pad_images(y_first,nlayers_max)
    z_first = imseg_utils.pad_images(z_first,nlayers_max)

    # Write these other "views" of the images to disk
    np.save(prefix+'-x_first.npy',x_first)
    np.save(prefix+'-y_first.npy',y_first)
    np.save(prefix+'-z_first.npy',z_first)