First, pre-process the inference images, e.g., in Python:

    ```python
    # Load pertinent modules
    import numpy as np
    import bids-hpc-utils as bhu
    
    # Load images
    images = np.load('/home/weismanal/links/1-pre-processing/roi3/1-not_padded/roi3_images_original.npy')
    
    # Automatically scale the images
    images = bhu.normalize_images(images,2)

    # Make the other two dimensions of the images accessible
    x_first,y_first,z_first = bhu.transpose_stack(images)

    # Pad the each of 2D planes of the images
    x_first = bhu.pad_images(x_first,6)
    y_first = bhu.pad_images(y_first,6)
    z_first = bhu.pad_images(z_first,6)
    
    # Write these other "views" of the test data to disk
    np.save('inference_images/roi3_prepared-x_first.npy',x_first)
    np.save('inference_images/roi3_prepared-y_first.npy',y_first)
    np.save('inference_images/roi3_prepared-z_first.npy',z_first)
    ```

blah