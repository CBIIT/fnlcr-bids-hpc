# Note: This requires the imgaug package, which can be acquired on Github at https://github.com/aleju/imgaug
def augment_images(images, masks=None, do_composite=True, imgaug_repo='/Users/weismanal/checkouts/imgaug', output_dir='.'):

    # Since imgaug is a GitHub repo, we need to point to where the clone is so we can import the library
    import sys
    sys.path.append(imgaug_repo)

    # Import relevant modules
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np
    from skimage import io
    from . import utils
    
    # Augmentation parameters; these are nice set of settings that worked well for Kedar's data
    flip_factor = 0.5
    add_vals = (-30,30)
    multiply_factors = (0.75,1.25)
    gaussian_blur_sigma = (0,0.75)
    average_blur_pixels = (1,2)
    median_blur_pixels = (1,3)
    gaussian_noise_vals = (0,0.035*255)
    contrast_normalization_factors = (0.75,1.25)
    rotation_degrees = (-90,90)
    scale_factors = (0.8,1.2)
    
    # Other parameter
    nsingle_aug = 13 # This should be set to the number of individual augmentations below, i.e., the number in the last 'elif aug_type ==' line below.
    
    # Initialize the randomizer
    ia.seed(1)

    # Preprocess the images and masks (note that masks in imgaug are assumed to be integer-based, bool, or float-based; we're settling on int16)
    images = utils.normalize_images(images,1) # ensure the images are uint8; inputs can therefore be [0,1], [0,2^8-1], or [0,2^16-1]
    images, masks = utils.stack_and_color_images(images, masks=masks) # allow for flexible dimensions of input images and masks

    # Get the shapes of the images and masks
    N, H, W, C = images.shape

    # If we want to do each augmentation individually...
    if not(do_composite):

        # For each augmentation...
        for aug_type in range(1,nsingle_aug+1):

            # Define the single sequence
            if aug_type == 1:
                seq = iaa.Sequential(iaa.Flipud(1))
                outname = 'flipud'
            elif aug_type == 2:
                seq = iaa.Sequential(iaa.Fliplr(1))
                outname = 'fliplr'
            elif aug_type == 3:
                seq = iaa.Sequential(iaa.Add((add_vals[0],add_vals[0])))
                outname = 'add_low'
            elif aug_type == 4:
                seq = iaa.Sequential(iaa.Add((add_vals[1],add_vals[1])))
                outname = 'add_high'
            elif aug_type == 5:
                seq = iaa.Sequential(iaa.Multiply((multiply_factors[0],multiply_factors[0])))
                outname = 'multiply_low'
            elif aug_type == 6:
                seq = iaa.Sequential(iaa.Multiply((multiply_factors[1],multiply_factors[1])))
                outname = 'multiply_high'
            elif aug_type == 7:
                seq = iaa.Sequential(iaa.GaussianBlur(sigma=(gaussian_blur_sigma[1],gaussian_blur_sigma[1])))
                outname = 'gaussian_blur'
            elif aug_type == 8:
                seq = iaa.Sequential(iaa.AverageBlur(k=(average_blur_pixels[1],average_blur_pixels[1])))
                outname = 'average_blur'
            elif aug_type == 9:
                seq = iaa.Sequential(iaa.MedianBlur(k=(median_blur_pixels[1],median_blur_pixels[1])))
                outname = 'median_blur'
            elif aug_type == 10:
                seq = iaa.Sequential(iaa.AdditiveGaussianNoise(loc=0,scale=(gaussian_noise_vals[1],gaussian_noise_vals[1])))
                outname = 'additive_gaussian_noise'
            elif aug_type == 11:
                seq = iaa.Sequential(iaa.ContrastNormalization((contrast_normalization_factors[0],contrast_normalization_factors[0])))
                outname = 'contrast_norm_low'
            elif aug_type == 12:
                seq = iaa.Sequential(iaa.ContrastNormalization((contrast_normalization_factors[1],contrast_normalization_factors[1])))
                outname = 'contrast_norm_high'
            elif aug_type == 13:
                seq = iaa.Sequential(iaa.Affine(rotate=rotation_degrees,scale=scale_factors))
                outname = 'affine'
            outname = '{:02d}'.format(aug_type) + '-' + outname

            # Define the array to contain the stack for the current single augmentation
            image_aug = []

            # For each image in the stack...
            for i in range(N):
                seq_det = seq.to_deterministic()
                img_channels = []
                for ic in range(C):
                    img_channels.append(seq_det.augment_image(images[i,:,:,ic].squeeze()))
                image_aug.append(img_channels)

            # Convert the lists to numpy arrays
            image_aug = np.array(image_aug,dtype=np.uint8)

            # Transpose the indices since they come out like (N,C,H,W)
            image_aug = np.transpose(image_aug,[0,2,3,1])

            # Save the outputs as .tif files so you can check the individual augmentations
            io.imsave(output_dir+'/'+outname+'.tif',image_aug)
        
    else:

        # Determine if there are masks present
        if masks is not None:
            do_masks = True
        else:
            do_masks = False
    
        # Define the composite sequence
        seq = iaa.Sequential([
            iaa.Flipud(flip_factor),
            iaa.Fliplr(flip_factor),
            iaa.Sometimes(0.5,
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=gaussian_blur_sigma),
                    #iaa.AverageBlur(k=average_blur_pixels), # There at least used to be a weird shift for this augmentation so I removed it
                    iaa.MedianBlur(k=median_blur_pixels)
                ])),
            iaa.ContrastNormalization(contrast_normalization_factors),
            iaa.AdditiveGaussianNoise(loc=0,scale=(gaussian_noise_vals)),
            iaa.OneOf([
                iaa.Add(add_vals),
                iaa.Multiply(multiply_factors)
            ]),
            iaa.Affine(
                rotate=rotation_degrees,
                scale=scale_factors
            )
        ])

        # Define the arrays to contain the stack for the current single augmentation
        image_aug = []
        if do_masks:
            mask_aug = []

        # For each image in the stack...
        for i in range(N):
            seq_det = seq.to_deterministic()
            img_channels = []
            for ic in range(C):
                img_channels.append(seq_det.augment_image(images[i,:,:,ic].squeeze()))
            image_aug.append(img_channels)
            if do_masks:
                mask_segmap = ia.SegmentationMapOnImage(masks[i,:,:].squeeze(),shape=(H,W),nb_classes=2)
                mask_aug.append(seq_det.augment_segmentation_maps([mask_segmap])[0].get_arr_int())

        # Convert the lists to numpy arrays
        image_aug = np.array(image_aug,dtype=np.uint8)
        if do_masks:
            mask_aug = np.array(mask_aug,dtype=np.int16) # comes out as (N,H,W)

        # Transpose the indices since they come out like (N,C,H,W)
        image_aug = np.transpose(image_aug,[0,2,3,1])

        # Overlay the masks on the images and save as .tif
        if do_masks:
            rgba1 = utils.arr2rgba(image_aug,A=255,mycolor=[1,1,1],makeBGTransp=False)
            rgba2 = utils.arr2rgba(mask_aug,A=round(0.25*255),mycolor=[1,0,0],makeBGTransp=True)
            io.imsave(output_dir+'/'+'overlay.tif',utils.overlay_images(rgba1, rgba2))

        # Save the outputs as .tif files
        io.imsave(output_dir+'/'+'composite.tif',image_aug)
        
        # Save the outputs as .npy files
        np.save(output_dir+'/'+'composite.npy',image_aug)
        if do_masks:
            np.save(output_dir+'/'+'mask-composite.npy',mask_aug)

    # Save the input image in the same format (.tif) as the augmented images for comparison purposes
    io.imsave(output_dir+'/'+'input.tif',images)
    if masks is not None:
        rgba1 = utils.arr2rgba(images,A=255,mycolor=[1,1,1],makeBGTransp=False)
        rgba2 = utils.arr2rgba(masks,A=round(0.25*255),mycolor=[1,0,0],makeBGTransp=True)
        io.imsave(output_dir+'/'+'overlay-input.tif',utils.overlay_images(rgba1, rgba2))
