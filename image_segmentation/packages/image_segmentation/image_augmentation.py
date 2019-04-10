
# Note: This requires the imgaug package, which can be acquired on Github at https://github.com/aleju/imgaug
def augment_images(images, sequence, masks = None, n_outputs = None, normalize = None):
    """
    Augment a set of images based on a sequence 

    Arguments:
        images: 3d numpy_array
           The stack of input images to be augmented 

        masks: optional (same size of images)
           Stack of mask that corresponds ot the image

        sequence: imgaug seq 
            imgaug sequence to be applied

        n_outputs: int 
            The number of augmented images to generate, default is the number of input images

        normalize: int
            Augmentation has to be done on uint8 images.
            This value will normalize the image to 0:normalize.

    Returns
        img_aug, mask_aug 
            The augmented numpy array for the images and masks(if needed)
    """

    # Import relevant modules
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np
    from skimage import io
    import utils
    import math
    
    # Initialize the randomizer
    ia.seed(1)

    if normalize is not None:
        # ensure the images are uint8; inputs can therefore be [0,1], [0,2^8-1], or [0,2^16-1]
        images = utils.normalize_images(images, normalize) 

    # allow for flexible dimensions of input images and masks
    images, masks = utils.stack_and_color_images(images, masks=masks) 

    # Get the shapes of the images and masks
    N, H, W, C = images.shape

    # Determine if there are masks present
    if masks is not None:
        do_masks = True
    else:
        do_masks = False

    #sequence is already defined
    # Define the arrays to contain the stack for the current single augmentation
    image_aug = []
    if do_masks:
        mask_aug = []
        classes =np.max(masks) + 1
    else:
        mask_aug = None


    # For each image in the stack...

    if n_outputs is None: 
        n_outputs = N

    n_iter = int(math.ceil(n_outputs / N))


    utils.arr_info(images)
    utils.arr_info(masks)
    
    for iter in range(n_iter):
        for i in range(N):
            seq_det = sequence.to_deterministic()
            img_channels = []
            for ic in range(C):
                img_channels.append(seq_det.augment_image(images[i,:,:,ic].squeeze()))
            image_aug.append(img_channels)
            if do_masks:
                mask_segmap = ia.SegmentationMapOnImage(masks[i,:,:].squeeze(),shape=(H,W),nb_classes = classes)
                mask_aug.append(seq_det.augment_segmentation_maps([mask_segmap])[0].get_arr_int())
                
   
    
    # Convert the lists to numpy arrays and select n_outputs only
    image_aug = image_aug[0:n_outputs]
    image_aug = np.array(image_aug,dtype=np.uint8)
    print("BEFORE")
    utils.arr_info (mask_aug[0])
    if do_masks:
        mask_aug = mask_aug[0:n_outputs]
        print("mask_aug type:{})".format(type(mask_aug[0])))
        mask_aug = np.array(mask_aug,dtype=np.int16) # comes out as (N,H,W)

    print("AFTER")
    utils.arr_info (mask_aug)
    # Transpose the indices since they come out like (N,C,H,W)
    image_aug = np.transpose(image_aug,[0,2,3,1])

    return image_aug, mask_aug



def build_augmentation_seq(configuration):
    """
    Builds the augmentation pipeline 

    Arguments:
        configuration: file pointer
            The hitif configuration file 
    
    Returns:
       imgaug seq 
    """
    import configparser   
    from imgaug import augmenters as iaa
    
    config = configparser.ConfigParser()
    config.read(configuration)

    #Parse the augmentation parameters
    aug_prms = config['augmentation']
    
    CLAHE= eval(aug_prms['AllChannelsCLAHE'])
    impulse_noise = float(aug_prms['ImpulseNoise'])
    gaussian_blur = eval(aug_prms['GaussianBlur'])
    poisson = eval(aug_prms['AdditivePoissonNoise'])
    median = eval(aug_prms['MedianBlur'])

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        #iaa.AllChannelsCLAHE(CLAHE),
        iaa.OneOf([
            iaa.GaussianBlur(gaussian_blur),
            iaa.MedianBlur(median),
        ]),
        iaa.OneOf([
            iaa.AdditivePoissonNoise(poisson),
            iaa.ImpulseNoise(impulse_noise),
        ])
    ])

    return seq
