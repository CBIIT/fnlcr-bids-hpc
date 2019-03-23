# Set up paths and imports
import sys, os
image_seg_path = os.environ['IMAGE_SEG']
imgaug_path = os.environ['IMGAUG']
sys.path.append(image_seg_path)
from image_segmentation.image_augmentation import augment_images
import numpy as np
from skimage import io

# Define a function to create paths in the working directory
def create_dir(path):
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)

# Load the sample images and masks
image = io.imread(image_seg_path+'/'+'image_segmentation/lady.jpg') # (H,W,3)
images = np.load(image_seg_path+'/'+'image_segmentation/lady_images_rgb_original_15.npy') # (N,H,W,3)
images_gray = images[:,:,:,0] # (N,H,W)
masks = np.load(image_seg_path+'/'+'image_segmentation/lady_masks_original_15.npy') # (N,H,W)

# # Augment a single color image only
# ex_dir = './color_image'
# create_dir(ex_dir)
# augment_images(image, masks=None, do_composite=False, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=False allows you to see the individual augmentations
# augment_images(image, masks=None, do_composite=True, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=True does all the augmentations together ("composite")

# # Augment a stack of color images only
# ex_dir = './color_images'
# create_dir(ex_dir)
# augment_images(images, masks=None, do_composite=False, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=False allows you to see the individual augmentations
# augment_images(images, masks=None, do_composite=True, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=True does all the augmentations together ("composite")

# # Augment a stack of grayscale images only
# ex_dir = './grayscale_images'
# create_dir(ex_dir)
# augment_images(images_gray, masks=None, do_composite=False, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=False allows you to see the individual augmentations
# augment_images(images_gray, masks=None, do_composite=True, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=True does all the augmentations together ("composite")

# # Augment a stack of color images and masks
# ex_dir = './color_images_and_masks'
# create_dir(ex_dir)
# augment_images(images, masks=masks, do_composite=False, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=False allows you to see the individual augmentations
# augment_images(images, masks=masks, do_composite=True, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=True does all the augmentations together ("composite")

# # Augment a stack of grayscale images and masks
# ex_dir = './grayscale_images_and_masks'
# create_dir(ex_dir)
# augment_images(images_gray, masks=masks, do_composite=False, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=False allows you to see the individual augmentations
# augment_images(images_gray, masks=masks, do_composite=True, imgaug_repo=imgaug_path, output_dir=ex_dir) # do_composite=True does all the augmentations together ("composite")

# Augment a single color multiple times
ex_dir = './george-single_input_image_multiple_augmentations'
create_dir(ex_dir)
image_aug = augment_images(image, num_aug=50, masks=None, do_composite=True, imgaug_repo=imgaug_path, output_dir=ex_dir, aug_params=None, composite_sequence=None, individual_seqs_and_outnames=None)
print(image_aug.shape)

ex_dir = './george-input_stack_multiple_augmentations'
create_dir(ex_dir)
images_aug, masks_aug = augment_images(images, num_aug=3, masks=masks, do_composite=True, imgaug_repo=imgaug_path, output_dir=ex_dir, aug_params=None, composite_sequence=None, individual_seqs_and_outnames=None)
print(images_aug.shape, masks_aug.shape)