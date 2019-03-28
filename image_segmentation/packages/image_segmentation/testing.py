def load_data():
    # Loads data specifically for testing; implicitly tests load_inferred_masks() function in post_inference module

    from .post_inference import load_inferred_masks
    from . import utils
    import numpy as np

    #roi_list = ['roi3','roi3']
    roi_list = ['roi3']
    # The following list of models is created by running in Bash: tmp=$(find . -type d -iregex "\./[0-9][0-9]-hpset.*" | sort | awk -v FS="./" -v ORS="','" '{print $2}'); models="['${tmp:0:${#tmp}-2}]"; echo $models
    #models = ['01-hpset_10','02-hpset_11','03-hpset_16','04-hpset_17','05-hpset_21a','06-hpset_21b','07-hpset_21c','08-hpset_21d','09-hpset_22','10-hpset_23','11-hpset_28','12-hpset_30','13-hpset_32','14-hpset_33','15-hpset_34','16-hpset_last_good_unet','17-hpset_resnet']
    #models = ['08-hpset_21d','09-hpset_22']
    models = ['08-hpset_21d']
    inference_directions = ['x','y','z']
    data_dir = '/home/weismanal/notebook/2019-02-13'

    images_list = []
    known_masks_list = []
    inferred_masks_list = []
    for roi in roi_list:
        images_list.append(utils.normalize_images(np.load(data_dir+'/'+roi+'_input_img.npy'), 1, do_output=False)) # normalize to uint8
        known_masks_list.append(utils.normalize_images(np.load(data_dir+'/'+'known_masks_'+roi+'.npy'), 1, do_output=False)) # normalize to uint8
        inferred_masks_list.append(load_inferred_masks(roi, images_list[-1].shape, models, inference_directions, data_dir, do_output=False)) # these are ultimately uint8

    # for images, known_masks, inferred_masks, roi, model in zip(images_list, known_masks_list, inferred_masks_list, roi_list, models):
    #     utils.arr_info(images)
    #     utils.arr_info(known_masks)
    #     utils.arr_info(inferred_masks)

    return(images_list, known_masks_list, inferred_masks_list, roi_list, models)

def testing__image_augmentation__augment_images():

    # See also image_augmentation_example.py and instructions on running it at https://cbiit.github.io/fnlcr-bids-hpc/image_segmentation/packages/image_segmentation/

    from .image_augmentation import augment_images
    from skimage import io
    import numpy as np

    # (H,W,3)
    # image = io.imread('/Users/weismanal/links/local/1-pre-processing/lady/lady.jpg')
    # # print(type(augment_images(image, do_composite=False, output_dir='/Users/weismanal/notebook/2019-03-23/output')))
    # # print(type(augment_images(image, num_aug=5, do_composite=False, output_dir='/Users/weismanal/notebook/2019-03-23/output')))
    # print(type(augment_images(image, num_aug=1, do_composite=True, output_dir='/Users/weismanal/notebook/2019-03-23/output')))
    # # print(type(augment_images(image, num_aug=25, do_composite=True, output_dir='/Users/weismanal/notebook/2019-03-23/output')))


    # (N,H,W,3)
    # images = np.load('/Users/weismanal/links/local/1-pre-processing/lady/lady_images_rgb_original_15.npy')
    # #augment_images(images, do_composite=False, output_dir='/Users/weismanal/notebook/2019-03-23/output')
    # #augment_images(images, num_aug=2, do_composite=False, output_dir='/Users/weismanal/notebook/2019-03-23/output')
    # augment_images(images, num_aug=1, do_composite=True, output_dir='/Users/weismanal/notebook/2019-03-23/output')

    # (N,H,W,3) images and masks
    # images = np.load('/Users/weismanal/links/local/1-pre-processing/lady/lady_images_rgb_original_15.npy')
    # masks = np.load('/Users/weismanal/links/local/1-pre-processing/lady/lady_masks_original_15.npy')
    # augment_images(images, masks=masks, do_composite=True, output_dir='/Users/weismanal/notebook/2019-03-23/output')

    # (N,H,W) images and masks
    images = np.load('/Users/weismanal/links/local/1-pre-processing/lady/lady_images_rgb_original_15.npy')[:,:,:,0]
    masks = np.load('/Users/weismanal/links/local/1-pre-processing/lady/lady_masks_original_15.npy')
    augment_images(images, masks=masks, do_composite=True, output_dir='/Users/weismanal/notebook/2019-03-23/output')

def testing__post_inference__calculate_metrics():

    from .post_inference import calculate_metrics

    __, known_masks_list, inferred_masks_list, __, __ = load_data()

    metrics_2d_list, metrics_3d_list = calculate_metrics(known_masks_list, inferred_masks_list)

    import numpy as np
    print(np.squeeze(metrics_2d_list[0][0,2,0,:,:]))

    return(metrics_2d_list, metrics_3d_list)

def testing__post_inference__output_metrics():

    from .post_inference import output_metrics

    __, known_masks_list, inferred_masks_list, roi_list, models = load_data()

    from .post_inference import calculate_metrics
    __, metrics_3d_list = calculate_metrics(known_masks_list, inferred_masks_list)

    output_metrics(metrics_3d_list, roi_list, models)

def testing__post_inference__make_movies():

    from .post_inference import make_movies

    images_list, known_masks_list, inferred_masks_list, roi_list, models = load_data()

    from .post_inference import calculate_metrics
    metrics_2d_list, metrics_3d_list = calculate_metrics(known_masks_list, inferred_masks_list)
    
    make_movies(roi_list, images_list, inferred_masks_list, models, nframes=40, known_masks_list=known_masks_list, metrics_2d_list=metrics_2d_list, metrics_3d_list=metrics_3d_list)