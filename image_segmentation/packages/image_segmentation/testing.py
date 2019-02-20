def load_data():

    from .post_inference import load_images, load_inferred_masks

    # npy_image_list = ['roi3_input_img.npy','roi3_input_img.npy']
    # npy_label_list = ['known_masks_roi3.npy','known_masks_roi3.npy']
    # roi_list = ['roi3','roi3']
    npy_image_list = ['roi3_input_img.npy']
    npy_label_list = ['known_masks_roi3.npy']
    roi_list = ['roi3']
    # The following list of models is created by running in Bash: tmp=$(find . -type d -iregex "\./[0-9][0-9]-hpset.*" | sort | awk -v FS="./" -v ORS="','" '{print $2}'); models="['${tmp:0:${#tmp}-2}]"; echo $models
    #models = ['01-hpset_10','02-hpset_11','03-hpset_16','04-hpset_17','05-hpset_21a','06-hpset_21b','07-hpset_21c','08-hpset_21d','09-hpset_22','10-hpset_23','11-hpset_28','12-hpset_30','13-hpset_32','14-hpset_33','15-hpset_34','16-hpset_last_good_unet','17-hpset_resnet']
    #models = ['08-hpset_21d','09-hpset_22']
    models = ['08-hpset_21d']
    inference_directions = ['x','y','z']
    data_dir = '/home/weismanal/notebook/2019-02-13'

    images_list = load_images(npy_image_list,data_dir)
    known_masks_list = load_images(npy_label_list,data_dir)

    unpadded_shape_list = []
    for image in images_list:
        unpadded_shape_list.append(image.shape)
    inferred_masks_list = load_inferred_masks(roi_list, unpadded_shape_list, models, inference_directions, data_dir)

    return(images_list, known_masks_list, inferred_masks_list, roi_list, models)

def calculate_metrics():

    from .post_inference import calculate_metrics

    __, known_masks_list, inferred_masks_list, roi_list, models = load_data()

    metrics_2d_list, metrics_3d_list = calculate_metrics(known_masks_list,inferred_masks_list)

    import numpy as np
    print(np.squeeze(metrics_2d_list[0][0,2,0,:,:]))

    return(metrics_2d_list, metrics_3d_list, roi_list, models)

def make_plots():

    from .post_inference import make_plots

    images_list, known_masks_list, inferred_masks_list, __, models = load_data()

    metrics_2d_list, metrics_3d_list, __, __ = calculate_metrics()
    
    make_plots('andrew', images_list, inferred_masks_list, models, nframes=40, known_masks_list=known_masks_list, metrics_2d_list=metrics_2d_list, metrics_3d_list=metrics_3d_list)

def output_metrics():

    from .post_inference import output_metrics

    # roi_list = ['roi3','roi3']
    # models = ['08-hpset_21d','09-hpset_22']
    
    __, metrics_3d_list, roi_list, models = calculate_metrics()

    output_metrics(metrics_3d_list,roi_list,models)