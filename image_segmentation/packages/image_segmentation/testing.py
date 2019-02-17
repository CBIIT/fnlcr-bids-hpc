def load_data():

    # Load relevant modules
    from .post_inference import load_images, load_inferred_masks

    # Parameters
    roi = 'roi3'
    # The following list of models is created by running in Bash: tmp=$(find . -type d -iregex "\./[0-9][0-9]-hpset.*" | sort | awk -v FS="./" -v ORS="','" '{print $2}'); models="['${tmp:0:${#tmp}-2}]"; echo $models
    #models = ['01-hpset_10','02-hpset_11','03-hpset_16','04-hpset_17','05-hpset_21a','06-hpset_21b','07-hpset_21c','08-hpset_21d','09-hpset_22','10-hpset_23','11-hpset_28','12-hpset_30','13-hpset_32','14-hpset_33','15-hpset_34','16-hpset_last_good_unet','17-hpset_resnet']
    models = ['08-hpset_21d','09-hpset_22']
    inference_directions = ['x','y','z']

    # Load the data
    images = load_images(roi+'_input_img.npy','/home/weismanal/notebook/2019-02-13')
    known_masks = load_images('known_masks_'+roi+'.npy','/home/weismanal/notebook/2019-02-13')
    inferred_masks = load_inferred_masks(roi, images.shape, models, inference_directions, '/home/weismanal/notebook/2019-02-13')

    return(images, known_masks, inferred_masks)

def calculate_metrics():

    # Load relevant modules
    from .post_inference import calculate_metrics

    # Parameters
    nviews = 3

    # Load the data
    images, known_masks, inferred_masks = load_data()

    # Calculate the metrics
    metrics_2d, metrics_3d = calculate_metrics(known_masks,inferred_masks,nviews)

    # Print some three arrays, two of which should have zeros at the end
    import numpy as np
    print(np.squeeze(metrics_2d[0,2,0,:,:]))

    #return(metrics_2d, metrics_3d)