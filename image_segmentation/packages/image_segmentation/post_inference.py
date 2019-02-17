# These are "compound" functions used in the post-inference workflow. Simpler functions should be placed in the utils module.

def calculate_metrics(known_masks, inferred_masks, nviews):
    # Calculate the metrics (see calculate_metrics() in the utils module) for every model and inference direction 

    # Load relevant modules
    import numpy as np
    from . import utils

    # Variables
    nmodels = inferred_masks.shape[0]
    ninfdir = inferred_masks.shape[1]
    max_stack_size = max(known_masks.shape)

    # Constant
    nmetrics = 5

    # Define the metric arrays
    metrics_2d = np.zeros((nmodels,ninfdir,nmetrics,nviews,max_stack_size))
    metrics_3d = np.zeros((nmodels,ninfdir,nmetrics))

    # For each model...
    for imodel in range(nmodels):

        # For each inference direction...
        for iinfdir in range(ninfdir):

            inferred_masks_single = np.squeeze(inferred_masks[imodel,iinfdir,:,:,:]) # get the current set of inferred masks

            metrics_3d[imodel,iinfdir,:] = utils.calculate_metrics(known_masks,inferred_masks_single) # get the current 3D metrics

            # For each view...
            for iview in range(nviews):
                # Get the current 2D metrics
                curr_stack_size = known_masks.shape[iview]
                metrics_2d[imodel,iinfdir,:,iview,0:curr_stack_size] = utils.calculate_metrics(known_masks,inferred_masks_single,twoD_stack_dim=iview)

    return(metrics_2d, metrics_3d)

def load_images(npy_file, dir='.'):
    # Load and normalize the images to uint8
    import numpy as np
    from . import utils
    images = np.load(dir+'/'+npy_file)
    images = utils.normalize_images(images,1)
    return(images)

def load_inferred_masks(roi, unpadded_shape, models, inference_directions, dir='.'):
    # Load and process the masks inferred by every model in every inference direction

    # Import relevant modules
    import numpy as np
    from . import utils

    # Constants
    reverse_transpose_indices = ((2,0,1),(1,2,0),(0,1,2))
    dirs = ['x','y','z']

    # Load, undo stack transpose on, round, remove padding from, and normalize the inferred masks
    inferred_masks = np.zeros(tuple([len(models),len(inference_directions)]+list(unpadded_shape)))
    imodel = 0
    for model in models:
        iinfdir = 0
        for inference_direction in inference_directions:
            dir_index = dirs.index(inference_direction)
            msks = np.load(dir+'/'+model+'/inferred_masks-'+roi+'-'+inference_direction+'_first.npy')
            msks = msks.transpose(reverse_transpose_indices[dir_index])
            msks = np.round(msks)
            msks = msks[:unpadded_shape[0],:unpadded_shape[1],:unpadded_shape[2]]
            inferred_masks[imodel,iinfdir,:,:,:] = utils.normalize_images(msks,1)
            iinfdir += 1
        imodel += 1

    return(inferred_masks)