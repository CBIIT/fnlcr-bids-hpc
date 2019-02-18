# These are "compound" functions used in the post-inference workflow. Simpler functions should be placed in the utils module.

def calculate_metrics(known_masks_list, inferred_masks_list):
    # Calculate the metrics (see calculate_metrics() in the utils module) for every model and inference direction 

    # Load relevant modules
    import numpy as np
    from . import utils

    metrics_2d_list = []
    metrics_3d_list = []
    for known_masks, inferred_masks in zip(known_masks_list,inferred_masks_list):

        # Variables
        nmodels = inferred_masks.shape[0]
        ninfdir = inferred_masks.shape[1]
        max_stack_size = max(known_masks.shape)
        nviews = ninfdir

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

        metrics_2d_list.append(metrics_2d)
        metrics_3d_list.append(metrics_3d)

    return(metrics_2d_list, metrics_3d_list)

def load_images(npy_file_list, dir='.'):
    # Load and normalize the images to uint8
    import numpy as np
    from . import utils
    images_list = []
    for npy_file in npy_file_list:
        images = np.load(dir+'/'+npy_file)
        images_list.append(utils.normalize_images(images,1))
    return(images_list)

def load_inferred_masks(roi_list, unpadded_shape_list, models, inference_directions, dir='.'):
    # Load and process the masks inferred by every model in every inference direction

    # Raw images correspond to what we're defining below as the (z,x,y) (stack, rows?, cols?) directions
    # iinfdir linearly (0,1,2) refers to the order of the input inference_directions, e.g., ['x','y','z'] or ['z']
    # This corresponds to the order of the second indices in inferred_masks, metrics_2d, and metrics_3d
    # The inference_direction letter A corresponds to the inferred_masks-A_first.npy file read in
    # dirs_index just helps us figure out how to undo the stack transpose for the current inference direction (it's purely an intermediate variable)
    # iview is simply the direction along which we're observing the data (whether in the metrics or the videos)
    # iview linearly (0,1,2) refers to the (z,x,y) directions, respectively, and corresponds to the fourth index of metrics_2d

    # Import relevant modules
    import numpy as np
    from . import utils

    # Constants
    reverse_transpose_indices = ((2,0,1),(1,2,0),(0,1,2)) # corresponds to x, y, z
    dirs = ['x','y','z']

    inferred_masks_list = []
    for roi, unpadded_shape in zip(roi_list,unpadded_shape_list):

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

        inferred_masks_list.append(inferred_masks)

    return(inferred_masks_list)

def output_metrics(metrics_3d_list, roi_list, models):

    # Import relevant modules
    import numpy as np
    from . import utils

    # Open files for writing and write the introductory HTML
    text_file = open('metrics_3d.txt','a')
    html_file = open('metrics_3d.html','a')
    html_file.write('<html><head><title>Table of 3D metrics</title></head><body><pre>\n')
    
    # For each pair of metrics_3d and roi...
    for metrics_3d, roi in zip(metrics_3d_list,roi_list):

        # Variables
        shp = metrics_3d.shape
        nmodels = shp[0]
        ninfdir = shp[1]
        nmetrics = shp[2]
        roi_num = int(roi.split('roi',1)[1])

        # Write the text file output
        for imodel in range(nmodels):
            model_num = int(models[imodel].split('-',1)[0])
            for iinfdir in range(ninfdir):
                string = str(model_num) + '\t' + str(iinfdir) + '\t' + str(roi_num) + '\t'
                for imetric in range(nmetrics):
                    string = string + str(metrics_3d[imodel,iinfdir,imetric])
                    if not imetric == (nmetrics-1):
                        string = string + '\t'
                    else:
                        string = string + '\n'
                text_file.write(string)

        # Write the HTML file output for the current ROI
        html_file.write('++++ ROI'+str(roi_num)+' ++++\n\n\n')
        html_file.write('                         |     TPR     |     TNR     |     PPV     |    BACC     |     F1      |\n')
        html_file.write('------------------------------------------------------------------------------------------------')
        for imodel in range(nmodels):
            html_file.write('\n '+'{:23}'.format(models[imodel])+' |')
            for imetric in range(nmetrics):
                string = ''
                for iinfdir in range(ninfdir):
                    score = np.round(metrics_3d[imodel,iinfdir,imetric]*100).astype('uint8')
                    score_str = utils.get_colored_str(score)
                    string = string + score_str
                if ninfdir == 3:
                    html_file.write(string+' |')
                else:
                    html_file.write('    '+string+'     |')
            string = string + '\n'
        html_file.write('\n\n\n\n')

    # Write closing HTML and close files
    html_file.write('</pre></body></html>\n')
    html_file.close()
    text_file.close()