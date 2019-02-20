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

def make_plots(roi_name, images_list, inferred_masks_list, models, nframes=40, known_masks_list=None, metrics_2d_list=None, metrics_3d_list=None):
    # Note that roi_name is only used for creating the images filenames and need not correspond to any actual names

    # Import relevant modules
    import numpy as np
    import matplotlib.pyplot as plt
    from . import utils
    import os

    # Constants
    transpose_indices = ((0,1,2),(1,2,0),(2,0,1)) # corresponds to z, x, y as for other iview-dependent variables
    legend_arr = ['true positive rate / sensitivity / recall','true negative rate / specificity / selectivity','positive predictive value / precision','balanced accuracy','f1 score']
    do_transpose_for_view = [False,True,True] # z, x, y, as normal for views

    # Create the figures directory if it doesn't already exist
    if not os.path.exists('figures'):
        os.mkdir('figures')

    # Determine whether we have a situation in which the true masks (and therefore metrics) are known
    if known_masks_list is None:
        truth_known = False
    else:
        truth_known = True

    # For every set of images, inferred masks, and, if applicable, known masks and metrics...
    ilist = 0
    for images, inferred_masks in zip(images_list,inferred_masks_list):
        if truth_known:
            known_masks = known_masks_list[ilist]
            metrics_2d = metrics_2d_list[ilist]
            metrics_3d = metrics_3d_list[ilist]
            nmetrics = metrics_3d.shape[2]

        # Set some variables needed later
        shp = inferred_masks.shape
        nmodels = shp[0]
        ninfdir = shp[1]
        unpadded_shape = shp[2:]
        nviews = ninfdir

        # For every model...
        for imodel in range(nmodels):

            print('On model '+str(imodel+1)+' of '+str(nmodels)+' ('+models[imodel]+')')

            # For every view...
            for iview in range(nviews):

                print('  On view '+str(iview+1)+' of '+str(nviews))

                # Clear the figure if it exists
                plt.clf()

                # Get the current data
                curr_stack_size = unpadded_shape[iview]
                curr_images = images.transpose(transpose_indices[iview])
                if truth_known:
                    curr_known_masks = known_masks.transpose(transpose_indices[iview])
                if ninfdir == 3:
                    curr_inferred_masks = [np.squeeze(inferred_masks[imodel,0,:,:,:]).transpose(transpose_indices[iview]), np.squeeze(inferred_masks[imodel,1,:,:,:]).transpose(transpose_indices[iview]), np.squeeze(inferred_masks[imodel,2,:,:,:]).transpose(transpose_indices[iview])]
                    labels_infdirs = ['X','Y','Z']
                    labels_views = ['Z','X','Y']
                    if truth_known:
                        curr_metrics_2d = [np.squeeze(metrics_2d[imodel,0,:,iview,:curr_stack_size]), np.squeeze(metrics_2d[imodel,1,:,iview,:curr_stack_size]), np.squeeze(metrics_2d[imodel,2,:,iview,:curr_stack_size])]
                        curr_metrics_3d = [np.squeeze(metrics_3d[imodel,0,:]), np.squeeze(metrics_3d[imodel,1,:]), np.squeeze(metrics_3d[imodel,2,:])]
                else:
                    curr_inferred_masks = [np.squeeze(inferred_masks[imodel,0,:,:,:]).transpose(transpose_indices[iview])]
                    labels_infdirs = ['Z']
                    labels_views = ['Z']
                    if truth_known:
                        curr_metrics_2d = [np.squeeze(metrics_2d[imodel,0,:,iview,:curr_stack_size])]
                        curr_metrics_3d = [np.squeeze(metrics_3d[imodel,0,:])]

                # Determine the figure size (and correspondingly, the subplots size)
                if ninfdir == 3:
                    fig_width = 16 # inches
                    nsp_cols = 3 # sp = subplot
                else:
                    fig_width = 6
                    nsp_cols = 1
                if truth_known:
                    fig_height = 9
                    nsp_rows = 2
                else:
                    fig_height = 5
                    nsp_rows = 1

                # Set the figure size
                plt.figure(figsize=(fig_width,fig_height)) # interestingly you must initialize figsize here in order to make later calls to myfig.set_figwidth(X) work

                # Set the subplots size and get the axes handles
                if ninfdir == 3:
                    axes_images = [plt.subplot(nsp_rows,nsp_cols,1), plt.subplot(nsp_rows,nsp_cols,2), plt.subplot(nsp_rows,nsp_cols,3)]
                    if truth_known:
                        axes_metrics = [plt.subplot(nsp_rows,nsp_cols,ninfdir+1), plt.subplot(nsp_rows,nsp_cols,ninfdir+2), plt.subplot(nsp_rows,nsp_cols,ninfdir+3)]
                else:
                    axes_images = [plt.subplot(nsp_rows,nsp_cols,1)]
                    if truth_known:
                        axes_metrics = [plt.subplot(nsp_rows,nsp_cols,ninfdir+1)]
                
                # Frame-independent plotting
                iinfdir = 0
                for ax_images, label_infdirs in zip(axes_images,labels_infdirs):
                    ax_images.set_title('view='+labels_views[iview]+', infdir='+label_infdirs)
                    if truth_known:
                        ax_metrics = axes_metrics[iinfdir]
                        ax_metrics.plot(curr_metrics_2d[iinfdir].transpose())
                        ax_metrics.set_xlim(0,curr_stack_size-1)
                        ax_metrics.set_ylim(0,1)
                        ax_metrics.set_xlabel('3D stats: tpr='+'{:04.2f}'.format(curr_metrics_3d[iinfdir][0])+', tnr='+'{:04.2f}'.format(curr_metrics_3d[iinfdir][1])+', ppv='+'{:04.2f}'.format(curr_metrics_3d[iinfdir][2])+', bacc='+'{:04.2f}'.format(curr_metrics_3d[iinfdir][3])+', f1='+'{:04.2f}'.format(curr_metrics_3d[iinfdir][4]))
                        ax_metrics.set_ylabel(models[imodel])
                        ax_metrics.legend(legend_arr,loc='lower left')
                    iinfdir += 1

                # Now plot the frame-dependent data and metrics...for every frame...
                for frame in np.linspace(0,curr_stack_size-1,num=nframes).astype('uint16'):

                    print('    On frame '+str(frame+1)+' in '+str(curr_stack_size))

                    # Set variables that are the same for each inference direction: curr_images_frame, (curr_known_masks_frame)
                    curr_images_frame = prepare_for_plotting(curr_images[frame,:,:],do_transpose_for_view[iview],1,[1,1,1],False)
                    if truth_known:
                        curr_known_masks_frame = prepare_for_plotting(curr_known_masks[frame,:,:],do_transpose_for_view[iview],0.2,[0,0,1],True)

                    # Set variables that are different for each inference direction (ax_images, (ax_metrics), curr_inferred_masks_infdir_frame, (curr_metrics_2d_infdir_frame)) and do the plotting
                    iinfdir = 0
                    temporary_plots = []
                    for ax_images, curr_inferred_masks_infdir in zip(axes_images,curr_inferred_masks):
                        curr_inferred_masks_infdir_frame = prepare_for_plotting(curr_inferred_masks_infdir[frame,:,:],do_transpose_for_view[iview],0.2,[1,0,0],True)
                        temporary_plots.append(ax_images.imshow(curr_images_frame))
                        temporary_plots.append(ax_images.imshow(curr_inferred_masks_infdir_frame))
                        if truth_known:
                            ax_metrics = axes_metrics[iinfdir]
                            curr_metrics_2d_infdir_frame = np.squeeze(curr_metrics_2d[iinfdir][:,frame])
                            ax_metrics.set_title('tpr='+'{:04.2f}'.format(curr_metrics_2d_infdir_frame[0])+' tnr='+'{:04.2f}'.format(curr_metrics_2d_infdir_frame[1])+' ppv='+'{:04.2f}'.format(curr_metrics_2d_infdir_frame[2])+' bacc='+'{:04.2f}'.format(curr_metrics_2d_infdir_frame[3])+' f1='+'{:04.2f}'.format(curr_metrics_2d_infdir_frame[4]))
                            temporary_plots.append(ax_images.imshow(curr_known_masks_frame))
                            temporary_plots.append(ax_metrics.scatter(np.ones((nmetrics,1))*frame,curr_metrics_2d_infdir_frame,c=['C0','C1','C2','C3','C4']))
                        iinfdir += 1
                        # utils.arr_info(curr_images_frame[:,:,0:3])
                        # utils.arr_info(curr_images_frame[:,:,3])
                        # utils.arr_info(curr_known_masks_frame[:,:,0:3])
                        # utils.arr_info(curr_known_masks_frame[:,:,3])
                        # utils.arr_info(curr_inferred_masks_infdir_frame[:,:,0:3])
                        # utils.arr_info(curr_inferred_masks_infdir_frame[:,:,3])
                        # exit()

                    # Save the figure to disk
                    plt.savefig('figures/roi_'+roi_name+'__model_'+models[imodel]+'__view_'+labels_views[iview]+'__frame_'+'{:04d}'.format(frame)+'.png',dpi='figure')

                    # Delete temporary objects from the plot
                    for temporary_plot in temporary_plots:
                        temporary_plot.set_visible(False)
                    
        # Go to the next set of images etc. for the current ROI
        ilist += 1

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

def prepare_for_plotting(image, do_transpose, scale, mycolor, makeBGTransp):
    '''
    Transpose images for easier visualization (if required) and convert them to RGBA images
    '''
    from . import utils
    import numpy as np
    image = np.expand_dims(image,0)
    if do_transpose:
        image = image.transpose((0,2,1))
    image = utils.gray2rgba(image,A=scale*255,mycolor=mycolor,makeBGTransp=makeBGTransp)
    return(np.squeeze(image).astype('uint8'))