def overlay_mask(images,masks,alpha,colors):
    import numpy as np
    myshape = images.shape # save its shape
    N = myshape[0] # save the stack size
    H = myshape[1] # save the image height
    W = myshape[2] # save the image width
    masks = masks.reshape((N,H*W)) # load the .npy file of the mask
    colors = colors.reshape((N,H*W)) # load the .npy file of the colors
    both = images.reshape((N,H*W,3)) # copy the images as "both"
    for i in range(N): # for each image in the stack...
        ind = np.nonzero(masks[i,:]!=0) # find where the mask is
        #both[i,ind,0] = both[i,ind,0]*alpha + 0*(1-alpha) # in the both array, make the mask location red
        #both[i,ind,1] = both[i,ind,1]*alpha + 0*(1-alpha)
        #both[i,ind,2] = both[i,ind,2]*alpha + 255*(1-alpha)
        both[i,ind,0] = both[i,ind,0]*alpha + colors[i,ind]*(1-alpha) # in the both array, make the mask location red
        both[i,ind,1] = both[i,ind,1]*alpha + colors[i,ind]*(1-alpha)
        #both[i,ind,2] = both[i,ind,2]*alpha + 255*(1-alpha)
        both[i,ind,2] = both[i,ind,2]*alpha + colors[i,ind]*(1-alpha)
    return(both.reshape((N,H,W,3))) # make the both array a reasonable shape

def randomize_labels(labels):
    import random
    import numpy as np
    random.seed(1)
    ind_labels = np.nonzero(labels!=0) # goes into labels; get the indices of labels that aren't background (=ind above)
    labels_nonzero = labels[ind_labels] # get the foreground labels
    labels2 = np.copy(labels) # duplicate the original labels
    y = np.unique(labels_nonzero) # same as above
    nx = y.size + 1 # same as above
    z = y + 10*nx # same as above
    random.shuffle(z) # shuffle the shifted unique labels (same as above)    
    #for ilabel in y: # for each foreground label... (ilabel=iy above)
    #print(nx)
    for i in np.arange(0,nx-1): # for indices of the unique labels excluding 0...
        #print(i)
        ilabel = y[i]
        iz = z[i]
        ind_labels_nonzero = np.nonzero(labels_nonzero==ilabel) # goes into labels_nonzero and ind_labels[X]; determine where the foreground labels equal the current unique foreground label
        ind0 = ind_labels[0][ind_labels_nonzero] # get the "x" indices of labels that are the current foreground value
        ind1 = ind_labels[1][ind_labels_nonzero] # get the "y" indices of labels that are the current foreground value
        ind2 = ind_labels[2][ind_labels_nonzero] # get the "z" indices of labels that are the current foreground value
        labels2[ind0,ind1,ind2] = iz
        #print(labels2[ind0,ind1,ind2]) # print the shifted and randomized foreground values that are the current unshifted, unrandomized foreground value
    labels2[ind_labels] = labels2[ind_labels]  - 10*nx    
    return(labels2)

def make_axes_equal(ax,X,Y,Z):
    import numpy as np
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def fill_out_msk(dir_msk,int_msk):
    
    import numpy as np
    from skimage import measure

    dir_msk_good = np.zeros(dir_msk.shape).astype('uint8')
    
    for iplane in np.arange(dir_msk.shape[0]): # each plane is a 2D mask
        
        # Define the current 2D masks
        msk0 = int_msk[iplane,:,:] # this is the "sparse" mask
        msk1 = dir_msk[iplane,:,:] # this is the mask that should be pretty filled in

        # Get the patches in the "good" mask using skimage.measure.label
        labels = measure.label(msk1) # this assigns a nonzero patch ID to every foreground pixel in the good mask
        # Note that this will certainly include pancake pixels and mito-looking clouds

        # For each foreground pixel in the SPARSE mask, which should EXCLUDE pancake pixels and mito-looking clouds...
        px1,px2 = np.nonzero(msk0!=0)
        nnz = px1.size
        good_patches = []
        for inz in range(nnz): # each pixel will not overlap with pancakes/clouds and must correspond to a patch in "labels"
            ipx1 = px1[inz]
            ipx2 = px2[inz]
            good_patches.append(labels[ipx1,ipx2]) # record the corresponding patch in labels
        good_patches = np.unique(good_patches) # in general this should be a subset of all the patches in labels, check this!
        
        # Copy just the valid patches over to the current result
        for patch in good_patches:
            px1,px2 = np.nonzero(labels==patch) # get the pixels corresponding to the current patch
            dir_msk_good[iplane,px1,px2] = 1 # label the good patches all with 1
            
    return(dir_msk_good)

def read_or_create_npy_file(fn,cmd='print("Warning: No command string entered")'):
    from pathlib import Path
    import numpy as np
    my_file = Path(fn)
    if my_file.exists():
        print('File ' + fn + ' exists; loading it')
        return(np.load(fn))
    else:
        print('File ' + fn + ' does not exist; creating it from "' + cmd + '"')
        arr = eval(cmd)
        np.save(fn,arr)
        return(arr)

def get_new_masks(inferred_masks_list):

    # Import relevant modules
    from . import utils
    import numpy as np

    # Constant
    nnew_masks_per_seed = 10 # matches the number of hardcoded offset+ indices below
    copy_types = ['parallel', 'orth-union', 'orth-intersect'] # for the plane copy method

    # Variable to set in this function
    new_masks_list = []

    # For each set of inferred masks...
    for inferred_masks in inferred_masks_list:

        # Get the inferred masks shape, set nmodels correspondingly, and define the longer-term new variable we want to calculate, new_masks
        shp = inferred_masks.shape
        nmodels = shp[0]
        new_masks = np.zeros((nmodels,2*nnew_masks_per_seed,shp[2],shp[3],shp[4]),dtype='bool') # the 2 here corresponds to the number of items in the seeds list below

        # For each model...
        for imodel in range(nmodels):

            # Get the current inferred masks and cast them as Boolean
            masks = inferred_masks[imodel,:,:,:,:].astype('bool')

            # Calculate the corresponding sums, which can be thought of as a generalized intersection
            # sums = 0: none of the three inferences detects a mito; definitely not a mito
            # sums = 1: only one of the inferences detects a mito: probably not a mito
            # sums = 2: only one of the inferences does NOT detect a mito: probably a mito
            # sums = 3: all of the inferences detect a mito: definitely a mito; equals intersection (intersection = masks[0,:,:,:] & masks[1,:,:,:] & masks[2,:,:,:]; ((sums==3) == intersection).all()) is True
            # sums > 0: equals union (union = masks[0,:,:,:] | masks[1,:,:,:] | masks[2,:,:,:]; ((sums>0) == union).all()) is True
            sums = np.sum(masks, axis=0).astype('uint8')
            
            # Define the two types of seeds: (1) that in which the pixel probably corresponds to a mito, and (2) that in which the pixel definitely corresponds to a mito
            seeds = [np.where(sums>=2), np.where(sums==3)]

            # For each seed type...
            new_names = []
            for iseed, seed in enumerate(seeds):

                # Calculate the index offset
                offset = iseed * nnew_masks_per_seed
                seed_str = 'seed' + str(iseed+1)

                # Assign to the longer-term set of new masks the seed-only intermediate set of new masks
                new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                new_masks_one[seed] = True # seed also goes into sums
                new_masks[imodel,offset+0,:,:,:] = new_masks_one
                new_names.append(seed_str+'-'+'seed_only')

                # Run the plane copy method using three different settings to produce three different sets (of three) masks (the two "threes" here are independent of each other)
                new_masks_three_list = []
                for copy_type in copy_types:
                    new_masks_three_list.append(plane_copy(masks, seed, copy_type=copy_type)) # plane copy should send back an array of dimensions [3,X,Y,Z], just like inferred_masks[imodel,:,:,:,:] above

                # For each of three sets of plane copy results ("three"), combine them in three different ways to result in a single set of masks ("one")
                for offset_index, new_masks_three in enumerate(new_masks_three_list):

                    # Calculate the index offset
                    offset2 = offset_index * 3

                    # Start combining the plane copy results by summing as before
                    masks = new_masks_three.astype('bool')
                    sums = np.sum(masks, axis=0).astype('uint8')

                    # Assign to the longer-term set of new masks the current plane-copy method + union set of new masks
                    new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                    new_masks_one[np.where(sums>=1)] = True
                    new_masks[imodel,offset+offset2+1,:,:,:] = new_masks_one
                    new_names.append(seed_str+'-'+copy_types[offset_index]+'-'+'ge_1')

                    # Assign to the longer-term set of new masks the current plane-copy method + >=2 set of new masks
                    new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                    new_masks_one[np.where(sums>=2)] = True
                    new_masks[imodel,offset+offset2+2,:,:,:] = new_masks_one
                    new_names.append(seed_str+'-'+copy_types[offset_index]+'-'+'ge_2')

                    # Assign to the longer-term set of new masks the current plane-copy method + intersection set of new masks
                    new_masks_one = np.zeros((shp[2],shp[3],shp[4]),dtype='bool')
                    new_masks_one[np.where(sums==3)] = True
                    new_masks[imodel,offset+offset2+3,:,:,:] = new_masks_one
                    new_names.append(seed_str+'-'+copy_types[offset_index]+'-'+'eq_3')

        # Add this longer-term set of new masks to the set of new masks (a list) that we'll actually return
        new_masks_list.append(new_masks)

    # Return the new masks list and the names of the new aggregate masks
    return(new_masks_list, new_names)

def make_movies(roi_list, images_list, new_masks_list, models, nframes=40, known_masks_list=None, metrics_2d_list=None, metrics_3d_list=None, framerate=2, delete_frames=True, ffmpeg_preload_string='module load FFmpeg; ', new_names=None):
    # This is tested in its own function of the testing module

    # Import relevant modules
    import numpy as np
    import matplotlib.pyplot as plt
    from . import utils
    import os
    import glob

    # Constants
    transpose_indices = ((0,1,2),(1,2,0),(2,0,1)) # corresponds to z, x, y as for other iview-dependent variables
    legend_arr = ['true positive rate / sensitivity / recall','true negative rate / specificity / selectivity','positive predictive value / precision','balanced accuracy','f1 score']
    do_transpose_for_view = [False,True,True] # z, x, y, as normal for views

    # Create the movies directory if it doesn't already exist
    if not os.path.exists('movies'):
        os.mkdir('movies')

    # Determine whether we have a situation in which the true masks (and therefore metrics) are known
    if known_masks_list is None:
        truth_known = False
    else:
        truth_known = True

    # For every set of images, new masks, and, if applicable, known masks and metrics...
    ilist = 0
    for images, new_masks, roi in zip(images_list, new_masks_list, roi_list):
        if truth_known:
            known_masks = known_masks_list[ilist]
            metrics_2d = metrics_2d_list[ilist]
            metrics_3d = metrics_3d_list[ilist]
            nmetrics = metrics_3d.shape[2]

        # Set some variables needed later
        shp = new_masks.shape
        nmodels = shp[0]
        nnew_masks = shp[1]
        unpadded_shape = shp[2:]
        nviews = 3

        labels_views = ['Z','X','Y']

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

                for inew_mask in range(nnew_masks):
                    print('    On new mask '+str(inew_mask+1)+' of '+str(nnew_masks))
                
                    curr_new_masks = np.squeeze(new_masks[imodel,inew_mask,:,:,:]).transpose(transpose_indices[iview])
                    if truth_known:
                        curr_metrics_2d = np.squeeze(metrics_2d[imodel,inew_mask,:,iview,:curr_stack_size])
                        curr_metrics_3d = np.squeeze(metrics_3d[imodel,inew_mask,:])

                    # Determine the figure size (and correspondingly, the subplots size)
                    fig_width = 6 # inches
                    nsp_cols = 1 # sp = subplot
                    if truth_known:
                        fig_height = 9
                        nsp_rows = 2
                    else:
                        fig_height = 5
                        nsp_rows = 1

                    # Set the figure size
                    plt.figure(figsize=(fig_width,fig_height)) # interestingly you must initialize figsize here in order to make later calls to myfig.set_figwidth(X) work

                    # Set the subplots size and get the axes handles
                    ax_images = plt.subplot(nsp_rows,nsp_cols,1)
                    if truth_known:
                        ax_metrics = plt.subplot(nsp_rows,nsp_cols,1+1)
                    
                    # Frame-independent plotting
                    ax_images.set_title('view='+labels_views[iview])
                    new_mask_name = models[imodel].split('-',1)[0]+'-'+new_names[inew_mask]
                    if truth_known:
                        ax_metrics.plot(curr_metrics_2d.transpose())
                        ax_metrics.set_xlim(0,curr_stack_size-1)
                        ax_metrics.set_ylim(0,1)
                        ax_metrics.set_xlabel('3D stats: tpr='+'{:04.2f}'.format(curr_metrics_3d[0])+', tnr='+'{:04.2f}'.format(curr_metrics_3d[1])+', ppv='+'{:04.2f}'.format(curr_metrics_3d[2])+', bacc='+'{:04.2f}'.format(curr_metrics_3d[3])+', f1='+'{:04.2f}'.format(curr_metrics_3d[4]))
                        ax_metrics.set_ylabel(new_mask_name)
                        ax_metrics.legend(legend_arr,loc='lower left')
                  
                    # Determine if for the current view we should rotate the 2D plot by 90 degrees
                    if do_transpose_for_view[iview]:
                        rotate_2d = (1,0,2)
                    else:
                        rotate_2d = (0,1,2)

                    # Now plot the frame-dependent data and metrics...for every frame...
                    for frame in np.linspace(0,curr_stack_size-1,num=nframes).astype('uint16'):
                        print('    On frame '+str(frame+1)+' in '+str(curr_stack_size))

                        # Set variables that are the same for each inference direction: curr_images_frame, (curr_known_masks_frame)
                        curr_images_frame = np.transpose(np.squeeze(utils.arr2rgba(curr_images[frame,:,:],A=1*255,shade_color=[1,1,1],makeBGTransp=False)),rotate_2d)
                        if truth_known:
                            curr_known_masks_frame = np.transpose(np.squeeze(utils.arr2rgba(curr_known_masks[frame,:,:],A=0.2*255,shade_color=[0,0,1],makeBGTransp=True)),rotate_2d)

                        # Set variables that are different for each inference direction (ax_images, (ax_metrics), curr_new_masks_frame, (curr_metrics_2d_frame)) and do the plotting
                        temporary_plots = []
                        curr_new_masks_frame = np.transpose(np.squeeze(utils.arr2rgba(curr_new_masks[frame,:,:],A=0.2*255,shade_color=[1,0,0],makeBGTransp=True)),rotate_2d)
                        temporary_plots.append(ax_images.imshow(curr_images_frame))
                        temporary_plots.append(ax_images.imshow(curr_new_masks_frame))
                        if truth_known:
                            curr_metrics_2d_frame = np.squeeze(curr_metrics_2d[:,frame])
                            ax_metrics.set_title('tpr='+'{:04.2f}'.format(curr_metrics_2d_frame[0])+' tnr='+'{:04.2f}'.format(curr_metrics_2d_frame[1])+' ppv='+'{:04.2f}'.format(curr_metrics_2d_frame[2])+' bacc='+'{:04.2f}'.format(curr_metrics_2d_frame[3])+' f1='+'{:04.2f}'.format(curr_metrics_2d_frame[4]))
                            temporary_plots.append(ax_images.imshow(curr_known_masks_frame))
                            temporary_plots.append(ax_metrics.scatter(np.ones((nmetrics,1))*frame,curr_metrics_2d_frame,c=['C0','C1','C2','C3','C4']))

                        # Save the figure to disk
                        plt.savefig('movies/'+roi+'__model_'+new_mask_name+'__view_'+labels_views[iview]+'__frame_'+'{:04d}'.format(frame)+'.png',dpi='figure')

                        # Delete temporary objects from the plot
                        for temporary_plot in temporary_plots:
                            temporary_plot.set_visible(False)

                    # Determine the string that is a glob of all the frames
                    frame_glob = 'movies/'+roi+'__model_'+new_mask_name+'__view_'+labels_views[iview]+'__frame_'+'*'+'.png'

                    # Create the movies
                    os.system(ffmpeg_preload_string + 'ffmpeg -r '+str(framerate)+' -pattern_type glob -i "'+frame_glob+'" -c:v libx264 -crf 23 -profile:v baseline -level 3.0 -pix_fmt yuv420p -c:a aac -ac 2 -b:a 128k -movflags faststart ' + 'movies/'+roi+'__model_'+new_mask_name+'__view_'+labels_views[iview]+'.mp4')

                    # Unless otherwise specified, delete the frames
                    if delete_frames:
                        for frame in glob.glob(frame_glob):
                            os.remove(frame)
                    
        # Go to the next set of images etc. for the current ROI
        ilist += 1

def main_orig():

    #### Step 1 according to instructions
    # Create good masks (both intersection and union) by "filling out" the regions that we know are true mitochondria based on the intersection filtering out the false positives
    
    import numpy as np

    # Parameter
    num = 1

    # Inputs (inferences on the FOV)
    #x_msk, y_msk, z_msk
    z_msk = np.load(str(num)+'-z_mask.npy')
    nz = z_msk.shape[0]
    nx = z_msk.shape[1]
    ny = z_msk.shape[2]
    x_msk = np.load(str(num)+'-x_mask.npy').transpose((1,2,0))[0:nz,:,:]
    y_msk = np.load(str(num)+'-y_mask.npy').transpose((2,0,1))[0:nz,:,:]

    # Take intersection to eliminate pancake structures and regions that look like mitochondria in the current direction
    int_msk = x_msk & y_msk & z_msk # this will be somewhat sparse but its centers should indicate where real mitochondria are
        
    # Go down z axis plane by plane, and repeat going down the x and y axes
    z_msk_good = fill_out_msk(z_msk,int_msk)
    x_msk_good = fill_out_msk(x_msk.transpose((1,2,0)),int_msk.transpose((1,2,0))).transpose((2,0,1))
    y_msk_good = fill_out_msk(y_msk.transpose((2,0,1)),int_msk.transpose((2,0,1))).transpose((1,2,0))

    # Explore what the resulting intersection and union look like; these are our results
    #int_good = x_msk_good & y_msk_good & z_msk_good
    #union_good = x_msk_good | y_msk_good | z_msk_good

    # Save the results
    #np.save('good_intersection_masks.npy',int_good)
    int_good = read_or_create_npy_file('good_intersection_masks.npy',"x_msk_good & y_msk_good & z_msk_good")
    #np.save('good_union_masks.npy',union_good)
    union_good = read_or_create_npy_file('good_union_masks.npy',"x_msk_good | y_msk_good | z_msk_good")

    
    #### Step 2 according to instructions
    # Labeling (this takes a long time if the labels need to be created on the FOV)

    # Import relevant modules
    import numpy as np
    from skimage import measure

    # Parameters
    #img_npy_file = '/Users/weismanal/notebook/2018-11-15/roi1_images_gray_unpadded.npy'
    img_npy_file = '/Users/weismanal/notebook/2018-11-03/inferences/field_of_view.npy'
    #msk_npy_file = '/Users/weismanal/notebook/2018-11-15/roi1_masks_unpadded.npy'
    #focus = 'intersection'
    focus = 'intersection'
    #msk_npy_file = 'good_union_masks.npy'
    #alpha = 0.8
    alpha = 0
    write_tif_file = False

    # Variables
    msk_npy_file = 'good_' + focus + '_masks.npy'
    lbl_npy_file = 'good_' + focus + '_labels.npy'

    # Load the data
    img = np.load(img_npy_file).astype('uint8')
    msk = np.load(msk_npy_file).astype('uint8')

    # Make the image RGB and blue
    img = np.tile(img.reshape(img.shape+(1,)),(1,1,1,3))
    img[:,:,:,0] = 255

    # Determine 3D labels/colors
    #labels = measure.label(msk).astype('uint64')
    #np.save('good_intersection_labels.npy',labels)
    labels = read_or_create_npy_file(lbl_npy_file,"measure.label(msk).astype('uint64')") # if in the end e.g. max(labels)<2^64, do np.save('good_intersection_labels.npy',labels.astype('uint16'))
    labels = randomize_labels(labels)
    colors = np.copy(labels)
    colors[labels!=0] -= 1
    colors = colors / np.max(colors) * (2**8-1)
    colors = colors.astype('uint8')

    # Overlay the masks on the images
    #overlay = overlay_mask(img,labels0,alpha,colors0)
    overlay = overlay_mask(img,labels,alpha,colors)

    if write_tif_file:
        import my_module as mm
        mm.numpy_array_to_tif_file(overlay,'good_'+focus+'_overlay.tif')

    
    #### Step 3 according to instructions
    # Plot the results

    #%matplotlib
    #%matplotlib inline
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import numpy as np

    myindex = 112
    fig_width = 10 # note DPI seems to be 100
    spacing = 100 # the lower the spacing, the larger the file, most likely
    #angle_spacing = 10 # the lower the angle_spacing, the larger the movie
    angle_spacing = 15

    if True:
        # Plot the result
        fig1 = plt.figure(figsize=(fig_width,fig_width))
        ax1 = fig1.add_subplot(111)
        #plt.imshow(overlay[myindex,:,:,:])
        #plt.imshow(overlay[:,myindex,:,:])
        ax1.imshow(overlay[:,myindex,:,:])
        plt.show()

    if False:
        z_axis = (0,1,2)
        x_axis = (1,2,0)
        y_axis = (2,0,1)
        ax_choice = y_axis
        
        fig2 = plt.figure(figsize=(fig_width,fig_width))
        ax2 = fig2.gca(projection='3d')
        ax2.set_aspect('equal')
        #fig2.axis('equal')
        ix,iy,iz = np.nonzero(labels.transpose(ax_choice)!=0)
        ix = ix[0::spacing]
        iy = iy[0::spacing]
        iz = iz[0::spacing]
        c = labels.transpose(ax_choice)[ix,iy,iz]
        ax2.scatter(ix,iy,iz,c=c-1,marker='.')
        make_axes_equal(ax2,ix,iy,iz)
        az = np.arange(360)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        for iaz in az[0::angle_spacing]:
            ax2.view_init(30,iaz)
            iaz_str = '{:03d}'.format(iaz)
            fn = 'movie/frame_'+iaz_str+'.png'
            plt.savefig(fn,dpi='figure')
            #plt.show()
        #plt.savefig('test.png',dpi='figure')

def main():

    if False:

        #### Step 1 according to instructions
        # Create good masks (both intersection and union) by "filling out" the regions that we know are true mitochondria based on the intersection filtering out the false positives
        
        import numpy as np

        # Parameter
        num = 1

        # Inputs (inferences on the FOV)
        #x_msk, y_msk, z_msk
        z_msk = np.load(str(num)+'-z_mask.npy')
        nz = z_msk.shape[0]
        nx = z_msk.shape[1]
        ny = z_msk.shape[2]
        x_msk = np.load(str(num)+'-x_mask.npy').transpose((1,2,0))[0:nz,:,:]
        y_msk = np.load(str(num)+'-y_mask.npy').transpose((2,0,1))[0:nz,:,:]

        # Take intersection to eliminate pancake structures and regions that look like mitochondria in the current direction
        int_msk = x_msk & y_msk & z_msk # this will be somewhat sparse but its centers should indicate where real mitochondria are
            
        # Go down z axis plane by plane, and repeat going down the x and y axes
        z_msk_good = fill_out_msk(z_msk,int_msk)
        x_msk_good = fill_out_msk(x_msk.transpose((1,2,0)),int_msk.transpose((1,2,0))).transpose((2,0,1))
        y_msk_good = fill_out_msk(y_msk.transpose((2,0,1)),int_msk.transpose((2,0,1))).transpose((1,2,0))

        # Explore what the resulting intersection and union look like; these are our results
        #int_good = x_msk_good & y_msk_good & z_msk_good
        #union_good = x_msk_good | y_msk_good | z_msk_good

        # Save the results
        #np.save('good_intersection_masks.npy',int_good)
        int_good = read_or_create_npy_file('good_intersection_masks.npy',"x_msk_good & y_msk_good & z_msk_good")
        #np.save('good_union_masks.npy',union_good)
        union_good = read_or_create_npy_file('good_union_masks.npy',"x_msk_good | y_msk_good | z_msk_good")

        
        #### Step 2 according to instructions
        # Labeling (this takes a long time if the labels need to be created on the FOV)

        # Import relevant modules
        import numpy as np
        from skimage import measure

        # Parameters
        #img_npy_file = '/Users/weismanal/notebook/2018-11-15/roi1_images_gray_unpadded.npy'
        img_npy_file = '/Users/weismanal/notebook/2018-11-03/inferences/field_of_view.npy'
        #msk_npy_file = '/Users/weismanal/notebook/2018-11-15/roi1_masks_unpadded.npy'
        #focus = 'intersection'
        focus = 'intersection'
        #msk_npy_file = 'good_union_masks.npy'
        #alpha = 0.8
        alpha = 0
        write_tif_file = False

        # Variables
        msk_npy_file = 'good_' + focus + '_masks.npy'
        lbl_npy_file = 'good_' + focus + '_labels.npy'

        # Load the data
        img = np.load(img_npy_file).astype('uint8')
        msk = np.load(msk_npy_file).astype('uint8')

        # Make the image RGB and blue
        img = np.tile(img.reshape(img.shape+(1,)),(1,1,1,3))
        img[:,:,:,0] = 255

        # Determine 3D labels/colors
        #labels = measure.label(msk).astype('uint64')
        #np.save('good_intersection_labels.npy',labels)
        labels = read_or_create_npy_file(lbl_npy_file,"measure.label(msk).astype('uint64')") # if in the end e.g. max(labels)<2^64, do np.save('good_intersection_labels.npy',labels.astype('uint16'))
        labels = randomize_labels(labels)
        colors = np.copy(labels)
        colors[labels!=0] -= 1
        colors = colors / np.max(colors) * (2**8-1)
        colors = colors.astype('uint8')

        # Overlay the masks on the images
        #overlay = overlay_mask(img,labels0,alpha,colors0)
        overlay = overlay_mask(img,labels,alpha,colors)

        if write_tif_file:
            import my_module as mm
            mm.numpy_array_to_tif_file(overlay,'good_'+focus+'_overlay.tif')

        
        #### Step 3 according to instructions
        # Plot the results

        #%matplotlib
        #%matplotlib inline
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import numpy as np

        myindex = 112
        fig_width = 10 # note DPI seems to be 100
        spacing = 100 # the lower the spacing, the larger the file, most likely
        #angle_spacing = 10 # the lower the angle_spacing, the larger the movie
        angle_spacing = 15

        if True:
            # Plot the result
            fig1 = plt.figure(figsize=(fig_width,fig_width))
            ax1 = fig1.add_subplot(111)
            #plt.imshow(overlay[myindex,:,:,:])
            #plt.imshow(overlay[:,myindex,:,:])
            ax1.imshow(overlay[:,myindex,:,:])
            plt.show()

        if False:
            z_axis = (0,1,2)
            x_axis = (1,2,0)
            y_axis = (2,0,1)
            ax_choice = y_axis
            
            fig2 = plt.figure(figsize=(fig_width,fig_width))
            ax2 = fig2.gca(projection='3d')
            ax2.set_aspect('equal')
            #fig2.axis('equal')
            ix,iy,iz = np.nonzero(labels.transpose(ax_choice)!=0)
            ix = ix[0::spacing]
            iy = iy[0::spacing]
            iz = iz[0::spacing]
            c = labels.transpose(ax_choice)[ix,iy,iz]
            ax2.scatter(ix,iy,iz,c=c-1,marker='.')
            make_axes_equal(ax2,ix,iy,iz)
            az = np.arange(360)
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_zticklabels([])
            for iaz in az[0::angle_spacing]:
                ax2.view_init(30,iaz)
                iaz_str = '{:03d}'.format(iaz)
                fn = 'movie/frame_'+iaz_str+'.png'
                plt.savefig(fn,dpi='figure')
                #plt.show()
            #plt.savefig('test.png',dpi='figure')