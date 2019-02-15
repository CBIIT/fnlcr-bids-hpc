def gray2rgba(img,A=255,mycolor=[1,1,1],makeBGTransp=False):
    # By default, make the image not transparent at all, make the transparency color white, and don't make the 0-pixels transparent
    import numpy as np
    mycolor = np.array(mycolor,dtype='float32')
    tmp = np.expand_dims(img,3)
    tmp = np.tile(tmp,(1,1,1,4))
    tmp[:,:,:,0] = (tmp[:,:,:,0]*mycolor[0]).astype('uint8')
    tmp[:,:,:,1] = (tmp[:,:,:,1]*mycolor[1]).astype('uint8')
    tmp[:,:,:,2] = (tmp[:,:,:,2]*mycolor[2]).astype('uint8')
    tmp[:,:,:,3] = A
    if makeBGTransp:
        bg0,bg1,bg2 = np.where(img==0)
        tmp[bg0,bg1,bg2,3] = 0
    return(tmp)

def count(arr,only2D=False):
    if not only2D:
        return(np.sum(arr))
    else:
        return(np.sum(arr,axis=(1,2)))

def calculate_metrics_func(msk0,msk1,only2D=False):

    # Arrays required for calculations of metrics
    target = msk0.astype('bool')
    guess = msk1.astype('bool')
    overlap_fg = target & guess
    overlap_bg = (~target) & (~guess)

    # Process the arrays for the true number of foreground and background pixels and those the model gets correct
    nfg = count(target,only2D=only2D)
    nbg = count(~target,only2D=only2D)
    noverlap_fg = count(overlap_fg,only2D=only2D)
    noverlap_bg = count(overlap_bg,only2D=only2D)

    # Convert to true/false positives/negatives
    npos = nfg # purple + blue
    nneg = nbg # gray + red
    ntruepos = noverlap_fg # --> should be number of purples
    ntrueneg = noverlap_bg # --> should be number of grays
    nfalsepos = nneg - ntrueneg # --> should be number of reds (reason this makes sense: a false positive is really a negative that's not a true negative)
    nfalseneg = npos - ntruepos # --> should be number of blues (reason this makes sense: a false negative is really a positive that's not a true positive)

    # Metrics that should all be large
    tpr = ntruepos / (ntruepos + nfalseneg) # = ntruepos / npos # i.e., how good you are at detecting mito (sensitivity = recall = true positive rate)
    tnr = ntrueneg / (ntrueneg + nfalsepos) # = ntrueneg / nneg # i.e., how good you are at detecting NOT mito (specificity = selectivity = true negative rate)
    ppv = ntruepos / (ntruepos + nfalsepos) # positive predictive value = precision (i.e., of all the red that you see, how much of it is correct?)
    bacc = (tpr+tnr) / 2 # Overall accuracy (balanced accuracy)
    f1 = 2 / ( (1/tpr) + (1/ppv) )
    
    return(tpr,tnr,ppv,bacc,f1)

def process_data_for_view(view_transpose_index,inference_on_dir,roi_num,model): # order of parameters here is view, inference direction, ROI

    # Import relevant modules
    import bids_hpc_utils as bhu
    import numpy as np
    
    # Constants
    # reverse_transpose_indices = {
    #   'z': 0,
    #   'x': 2,
    #   'y': 1
    # }
    reverse_transpose_indices = {
      'z': 0,
      'x': 1,
      'y': 2
    }
    transposes = ((0,1,2),(2,0,1),(1,2,0))
    #transposes = ((0,1,2),(1,2,0),(2,0,1))
    plotting_transposes = ((0,1,2),(0,2,1),(0,2,1))

    # Variables
    reverse_transpose_index = reverse_transpose_indices[inference_on_dir]
    plotting_transpose = plotting_transposes[view_transpose_index]

    # Load the data and normalize to uint8
    msk0 = bhu.normalize_images(np.load('known_masks_roi'+str(roi_num)+'.npy').transpose(transposes[view_transpose_index]),1)
    shp = msk0.shape
    img = bhu.normalize_images((np.load('roi'+str(roi_num)+'_input_img.npy').transpose(transposes[view_transpose_index]))[:shp[0],:shp[1],:shp[2]],1) # this is actually specific to the ROI only and should be renamed with a ROI index, e.g., 'images_roiX.npy'
    msk1 = bhu.normalize_images((np.round(np.load(model+'/inferred_masks-roi'+str(roi_num)+'-'+inference_on_dir+'_first.npy')).transpose(transposes[reverse_transpose_index]).transpose(transposes[view_transpose_index]))[:shp[0],:shp[1],:shp[2]],1)

    # Do some transposing just to make the plotting a reasonable orientation so we don't have to turn our heads
    msk0 = np.transpose(msk0,plotting_transpose)
    shp = (shp[plotting_transpose[0]],shp[plotting_transpose[1]],shp[plotting_transpose[2]])
    img = np.transpose(img,plotting_transpose)
    msk1 = np.transpose(msk1,plotting_transpose)

    # Make the data RGBA
    img_rgba = gray2rgba(img)
    msk0_rgba = gray2rgba(msk0,A=0.2*255,mycolor=[0,0,1],makeBGTransp=True)
    msk1_rgba = gray2rgba(msk1,A=0.2*255,mycolor=[1,0,0],makeBGTransp=True)

    # bhu.arr_info(msk0_rgba)
    # bhu.arr_info(img_rgba)
    # bhu.arr_info(msk1_rgba)

    # sys.exit()

    # Calculate the metrics
    metrics_2d = np.array(calculate_metrics_func(msk0,msk1,only2D=True))
    #print('HERE1')
    metrics_3d = np.array(calculate_metrics_func(msk0,msk1))
    #print('HERE2')

    return(msk0,shp,img,msk1,img_rgba,msk0_rgba,msk1_rgba,metrics_2d,metrics_3d)

def get_colored_str(x):
    # Get HTML string that colors x according to its value so the table is colored
    col = 'black'
    if x >= 95:
        col = 'green'
    elif x >= 85:
        col = 'orange'
    elif x >= 75:
        col = 'red'
    numstr = '{: 4d}'.format(x)
    return('<font style="color:'+col+';">'+numstr+'</font>')


# Import relevant functions
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Process the parameters
model_name = sys.argv[1]
roi_nums = eval(sys.argv[2])
ninf = int(sys.argv[3])
calculate_metrics = bool(int(sys.argv[4]))
create_plots = bool(int(sys.argv[5]))
nframes = int(sys.argv[6])

print(model_name, roi_nums, ninf, calculate_metrics, create_plots, nframes)

#sys.exit()

# Variables
if ninf == 3:
    inferences_on_dir = ['z','x','y']
    views = [0,1,2]
    my_fig_size = (16,9)
else:
    inferences_on_dir = ['z']
    views = [0]
    my_fig_size = (6,9)

# Constant
legend_arr = ['true positive rate / sensitivity / recall','true negative rate / specificity / selectivity','positive predictive value / precision','balanced accuracy','f1 score']

# Variable
ax_dict = dict(zip(inferences_on_dir,views))

# Open the file for writing the metrics if we're going to be writing them
if calculate_metrics:
    file = open('3d_metrics.txt','a')
    
# For each ROI...
for roi_num in roi_nums:
    new_view = True

    # For each view...
    for view in views:
        view2 = inferences_on_dir[view]
    
        # Load the process the data for each view for each ROI (note that each image has different inference directions)
        msk0_left,shp_left,img_left,msk1_left,img_rgba_left,msk0_rgba_left,msk1_rgba_left,metrics_2d_left,metrics_3d_left = process_data_for_view(view,'z',roi_num,model_name) # these are named _x, _y, _z in name only for an older type of plot
        if ninf == 3:
            msk0_middle,shp_middle,img_middle,msk1_middle,img_rgba_middle,msk0_rgba_middle,msk1_rgba_middle,metrics_2d_middle,metrics_3d_middle = process_data_for_view(view,'x',roi_num,model_name)
            msk0_right,shp_right,img_right,msk1_right,img_rgba_right,msk0_rgba_right,msk1_rgba_right,metrics_2d_right,metrics_3d_right = process_data_for_view(view,'y',roi_num,model_name)
            labels = ['view_'+view2+'-inf_dir_'+'z'+'-roi'+str(roi_num),'view_'+view2+'-inf_dir_'+'x'+'-roi'+str(roi_num),'view_'+view2+'-inf_dir_'+'y'+'-roi'+str(roi_num)]
        else:
            labels = ['view_'+view2+'-inf_dir_'+'z'+'-roi'+str(roi_num)]
        dirname = model_name+'/movie-roi'+str(roi_num)+'_view_'+view2

        # Write the metrics if desired
        if calculate_metrics:
            if new_view:
                file.write(str(int(model_name.split('-',1)[0]))+'\t'+str(ax_dict[labels[0].split('inf_dir_',1)[1].split('-roi',1)[0]])+'\t'+str(roi_num)+'\t'+str(metrics_3d_left).replace('[','').replace(']','')+'\n')
                if ninf == 3:
                    file.write(str(int(model_name.split('-',1)[0]))+'\t'+str(ax_dict[labels[1].split('inf_dir_',1)[1].split('-roi',1)[0]])+'\t'+str(roi_num)+'\t'+str(metrics_3d_middle).replace('[','').replace(']','')+'\n')
                    file.write(str(int(model_name.split('-',1)[0]))+'\t'+str(ax_dict[labels[2].split('inf_dir_',1)[1].split('-roi',1)[0]])+'\t'+str(roi_num)+'\t'+str(metrics_3d_right).replace('[','').replace(']','')+'\n')

        # If we want to create plots of the data...
        if create_plots:
        
            # First plot the frame-independent data and metrics
            myfig = plt.figure(figsize=my_fig_size) # interestingly you must initialize figsize here in order to make later calls to myfig.set_figwidth(X) work
            ax1 = plt.subplot(2,ninf,0+1)
            ax2 = plt.subplot(2,ninf,0+ninf+1)
            ax1.set_title(labels[0])
            ax2.plot(np.transpose(metrics_2d_left))
            ax2.set_xlim(0,shp_left[0]-1)
            ax2.set_ylim(0,1)
            ax2.set_xlabel('3D stats: tpr='+'{:04.2f}'.format(metrics_3d_left[0])+', tnr='+'{:04.2f}'.format(metrics_3d_left[1])+', ppv='+'{:04.2f}'.format(metrics_3d_left[2])+', bacc='+'{:04.2f}'.format(metrics_3d_left[3])+', f1='+'{:04.2f}'.format(metrics_3d_left[4]))
            ax2.set_ylabel(model_name)
            ax2.legend(legend_arr,loc='lower left')
            if ninf == 3:
                ax3 = plt.subplot(2,ninf,1+1)
                ax4 = plt.subplot(2,ninf,1+ninf+1)
                ax3.set_title(labels[1])
                ax5 = plt.subplot(2,ninf,2+1)
                ax6 = plt.subplot(2,ninf,2+ninf+1)
                ax5.set_title(labels[2])
                ax4.plot(np.transpose(metrics_2d_middle))
                ax4.set_xlim(0,shp_middle[0]-1)
                ax4.set_ylim(0,1)
                ax4.set_xlabel('3D stats: tpr='+'{:04.2f}'.format(metrics_3d_middle[0])+', tnr='+'{:04.2f}'.format(metrics_3d_middle[1])+', ppv='+'{:04.2f}'.format(metrics_3d_middle[2])+', bacc='+'{:04.2f}'.format(metrics_3d_middle[3])+', f1='+'{:04.2f}'.format(metrics_3d_middle[4]))
                ax4.legend(legend_arr,loc='lower left')
                ax6.plot(np.transpose(metrics_2d_right))
                ax6.set_xlim(0,shp_right[0]-1)
                ax6.set_ylim(0,1)
                ax6.set_xlabel('3D stats: tpr='+'{:04.2f}'.format(metrics_3d_right[0])+', tnr='+'{:04.2f}'.format(metrics_3d_right[1])+', ppv='+'{:04.2f}'.format(metrics_3d_right[2])+', bacc='+'{:04.2f}'.format(metrics_3d_right[3])+', f1='+'{:04.2f}'.format(metrics_3d_right[4]))
                ax6.legend(legend_arr,loc='lower left')
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            # Now plot the frame-dependent data and metrics
            frames_left = np.linspace(0,shp_left[0]-1,num=nframes).astype('uint16')
            if ninf == 3:
                frames_middle = np.linspace(0,shp_middle[0]-1,num=nframes).astype('uint16')
                frames_right = np.linspace(0,shp_right[0]-1,num=nframes).astype('uint16')
            iframe = 0
            while iframe < nframes:
            #for frame2plot_left,frame2plot_middle,frame2plot_right in zip(frames_left,frames_middle,frames_right):
                frame2plot_left = frames_left[iframe]
                if ninf == 3:
                    frame2plot_middle = frames_middle[iframe]
                    frame2plot_right = frames_right[iframe]
                iframe += 1
                ax2.set_title('tpr='+'{:04.2f}'.format(metrics_2d_left[0,frame2plot_left])+' tnr='+'{:04.2f}'.format(metrics_2d_left[1,frame2plot_left])+' ppv='+'{:04.2f}'.format(metrics_2d_left[2,frame2plot_left])+' bacc='+'{:04.2f}'.format(metrics_2d_left[3,frame2plot_left])+' f1='+'{:04.2f}'.format(metrics_2d_left[4,frame2plot_left]))
                im1_left = ax1.imshow(img_rgba_left[frame2plot_left,:,:,:])
                im2_left = ax1.imshow(msk0_rgba_left[frame2plot_left,:,:,:])
                im3_left = ax1.imshow(msk1_rgba_left[frame2plot_left,:,:,:])
                sct_left = ax2.scatter([frame2plot_left,frame2plot_left,frame2plot_left,frame2plot_left,frame2plot_left],metrics_2d_left[:,frame2plot_left],c=['C0','C1','C2','C3','C4'])
                if ninf == 3:
                    ax4.set_title('tpr='+'{:04.2f}'.format(metrics_2d_middle[0,frame2plot_middle])+' tnr='+'{:04.2f}'.format(metrics_2d_middle[1,frame2plot_middle])+' ppv='+'{:04.2f}'.format(metrics_2d_middle[2,frame2plot_middle])+' bacc='+'{:04.2f}'.format(metrics_2d_middle[3,frame2plot_middle])+' f1='+'{:04.2f}'.format(metrics_2d_middle[4,frame2plot_middle]))
                    ax6.set_title('tpr='+'{:04.2f}'.format(metrics_2d_right[0,frame2plot_right])+' tnr='+'{:04.2f}'.format(metrics_2d_right[1,frame2plot_right])+' ppv='+'{:04.2f}'.format(metrics_2d_right[2,frame2plot_right])+' bacc='+'{:04.2f}'.format(metrics_2d_right[3,frame2plot_right])+' f1='+'{:04.2f}'.format(metrics_2d_right[4,frame2plot_right]))
                    im1_middle = ax3.imshow(img_rgba_middle[frame2plot_middle,:,:,:])
                    im2_middle = ax3.imshow(msk0_rgba_middle[frame2plot_middle,:,:,:])
                    im3_middle = ax3.imshow(msk1_rgba_middle[frame2plot_middle,:,:,:])
                    sct_middle = ax4.scatter([frame2plot_middle,frame2plot_middle,frame2plot_middle,frame2plot_middle,frame2plot_middle],metrics_2d_middle[:,frame2plot_middle],c=['C0','C1','C2','C3','C4'])
                    im1_right = ax5.imshow(img_rgba_right[frame2plot_right,:,:,:])
                    im2_right = ax5.imshow(msk0_rgba_right[frame2plot_right,:,:,:])
                    im3_right = ax5.imshow(msk1_rgba_right[frame2plot_right,:,:,:])
                    sct_right = ax6.scatter([frame2plot_right,frame2plot_right,frame2plot_right,frame2plot_right,frame2plot_right],metrics_2d_right[:,frame2plot_right],c=['C0','C1','C2','C3','C4'])
                plt.show(block=False)
                plt.savefig(dirname+'/frame_'+'{:04d}'.format(iframe)+'.png',dpi='figure')
                if frame2plot_left != frames_left[-1]: # since they should all be together I'm just doing the left
                    im1_left.set_visible(False)
                    im2_left.set_visible(False)
                    im3_left.set_visible(False)
                    sct_left.set_visible(False)
                    if ninf == 3:
                        im1_middle.set_visible(False)
                        im2_middle.set_visible(False)
                        im3_middle.set_visible(False)
                        sct_middle.set_visible(False)
                        im1_right.set_visible(False)
                        im2_right.set_visible(False)
                        im3_right.set_visible(False)
                        sct_right.set_visible(False)
                    
        new_view = False
                
# If we've written the metrics, create an HTML table for displaying them
if calculate_metrics:

    # Close the file
    file.close()

    # Load the metrics into a single matrix
    metrics = np.loadtxt('3d_metrics.txt')

    # For each ROI...
    for roi in roi_nums:

        # Get the data for the current ROI
        data_for_roi = metrics[metrics[:,2]==roi,:] # get data for the current roi

        # Print the header
        print('++++ ROI'+str(roi)+' ++++',end='\n\n')
        print('        |     TPR     |     TNR     |     PPV     |    BACC     |     F1      |')
        print('-------------------------------------------------------------------------------',end='')

        # For each model...
        for model in np.unique(metrics[:,0]):
            
            # Get the data for the current model
            data_for_model = data_for_roi[data_for_roi[:,0]==model,:]
            
            # Print the header column for the current model
            print('')
            print('Model '+str(int(model))+' |',end='')

            # For each metric...
            for imetric in np.array(range(5))+3:

                # Get the metrics for all three inferences in z,x,y order since the output to the text file labels the inferences as 0,1,2 in that order
                zxy = np.round( data_for_model[:,imetric] * 100 ).astype('uint8')

                # For each datum, get a colored version of it
                z = get_colored_str(zxy[0])
                if ninf == 3:
                    x = get_colored_str(zxy[1])
                    y = get_colored_str(zxy[2])

                # Print the data for each inference in x,y,z order
                if ninf == 3:
                    print(x+y+z+' |',end='')
                else:
                    print('    '+z+'     |',end='')

        # Put a nice big space at the end of the table for the current ROI
        print('',end='\n\n\n')




#  96  97 100  --> 13 spaces
# 12 spaces for purely numbers
# nominally will be ----__XX----_
