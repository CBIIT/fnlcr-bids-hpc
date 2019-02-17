# Load relevant modules
from image_segmentation import load_images, load_inferred_masks, calculate_metrics

# Parameters
roi = 'roi3'
# The following list of models is created by running in Bash: tmp=$(find . -type d -iregex "\./[0-9][0-9]-hpset.*" | sort | awk -v FS="./" -v ORS="','" '{print $2}'); models="['${tmp:0:${#tmp}-2}]"; echo $models
#models = ['01-hpset_10','02-hpset_11','03-hpset_16','04-hpset_17','05-hpset_21a','06-hpset_21b','07-hpset_21c','08-hpset_21d','09-hpset_22','10-hpset_23','11-hpset_28','12-hpset_30','13-hpset_32','14-hpset_33','15-hpset_34','16-hpset_last_good_unet','17-hpset_resnet']
models = ['08-hpset_21d','09-hpset_22']
inference_directions = ['x','y','z']

# Load the data
images = load_images(roi+'_input_img.npy')
known_masks = load_images('known_masks_'+roi+'.npy')
inferred_masks = load_inferred_masks(roi, images.shape, models, inference_directions)

# Calculate the metrics
nviews = 3
metrics_2d, metrics_3d = calculate_metrics(known_masks,inferred_masks,nviews)



import numpy as np
print(np.squeeze(metrics_2d[0,2,0,:,:]))



# # Constants
# plotting_transposes = ((0,1,2),(0,2,1),(0,2,1))
# dirs = ['x','y','z']

# # Variable
# plotting_transpose = plotting_transposes[dirs.index(view)]

# # Do some transposing just to make the plotting a reasonable orientation so we don't have to turn our heads 90 degrees when we look at the movies
# known_masks = np.transpose(known_masks,plotting_transpose)
# unpadded_shape = (unpadded_shape[plotting_transpose[0]],unpadded_shape[plotting_transpose[1]],unpadded_shape[plotting_transpose[2]])
# images = np.transpose(images,plotting_transpose)
# inferred_masks = np.transpose(inferred_masks,plotting_transpose)

# # Make the data RGBA
# images_rgba = imseg_utils.gray2rgba(images)
# known_masks_rgba = imseg_utils.gray2rgba(known_masks,A=0.2*255,mycolor=[0,0,1],makeBGTransp=True)
# inferred_masks_rgba = imseg_utils.gray2rgba(inferred_masks,A=0.2*255,mycolor=[1,0,0],makeBGTransp=True)