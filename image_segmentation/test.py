# In Bash: module load python/3.6
import sys
sys.path.append('/home/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation/packages')
from image_segmentation import testing
#testing.load_data()
#testing.calculate_metrics()
#testing.output_metrics()
#testing.make_plots()



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