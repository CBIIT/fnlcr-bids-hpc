# In Bash: module load python/3.6
import sys
sys.path.append('/home/weismanal/checkouts/fnlcr-bids-hpc/image_segmentation/packages')
from image_segmentation import testing
#testing.load_data()
#testing.calculate_metrics()
#testing.output_metrics()
testing.make_plots()