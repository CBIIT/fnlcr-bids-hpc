
from image_augmentation import AugmentationSettings, augment_images

class hitif_aug(AugmentationSettings):


    def __init__build_augmentation_seq(self, configuration):
        """
        Initialized the configuration prameters 
    
        Arguments:
            configuration: file pointer
                The hitif configuration file 
        
        """
        import configparser   
        
        config = configparser.ConfigParser()
        config.read(configuration)
    
        #Parse the augmentation parameters
        aug_prms = config['augmentation']
        self.CLAHE= eval(aug_prms['AllChannelsCLAHE'])
        self.impulse_noise = float(aug_prms['ImpulseNoise'])
        self.gaussian_blur = eval(aug_prms['GaussianBlur'])
        self.poisson = eval(aug_prms['AdditivePoissonNoise'])
        self.median = eval(aug_prms['MedianBlur'])
        self.flip = 0.5


        from imgaug import augmenters as iaa
        self.augmenters = {} 
        augmenters = self.augmenters
        augmenters["CLAHE"] = iaa.AllChannelsCLAHE(self.CLAHE)
        augmenters["impulse"] = iaa.ImpulseNoise(self.impulse_noise)
        augmenters["fliplr"] = iaa.Fliplr(self.flip)
        augmenters["flipud"] = iaa.Flipup(self.flip)
        augmenters["gaussian"] = iaa.GaussianBlur(self.gaussian_blur)
        augmenters["median"] = iaa.MedianBlur(self.median)
        augmenters["poisson"] = iaa.AdditivePoissonNoise(self.poisson)



    def composite_sequence(self):
        """Return the composite sequence to run, i.e., a set of transformations to all be applied to a set of images and/or masks.

        :returns: Sequential object from the augmenters module of the imgaug package
        """

        augmenters = self.augmenters
    
        from imgaug import augmenters as iaa
        self.seq = iaa.Sequential([
            augmenters["fliplr"],
            augmenters["flipup"],
            augmenters["CLAHE"],
            iaa.OneOf([
                augmenters["gaussian"],
                augmenters["median"],
            ]),
            iaa.OneOf([
                augmenters["poisson"],
                augmenters["impulse"],
            ])
        ])
    
        return self.seq

    def individual_seqs_and_outnames(self):
        """Return a list of individual sequences to run, i.e., a set of transformations to be applied one-by-one to a set of images and/or masks in order to see what the augmentations do individually.

        :returns: List of Sequential objects from the augmenters module of the imgaug package
        """

        from imgaug import augmenters as iaa

        
        augmentation_tasks = []
        augmenters = self.augmenters
        for name, augmentation in self.augmenters.items():
            augmentation_tasks.append([augmentaiton, name])

        return augmentation_tasks
