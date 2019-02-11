from __future__ import print_function

from keras import backend as K
import os

def initialize_parameters():
    print('Initializing parameters...')
    
    # Parameters
    candle_lib = '/data/BIDS-HPC/public/candle/Candle/common'

    # Obtain the path of the directory of this script
    file_path = os.path.dirname(os.path.realpath(__file__))

    # Import the CANDLE library
    import sys
    sys.path.append(candle_lib)
    import candle_keras as candle

    # Instantiate the candle.Benchmark class
    mymodel_common = candle.Benchmark(file_path,os.getenv("DEFAULT_PARAMS_FILE"),'keras',prog='myprog',desc='My model')

    # Get a dictionary of the model hyperparamters
    gParameters = candle.initialize_parameters(mymodel_common)

    # Return the dictionary of the hyperparameters
    return(gParameters)
    
def run(gParameters):
    print('Running model...')

    #### Begin model input ##########################################################################################
    
    def get_model(model_json_fname,modelwtsfname):
        # This is only for prediction
        if os.path.isfile(model_json_fname):
             # Model reconstruction from JSON file
             with open(model_json_fname, 'r') as f:
                model = model_from_json(f.read())
        else:
             model = get_unet()
        
        #model.summary()     
        # Load weights into the new model
        model.load_weights(modelwtsfname)
        return model      
    
    def focal_loss(gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
        return focal_loss_fixed
    
    def jaccard_coef(y_true, y_pred):
        smooth = 1.0
        intersection = K.sum(y_true * y_pred, axis=[-0, -1, 2])
        sum_ = K.sum(y_true + y_pred, axis=[-0, -1, 2])
    
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
        return K.mean(jac)
    
    def jaccard_coef_int(y_true, y_pred):
        smooth = 1.0
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    
        intersection = K.sum(y_true * y_pred_pos, axis=[-0, -1, 2])
        sum_ = K.sum(y_true + y_pred_pos, axis=[-0, -1, 2])
    
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
        return K.mean(jac)
    
    def jaccard_coef_loss(y_true, y_pred):
        return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
    
    def flip_axis(x, axis):
        x = x.swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
    
    def form_batch(X,y,batch_size,img_rows,img_cols,num_channels,num_mask_channels):
        # I believe this takes random crops of size img_rows,img_cols from random images of the input batch
        X_batch = np.zeros((batch_size, img_rows, img_cols, num_channels))
        y_batch = np.zeros((batch_size, img_rows, img_cols, num_mask_channels))
        X_height = X.shape[1]
        X_width = X.shape[2]
    
        for i in range(batch_size):
            random_width = random.randint(0, X_width - img_cols - 1)
            random_height = random.randint(0, X_height - img_rows - 1)
    
            random_image = random.randint(0, X.shape[0] - 1)
    
            y_batch[i] = y[random_image, random_height: random_height + img_rows, random_width: random_width + img_cols,:]
            X_batch[i] = X[random_image, random_height: random_height + img_rows, random_width: random_width + img_cols,:]
        return X_batch, y_batch
    
    
    def batch_generator(X, y, batch_size,img_rows,img_cols,num_channels,num_mask_channels, horizontal_flip=False, vertical_flip=False, swap_axis=False):
        # This probably takes random crops and does random flips and 90-degree rotations of the input data
        while True:
            X_batch, y_batch = form_batch(X, y, batch_size,img_rows,img_cols,num_channels,num_mask_channels)
            #print('Size of X batch = ' + str(X_batch.shape))
            #print('Size of Y batch = ' + str(y_batch.shape))
    
            for i in range(X_batch.shape[0]):
                xb = X_batch[i]
                yb = y_batch[i]
    
                if horizontal_flip:
                    if random.random() < 0.5:
                        xb = flip_axis(xb, 0)
                        yb = flip_axis(yb, 0)
                        #print('h_flip xb = ' + str(xb.shape))
                        #print('h_flip yb = ' + str(yb.shape))
    
                if vertical_flip:
                    if random.random() < 0.5:
                        xb = flip_axis(xb, 1)
                        yb = flip_axis(yb, 1)
                        #print('v_flip xb = ' + str(xb.shape))
                        #print('v_flip yb = ' + str(yb.shape))
    
                if swap_axis:
                    if random.random() < 0.5:
                        xb = xb.swapaxes(0, 1)
                        yb = yb.swapaxes(0, 1)
                        #print('swap xb = ' + str(xb.shape))
                        #print('swap yb = ' + str(yb.shape))
    
                X_batch[i] = xb
                y_batch[i] = yb
    
            yield X_batch, y_batch
    
    def dice_coef_batch(y_true, y_pred):
        smooth = 1.0
        intersection = K.sum(y_true * y_pred, axis=[-0, -1, 2])
        sum_ = K.sum(y_true + y_pred, axis=[-0, -1, 2])
    
        dice = ((2.0*intersection) + smooth) / (sum_ + intersection + smooth)
    
        return K.mean(dice)
    
    def dice_coef(y_true, y_pred):
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_smooth = ((2. * intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return (dice_smooth)
    
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)
    
    def dice_coef_batch_loss(y_true, y_pred):
        return -dice_coef_batch(y_true, y_pred)
    
    #data consinstency check, input NPY arrays should be 3D (number of images, height, width)
    def data_consistency_check(imgs,masks):
        assert(len(imgs.shape)==len(masks.shape))
        assert(imgs.shape[0]==masks.shape[0])
        assert(imgs.shape[1]==masks.shape[1])
        assert(imgs.shape[2]==masks.shape[2])
        # check if the masks is between 0 and 1
        assert(np.amin(masks) == 0 and np.amax(masks) == 1)
    
    # randomly change the standardization maximum value. We will agument each image by augFactor
    def random_standardize(imgs,masks, augFactor=10):
        data_consistency_check(imgs, masks)
        imgs_rand = np.empty([imgs.shape[0]*augFactor, imgs.shape[1], imgs.shape[2]], dtype='float32')
        masks_rand = np.empty([imgs.shape[0]*augFactor, imgs.shape[1], imgs.shape[2]], dtype='float32')
        k = 0
        for i in range(imgs.shape[0]):
            for j in range(1,augFactor+1):
                c_img = np.squeeze(imgs[i,:,:])
                c_mask = np.squeeze(masks[i,:,:])
                amax = np.amax(c_img)
                fac = (float(j)/float(augFactor))/amax
                print(str(i) + ":" + str(j) + ":" + str(augFactor) + ":" + str(fac) + ":" + str(amax))
                imgs_rand[k,:,:] = c_img*fac
                masks_rand[k,:,:] = c_mask  
                k = k+1
        imgs_rand = np.concatenate((imgs_rand, imgs), axis=0)
        masks_rand = np.concatenate((masks_rand, imgs), axis=0)
        return imgs_rand, masks_rand
    
    #Define the neural network
    def get_unet():
        droprate = 0.25
        filt_size = 32
        inputs = Input((None, None, 1))
        conv1 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Dropout(droprate)(conv1) 
        conv1 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        filt_size = filt_size*2
    
        conv2 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Dropout(droprate)(conv2) 
        conv2 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        filt_size = filt_size*2
    
        conv3 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Dropout(droprate)(conv3) 
        conv3 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        filt_size = filt_size*2
    
        conv4 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Dropout(droprate)(conv4) 
        conv4 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        filt_size = filt_size*2
        
        conv5 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Dropout(droprate)(conv5)
        conv5 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv5)
    
        filt_size = filt_size/2
    
        up6 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Dropout(droprate)(conv6)
        conv6 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv6)
    
        filt_size = filt_size/2
    
        up7 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Dropout(droprate)(conv7)
        conv7 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv7)
    
        filt_size = filt_size/2
        
        up8 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Dropout(droprate)(conv8)
        conv8 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv8)
        filt_size = filt_size/2
        
        up9 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Dropout(droprate)(conv9)
        conv9 = Conv2D(filt_size, (3, 3), activation='relu', padding='same')(conv9)
    
        
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        
        model = Model(inputs=[inputs], outputs=[conv10])
    
        #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
        #model.compile(optimizer=Nadam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
        #model.compile(optimizer=Adadelta(), loss=dice_coef_loss, metrics=[dice_coef])
        
        return model
    
    #Define the neural network
    def get_unet0(img_rows,img_cols):
        #droprate = 0.25
        filt_size = 32
        inputs = Input((img_rows, img_cols, 1))
        
        conv1 = Conv2D(filt_size, (3, 3), padding='same')(inputs)
        conv1 = BatchNormalization(axis=3)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        conv1 = Conv2D(filt_size, (3, 3), padding='same')(conv1)
        conv1 = BatchNormalization( axis=3)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        #filt_size = filt_size*2
        print('Filter size = ' + str(filt_size))
    
        conv2 = Conv2D(filt_size, (3, 3), padding='same')(pool1)
        conv2 = BatchNormalization( axis=3)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        conv2 = Conv2D(filt_size, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization( axis=3)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #filt_size = filt_size*2
        print('Filter size = ' + str(filt_size))
    
        conv3 = Conv2D(filt_size, (3, 3), padding='same')(pool2)
        conv3 = BatchNormalization( axis=3)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        conv3 = Conv2D(filt_size, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization( axis=3)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #filt_size = filt_size*2
        print('Filter size = ' + str(filt_size))
    
        conv4 = Conv2D(filt_size, (3, 3), padding='same')(pool3)
        conv4 = BatchNormalization(axis=3)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        conv4 = Conv2D(filt_size, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization( axis=3)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        #filt_size = filt_size*2
        print('Filter size = ' + str(filt_size))
    
        conv5 = Conv2D(filt_size, (3, 3), padding='same')(pool4)
        conv5 = BatchNormalization( axis=3)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        conv5 = Conv2D(filt_size, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization( axis=3)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        #filt_size = filt_size/2
        print('Filter size = ' + str(filt_size))
    
        up6 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(filt_size, (3, 3), padding='same')(up6)
        conv6 = BatchNormalization( axis=3)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Conv2D(filt_size, (3, 3), padding='same')(conv6)
        conv6 = BatchNormalization( axis=3)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        #filt_size = filt_size/2
        print('Filter size = ' + str(filt_size))
    
        up7 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(filt_size, (3, 3), padding='same')(up7)
        conv7 = BatchNormalization( axis=3)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Conv2D(filt_size, (3, 3), padding='same')(conv7)
        conv7 = BatchNormalization( axis=3)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        #filt_size = filt_size/2
        print('Filter size = ' + str(filt_size))
        
        up8 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(filt_size, (3, 3), padding='same')(up8)
        conv8 = BatchNormalization( axis=3)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Conv2D(filt_size, (3, 3), padding='same')(conv8)
        conv8 = BatchNormalization( axis=3)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        #filt_size = filt_size/2
        print('Filter size = ' + str(filt_size))
        
        up9 = concatenate([Conv2DTranspose(filt_size, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(filt_size, (3, 3), padding='same')(up9)
        conv9 = BatchNormalization( axis=3)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(filt_size, (3, 3), padding='same')(conv9)
        conv9 = BatchNormalization( axis=3)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        
        model = Model(inputs=[inputs], outputs=[conv10])
        
        return model
    
    def save_model_to_json(model,model_json_fname):
        
        #model = unet.UResNet152(input_shape=(None, None, 3), classes=1,encoder_weights="imagenet11k")	
        #model = get_unet()
        
        #model.summary()
        # serialize model to JSON
        model_json = model.to_json()
        with open(model_json_fname, "w") as json_file:
             json_file.write(model_json)
    
    def preprocess_data(do_prediction,inputnpyfname,targetnpyfname,expandChannel,backbone):
        # Preprocess the data (beyond what I already did before)
        
        print('-'*30)
        print('Loading and preprocessing data...')
        print('-'*30)
    
        # Load, normalize, and cast the data
        imgs_input = ( np.load(inputnpyfname).astype('float32') / (2**16-1) * (2**8-1) ).astype('uint8')
        print('Input images information:')
        print(imgs_input.shape)
        print(imgs_input.dtype)
        hist,bins = np.histogram(imgs_input)
        print(hist)
        print(bins)
        if not do_prediction:
            imgs_mask_train = np.load(targetnpyfname).astype('uint8')
            print('Input masks information:')
            print(imgs_mask_train.shape)
            print(imgs_mask_train.dtype)
            hist,bins = np.histogram(imgs_mask_train)
            print(hist)
            print(bins)
    
        # Make the grayscale images RGB since that's what the model expects apparently
        if expandChannel:       
           imgs_input = np.stack((imgs_input,)*3, -1)
        else:
           imgs_input = np.expand_dims(imgs_input, 3)
        print('New shape of input images:')
        print(imgs_input.shape)
        if not do_prediction:
           imgs_mask_train = np.expand_dims(imgs_mask_train, 3)
           print('New shape of masks:')
           print(imgs_mask_train.shape)
    
        # Preprocess as per https://github.com/qubvel/segmentation_models
        preprocessing_fn = get_preprocessing(backbone)
        imgs_input = preprocessing_fn(imgs_input)
    
        # Return appropriate variables
        if not do_prediction:
            return(imgs_input,imgs_mask_train)
        else:
            return(imgs_input)

    # Parameters
    segmentation_models_repo = gParameters['segmentation_models_repo'] #sys.path.append('/home/weismanal/checkouts/segmentation_models')
    do_prediction = gParameters['predict'] #do_prediction = bool(int(sys.argv[1]))
    inputnpyfname = gParameters['images'] #inputnpyfname = sys.argv[2]
    labels = gParameters['labels'] # sys.argv[3]
    oldmodelwtsfname = gParameters['oldmodelwtsfname'] #oldmodelwtsfname = './old_model_weights.h5'
    backbone = gParameters['backbone'] #backbone = 'resnet152'
    encoder = gParameters['encoder'] #encoder = 'imagenet11k'
    lr = float(gParameters['lr'])
    batch_size = gParameters['batch_size']
    obj_return = gParameters['obj_return']
    epochs = gParameters['epochs']

    # Import relevant modules and functions
    import sys
    sys.path.append(segmentation_models_repo)
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,CSVLogger
    from keras.layers.normalization import BatchNormalization
    from keras.backend import binary_crossentropy
    import keras
    import random
    import tensorflow as tf
    from keras.models import model_from_json
    from segmentation_models import Unet
    from segmentation_models.backbones import get_preprocessing
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code
    
    # Basically constants
    expandChannel = True
    reloadFlag = False
    modelwtsfname = 'model_weights.h5'
    model_json_fname  = 'model.json'
    csvfname = 'model.csv'
    
    if not do_prediction: # Train...
        print('Training...')
        # Preprocess the data
        imgs_train,imgs_mask_train = preprocess_data(do_prediction,inputnpyfname,labels,expandChannel,backbone)
        # Load, save, and compile the model
        model = Unet(backbone_name=backbone, encoder_weights=encoder)
        save_model_to_json(model,model_json_fname)
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['binary_crossentropy','mean_squared_error',dice_coef, dice_coef_batch, focal_loss()])
        # Load previous weights for restarting, if desired and possible
        if os.path.isfile(oldmodelwtsfname) and reloadFlag :
            print('-'*30)
            print('Loading previous weights ...')
            model.load_weights(oldmodelwtsfname)
        # Set up the training callback functions
        model_checkpoint = ModelCheckpoint(modelwtsfname, monitor=obj_return, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor=obj_return, factor=0.1,patience=100, min_lr=0.001,verbose=1)
        model_es = EarlyStopping(monitor=obj_return, min_delta=0.00000001, patience=100, verbose=1, mode='auto')
        csv_logger = CSVLogger(csvfname, append=True)
        # Train the model
        history_callback = model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, reduce_lr, model_es, csv_logger])
        print("Minimum validation loss:")
        print(min(history_callback.history[obj_return]))
    else: # ...or predict
        print('Inferring...')
        # Preprocess the data
        imgs_infer = preprocess_data(do_prediction,inputnpyfname,'',expandChannel,backbone)
        # Load the model
        model = get_model(model_json_fname,modelwtsfname)
        # Run inference
        imgs_test_predict = model.predict(imgs_infer, batch_size=1, verbose=1)
        # Save the predicted masks
        np.save(labels, np.squeeze(np.round(imgs_test_predict).astype('uint8')))
        history_callback = None
    
    #### End model input ############################################################################################
    
    return(history_callback)

def main():
    print('Running main program...')
    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:
        pass