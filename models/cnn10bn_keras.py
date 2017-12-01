def model_init(input_shape,**kwargs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization    
    from keras.backend import set_image_data_format
    
    assert(len(input_shape)==3 and input_shape[2]==3)
    set_image_data_format('channels_last')
    
    try:
        from keras.layers.core import SpatialDropout2D
    except:
        from keras import __version__ as __kv__
        from warnings import warn
        warn('no SpatialDropout2D layer in keras version: %s'%__kv__)
        SpatialDropout2D = Dropout

    # need to set the input_shape to first layer for a new model
    model = Sequential()
    model.add(Convolution2D(32,(3,3),padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))

    # 2
    model.add(Convolution2D(48,(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))

    # 3
    model.add(Convolution2D(64,(2,2)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.2))

    # 4
    model.add(Convolution2D(128,(2,2)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.2))

    # 5
    model.add(Convolution2D(164,(2,2)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.3))

    # 6
    model.add(Convolution2D(172,(2,2)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.3))      

    # 7
    model.add(Convolution2D(196,(2,2)))
    model.add(BatchNormalization())     
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.4))      

    # 8
    model.add(Convolution2D(224,(2,2)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.4))      

    # 9
    model.add(Convolution2D(248,(2,2)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))      

    # 10
    model.add(Convolution2D(296,(2,2)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))      
    
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return dict(model=model,lr_mult=1.0)

