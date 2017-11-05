def model_init(input_shape,**kwargs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
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
    
    if 'base_model' not in kwargs:
        # need to set the input_shape to first layer for a new model
        model = Sequential()
        layer0 = Convolution2D(16,(3,3),padding='same',input_shape=input_shape)
    else:
        # use the shape of the last layer of the base model as the input_shape
        model = kwargs.pop('base_model')
        layer0 = Convolution2D(16,(3,3),padding='same')

    model.add(layer0)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))
    
    model.add(Convolution2D(24,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))

    model.add(Convolution2D(30,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(36,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(42,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(48,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.3))      

    model.add(Convolution2D(54,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.4))      

    model.add(Convolution2D(60,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.4))      

    model.add(Convolution2D(66,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))      
    
    model.add(Convolution2D(72,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))      
    
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return model

