_model_id='CNN5FAT'
def model_init(input_shape,**kwargs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    
    assert(len(input_shape)==3)
    
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
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))
    
    model.add(Convolution2D(48,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Convolution2D(64,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(80,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.4))

    model.add(Convolution2D(96,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))    

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return dict(model=model,lr_mult=1.0)

