from warnings import warn
_model_id='CNN3'
def model_init(input_shape,**kwargs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.backend import set_image_data_format
    
    try:
        from keras.layers.core import SpatialDropout2D
    except:
        from keras import __version__ as __kv__
        from warnings import warn
        warn('no SpatialDropout2D layer in keras version: %s'%__kv__)
        SpatialDropout2D = Dropout

    assert(len(input_shape)==3 and input_shape[2]==3)
    set_image_data_format('channels_last')

    nb_hidden = kwargs.pop('nb_hidden',1024)
    
    model.add(Convolution2D(16,(3,3),input_shape=input_shape,padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))
    
    model.add(Convolution2D(32,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(64,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))

    model.add(Flatten())
    model.add(Dense(2*nb_hidden))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return model

