from warnings import warn
_model_id='CNN9SEPBN'
def model_init(input_shape,**kwargs):
    from keras import backend as _backend
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers import SeparableConv2D as Convolution2D
    from keras.layers import MaxPooling2D, GlobalAveragePooling2D
    from keras.layers.normalization import BatchNormalization
    try:
        from keras.layers.core import SpatialDropout2D
    except:
        from keras import __version__ as __kv__
        from warnings import warn
        warn('no SpatialDropout2D layer in keras version: %s'%__kv__)
        SpatialDropout2D = Dropout

    assert(len(input_shape)==3)
      
    model = Sequential()
    model.add(Convolution2D(64,(3,3),padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))
    
    model.add(Convolution2D(128,(2,2)))
    model.add(BatchNormalization())        
    model.add(Activation('relu'))

    model.add(Convolution2D(128,(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(256,(2,2)))
    model.add(BatchNormalization())        
    model.add(Activation('relu'))

    model.add(Convolution2D(256,(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))    

    model.add(Convolution2D(384,(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(384,(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.4))

    model.add(Convolution2D(512,(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(512,(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.5))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return dict(model=model,lr_mult=1.0)

