from warnings import warn
import os
_model_id=os.path.split(os.path.splitext(__file__)[0])[1].upper()
def model_init(input_shape,**kwargs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers import GlobalAveragePooling2D
    
    try:
        from keras.layers.core import SpatialDropout2D
    except:
        from keras import __version__ as __kv__
        from warnings import warn
        warn('no SpatialDropout2D layer in keras version: %s'%__kv__)
        SpatialDropout2D = Dropout

    assert(len(input_shape)==3)

    model = Sequential()
    model.add(Convolution2D(32,(3,3),padding='same',activation='relu',
                            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))
    
    model.add(Convolution2D(64,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(64,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048,activation='relu'))
    model.add(Dropout(0.5))

    return dict(model=model,lr_mult=1.0)
