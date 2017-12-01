from warnings import warn
_model_id='CNN3BN'
def model_init(input_shape,**kwargs):
    from keras.models import Sequential
    from keras.layers import TimeDistributed, LSTM
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization    
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


    if len(input_shape)==3:
        lstm_input_shape = [None]+list(input_shape)
        cnn_input_shape = input_shape
    elif len(input_shape)==4:
        lstm_input_shape = input_shape
        cnn_input_shape = input_shape[1:]
    
    model = Sequential()
    model.add(Convolution2D(16,(3,3),input_shape=cnn_input_shape,padding='same'))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.1))
    
    model.add(Convolution2D(32,(2,2)))
    model.add(BatchNormalization())        
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(64,(2,2)))
    model.add(BatchNormalization())        
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(SpatialDropout2D(0.5))

    model.add(Flatten())

    # define LSTM model
    lstm_model = Sequential()
    lstm_model.add(TimeDistributed(model,input_shape=lstm_input_shape))
    lstm_model.add(LSTM(64,stateful=False,dropout=0.2))
        
    lstm_model.add(Dense(2048))
    lstm_model.add(Activation('relu'))
    lstm_model.add(Dropout(0.5))
    
    return dict(model=model,lr_mult=1.0)

