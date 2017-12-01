from warnings import warn
_model_id = 'resnet50fcn'
def model_init(input_shape,**kwargs):
    from keras import backend
    from keras.applications import ResNet50
    fix_base = kwargs.pop('fix_base',False)
    nb_labels = kwargs.pop('nb_labels',2)
    
    assert(backend.backend()=='tensorflow')
    assert(len(input_shape)==3 and input_shape[2]==3)
    backend.set_image_data_format('channels_last')
    nb_rows,nb_cols = input_shape[:-1]
        
    base_model = ResNet50(weights="imagenet", include_top=False,
                          pooling='avg', input_shape=input_shape)

    # Get final 32x32, 16x16, and 8x8 layers in the original
    # ResNet by that layers's name.
    x32 = base_model.get_layer('final_32').output
    x16 = base_model.get_layer('final_16').output
    x8 = base_model.get_layer('final_x8').output

    # Compress each skip connection so it has nb_labels channels.
    c32 = Convolution2D(nb_labels, (1, 1))(x32)
    c16 = Convolution2D(nb_labels, (1, 1))(x16)
    c8 = Convolution2D(nb_labels, (1, 1))(x8)

    # Resize each compressed skip connection using bilinear interpolation.
    # This operation isn't built into Keras, so we use a LambdaLayer
    # which allows calling a Tensorflow operation.
    def resize_bilinear(images):
        import tensorflow as tf
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    r32 = Lambda(resize_bilinear)(c32)
    r16 = Lambda(resize_bilinear)(c16)
    r8 = Lambda(resize_bilinear)(c8)

    # Merge the three layers together using summation.
    m = Add()([r32, r16, r8])

    # Add softmax layer to get probabilities as output. We need to reshape
    # and then un-reshape because Keras expects input to softmax to
    # be 2D.
    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    fcn_model = Model(input_shape=input_shape, output=x)
    
    if fix_base:
        print('Fixing ResNet50 base_model layers')
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in fcn_model.layers:
            layer.trainable = False

    return dict(model=fcn_model,lr_mult=0.1)

