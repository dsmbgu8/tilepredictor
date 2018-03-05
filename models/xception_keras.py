from warnings import warn
_model_id = 'Xception'
MAX_BLOCK = 15 # xception max block=14
def model_init(input_shape,**kwargs):
    from keras.applications import Xception
    from keras import backend

    assert(backend.backend()=='tensorflow')
    assert(len(input_shape)==3 and input_shape[2]==3)
    backend.set_image_data_format('channels_last')

    fix_base = kwargs.pop('fix_base',True)
    if not fix_base:
        warn('%s model fix_base=False, training may take a long time'%_model_id)
    
    max_block = kwargs.pop('max_block',MAX_BLOCK) 

    print(_model_id+' input_shape: "%s"'%str((input_shape)))
    base_model = Xception(input_shape=input_shape, weights="imagenet",
                          include_top=False, pooling="avg")

    n_layers = len(base_model.layers)
    max_layer = kwargs.pop('max_layer',n_layers)
    max_layer = min(n_layers,max_layer) if max_layer>0 else n_layers+max_layer

    # pop any unwanted top layers
    n_pop = n_layers-max_layer
    if n_pop != 0:
        print('Popping %d layers'%n_pop)
        for i in range(n_pop):
            base_model.pop()
    
    if fix_base:
        print('Fixing %s base_model layers'%_model_id)
        # first: train only the top layers (which were randomly initialized)
        trainable = False # fix blocks until we hit max_block
        for i,layer in enumerate(base_model.layers):
            lname = layer.name
            if not trainable and max_block <= MAX_BLOCK:
                if lname.startswith('block'):                    
                    lid = int(lname.split('_')[0].replace('block',''))
                    if lid>max_block:
                        trainable = True
                        base_model.layers[i-1].trainable = trainable
                
            layer.trainable = trainable
        
    return dict(model=base_model,lr_mult=1.0)
