from __future__ import absolute_import, division, print_function

import sys
import numpy as np

_print_version=False
if _print_version:
    print_version()
    
def print_version():
    import keras
    print('using keras.__version__',keras.__version__)
    raw_input()

def load_model(modelf,**kwargs):
    from keras.models import clone_model as _clone_model, Sequential, \
        load_model as _load_model
    # import these since keras doesn't load them by default
    #from AdamW import AdamW
    #from SGDW import SGDW
    custom_objects = kwargs.pop('custom_objects',{})
    #custom_objects['AdamW'] = AdamW
    #custom_objects['SGDW'] = SGDW
    
    flatten = kwargs.pop('flatten',False)
    model = _load_model(modelf,custom_objects=custom_objects,**kwargs)
    if flatten and model.layers[0].name.startswith('sequential_'):
        _model = _clone_model(model.layers[0])
        for layer in model.layers[1:]:
            _model.add(layer)
        model = _model
    return model

def backend():
    from keras import backend as _backend
    return _backend

def update_base_outputs(model_base,output_shape,optparam,hidden_type='fc'):
    from keras.models import Model as KerasModel, Sequential
    from keras.layers import Dense, Input
    from keras.regularizers import l2 as activity_l2
    from keras.constraints import max_norm as max_norm_constraint
    
    n_hidden,n_classes = output_shape
    print('output_shape: "%s"'%str((output_shape)))
    print('Adding %d x %d relu hidden + softmax output layer'%(n_hidden,
                                                               n_classes))

    obj_lambda2 = optparam.get('obj_lambda2',0.0025)
    obj_param = dict(activity_regularizer=activity_l2(obj_lambda2))

    max_norm=optparam.get('max_norm',np.inf)
    if max_norm!=np.inf:
        obj_param['kernel_constraint']=max_norm_constraint(max_norm)

    model_input_shape = model_base.layers[0].input_shape[0]
    print('model_input_shape=%s'%str(model_input_shape))
    
    if hidden_type=='fc':
        hidden_layer = Dense(n_hidden, activation='relu')
    elif hidden_type=='none':
        hidden_layer = None
    else:
        print('Unknown hidden_type "%s", using "fc"'%hidden_type)
        hidden_layer = Dense(n_hidden, activation='relu')

    output_layer = Dense(n_classes, activation='softmax', **obj_param)

    mclass = model_base.__class__.__name__
    if 1 or mclass == 'Sequential':
        print("Using Sequential model")
        model = Sequential()
        model.add(model_base)        
        if hidden_layer:
            model.add(hidden_layer)
        model.add(output_layer)
    else:
        print("Using functional API")
        inputs = model_base.layers[0].get_input_at(0)
        outputs = model_base.layers[-1].get_output_at(0)
        if hidden_layer:
            outputs = hidden_layer(outputs) 
        preds = output_layer(outputs)
        model = KerasModel(inputs=inputs, outputs=preds)

    model.n_base_layers = len(model_base.layers)
    model.n_top_layers = len(model.layers)-model.n_base_layers
    
    return model

def model_transform(X,model,layer=0):
    from keras.backend import backend as _backend
    input_layer = model.layers[0]
    layer = model.layers[l]
    func = _backend.function([input_layer.get_input_at(0),
                              _backend.learning_phase()],
                             [layer.get_output_at(0)])
    return func([X,0])[0]

def model_init(model_base, model_flavor, state_dir, optparam, **params):    
    from keras import backend as _backend
    from keras.models import load_model

    verbose     = params.get('verbose',False)
    overwrite   = params.pop('overwrite',True)

    print('Initialzing optimizer')
    optclass = optparam.get('optclass','Nadam')
    print('Using',optclass,'optimizer')
    optimparams = dict(lr=optparam['lr_min'])
    optkeys = []
    if optclass=='Nadam':
        from keras.optimizers import Nadam as Optimizer
        optkeys = ['beta_1','beta_2']
        optimparams['schedule_decay']=optparam['weight_decay']
        optimparams['epsilon']=optparam['tol']
    elif optclass=='AdamW':
        from AdamW import AdamW as Optimizer
        optkeys = ['beta_1','beta_2','weight_decay','decay']
        optimparams['epsilon']=optparam['tol']        
    elif optclass=='SGD':
        from keras.optimizers import SGD as Optimizer
        optkeys = ['momentum','decay','nesterov']        
    elif optclass=='SGDW':
        from SGDW import SGDW as Optimizer
        optkeys = ['momentum', 'weight_decay','decay','nesterov']
        
    for key in optkeys:
        optimparams[key] = optparam[key]
        
    params.setdefault('loss','categorical_crossentropy')
    params['optimizer'] = Optimizer(**optimparams)
    params['optimizer_class'] = Optimizer
    params['optimizer_config'] = optimparams
    
    print('Initialzing model functions')
    model_backend = _backend.backend()

    if model_backend == 'tensorflow':        
        import tensorflow as tf
        if 0:
            from keras.backend.tensorflow_backend import get_session,set_session
            try:
                session = get_session()
                session._config.gpu_options.allow_growth = True
                set_session(session)
            except Exception as err: 
                print(sys.exc_info())

            try:
                run_opts = tf.RunOptions()
                run_opts.report_tensor_allocations_upon_oom=True
                params['options'] = run_opts            	
            except Exception as err: 
                print(sys.exc_info(),err)

    
    model_xform = lambda X,l=0: model_transform(X,model_base,l)
    model_pred  = lambda X,batch_size=32: model_base.predict(X,verbose=verbose,
                                                             batch_size=batch_size)
    model_batch = lambda X,y: model_base.train_on_batch(X,y)
    #model_save  = lambda weightf: model_base.save_weights(weightf,
    #                                                      overwrite=overwrite)

    model_load  = lambda modelf: load_model(modelf)
    return dict(package='keras',backend=model_backend,flavor=model_flavor,
                base=model_base,batch=model_batch,predict=model_pred,
                transform=model_xform,load_base=model_load,
                state_dir=state_dir,params=params)
