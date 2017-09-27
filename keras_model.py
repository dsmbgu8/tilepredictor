import numpy as np

from keras.models import load_model

def update_base_outputs(base_model,nb_classes,nb_hidden,obj_lambda2,max_norm):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2 as activity_l2
    from keras.constraints import max_norm as max_norm_constraint

    print('Adding %d x %d relu hidden + softmax output layer'%(nb_hidden,
                                                               nb_classes))

    outparam = dict(activity_regularizer=activity_l2(obj_lambda2))
    if max_norm!=np.inf:
        outparam['kernel_constraint']=max_norm_constraint(max_norm)
    
    model = Sequential()
    model.add(base_model)
    model.add(Dense(nb_hidden, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax',**outparam))

    model.nb_base_layers = len(base_model.layers)
    model.nb_top_layers = len(model.layers)-model.nb_base_layers
    
    return model

def model_init(model_base, model_flavor, state_dir, optkw, **params):
    from keras import backend
    #from keras.optimizers import Adam as Optimizer
    #optparams   = dict(lr=optkw['init_lr'],
    #                   beta_1=optkw['beta_1'],
    #                   beta_2=optkw['beta_2'],
    #                   decay=optkw['weight_decay'],
    #                   epsilon=optkw['tol'])    
    from keras.optimizers import Nadam as Optimizer
    optparams   = dict(lr=optkw['init_lr'],
                       beta_1=optkw['beta_1'],
                       beta_2=optkw['beta_2'],
                       schedule_decay=optkw['weight_decay'],
                       epsilon=optkw['tol'])

    verbose     = params.get('verbose',False)
    overwrite   = params.pop('overwrite',True)
    xform_func  = backend.function([model_base.layers[0].input,
                                    backend.learning_phase()],
                                   [model_base.layers[0].output])

    optimizer = Optimizer(**optparams)
    loss = params.pop('loss','categorical_crossentropy')

    model_xform = lambda X: xform_func([X])[0]
    model_pred  = lambda X: model_base.predict(X,verbose=verbose)
    model_batch = lambda X,y: model_base.train_on_batch(X,y)
    model_save  = lambda outf: model_base.save_weights(outf,overwrite=overwrite)
    #model_load  = lambda weightf: model_base.load_weights(weightf)
    model_load  = lambda weightf: load_model(weightf)
    model_compile = lambda: model_base.compile(optimizer=optimizer,loss=loss,
                                               **params)
    
    return dict(package='keras',backend=backend.backend(),flavor=model_flavor,
                base=model_base,batch=model_batch,predict=model_pred,
                transform=model_xform,save=model_save,load=model_load,
                compile=model_compile,state_dir=state_dir,params=params)
