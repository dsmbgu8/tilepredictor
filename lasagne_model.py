def update_base_outputs(base_model,nb_classes,nb_hidden,obj_lambda2,max_norm):
    pass

def load_model(*args,**kwargs):
    pass

def lasagne_model(model_base, model_flavor, state_dir, **params):
    import theano
    theano.config.floatX = 'float32'
    
    from theano import function as tfunction, shared as tshared
    from theano.tensor import tensor4, imatrix, nnet
    from theano.tensor import grad as Tgrad, mean as Tmean, reshape as Treshape

    from lasagne.utils import floatX
    from lasagne.updates import adam as lasagne_adam, total_norm_constraint
    from lasagne.layers import get_output as ll_output, \
        get_all_params as ll_all_params

    verbose     = params.get('verbose',False)
    overwrite   = params.get('overwrite',True)

    sym_x = tensor4() # [nbatch,imgchan,imgrows,imgcols] dims
    sym_y = imatrix() # one-hot vector of [nb_class x 1] dims

    l_A_net      = model_base['A_net']
    l_transform  = model_base['transform']
    l_out        = model_base['net_out']        
    output_train = ll_output(l_out, sym_x, deterministic=False)
    output_shape = (-1, l_out.shape[1]) # nb_classes = l_out.shape[1]
    output_flat  = treshape(output_train, output_shape)
    output_loss  = nnet.categorical_crossentropy
    output_cost  = tmean(output_loss(output_flat+tol,sym_y.flatten()))

    trainable_params = ll_all_params(l_out, trainable=True)

    all_grads = tgrad(output_cost, trainable_params)
    updates, norm = total_norm_constraint(all_grads, max_norm=max_norm,
                                          return_norm=True)

    shared_lr = tshared(floatX(init_lr))
    updates = lasagne_adam(updates, trainable_params, learning_rate=shared_lr,
                           beta_1=beta_1, beta_2=beta_2, epsilon=tol)

    model_train = tfunction([sym_x, sym_y], [output_cost, output_train, norm],
                            updates=updates)

    output_eval, l_A_eval = ll_output([l_out, l_A_net], sym_x,
                                      deterministic=True)
    model_eval = tfunction([sym_x], [output_eval.reshape(output_shape),
                                     l_A_eval.reshape(output_shape)])
    model_batch = lambda X,y: model_train(X, int32(y))[0]
    model_pred  = lambda X: model_eval(X)[0]
    model_xform = lambda X: layer_output(X,l_transform)
    model_save  = lambda outf: save_all_weights(l_out,outf,overwrite=overwrite)
    model_load  = lambda weightf: load_all_weights(l_out,weightf)
    model_compile = lambda: None
    return dict(package='lasagne',backend='theano',flavor=model_flavor,
                base=model_base,batch=model_batch,predict=model_pred,
                transform=model_xform,save=model_save,load=model_load,
                compile=model_compile,state_dir=state_dir,params=params)
