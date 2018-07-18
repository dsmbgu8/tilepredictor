from __future__ import absolute_import, division, print_function
import sys, os, time
gettime = time.time

from pylib import *

from tilepredictor_util import *

metrics_sort = ['precision','recall','fscore']

def keras_callback():
    from keras.callbacks import Callback
    return Callback
    
class ValidationCheckpoint(keras_callback()):
    def __init__(self,val_monitor='val_loss',model_dir=None,mode='auto',
                 test_ids=[],test_period=1,save_period=None,initial_epoch=0,
                 initial_monitor=None,warmup_epoch=5,max_vanish=5,
                 save_best_model=True,save_best_preds=True,
                 pid=None,verbose=0):
        super(ValidationCheckpoint, self).__init__()
        
        self.metrics      = dict([(m,[]) for m in metrics_sort])

        self.val_monitor  = val_monitor
        self.pid          = pid or os.getpid()
        self.test_data    = []
        self.test_labs    = []
        self.test_ids     = test_ids
        self.y_test       = []
        
        self.pred_outs    = []
        self.pred_labs    = []
        self.pred_prob    = []

        self.state_msg    = None

        if mode == 'auto':
            if any([s in val_monitor for s in ['fscore','acc']]):
                self.val_mode = 'max'
            elif any([s in val_monitor for s in ['loss','error']]):
                self.val_mode = 'min'
            else:
                raise Exception('Unable to determine mode for val_monitor='+val_monitor)
        else:
            self.val_mode = mode
        
        self.val_func     =  np.max if self.val_mode=='max' else np.min
        self.val_init     = -np.inf if self.val_mode=='max' else np.inf
        self.val_best     = self.val_init
        self.epoch_best   = initial_epoch if initial_monitor else 0

        self.val_prev     = self.val_init
        self.val_log      = []
        self.epoch_prev   = self.epoch_best
        
        self.test_period  = test_period
        self.save_period  = save_period

        self.warmup       = warmup_epoch
        self.max_vanish   = max_vanish
        self.epoch_vanish = 0
        
        self.out_suffix   = 'iter{epoch:d}_%s{%s:.6f}'%(self.val_monitor,
                                                        self.val_monitor)
        self.out_suffix   = self.out_suffix + '_pid%d'%self.pid
        self.best_suffix  = 'best_pid%d'%self.pid
        self.save_preds   = save_best_preds
        self.save_model   = save_best_model
        self.verbose      = verbose

        if (save_best_preds or save_best_model) and \
           not (model_dir and pathexists(model_dir)):
            print('model_dir "%s" not found.'%str((model_dir)),
                  '\nmust specify valid model_dir to save preds or model.',
                  '\nno output will be saved.')

        else:
            # save output files for (e.g.,) "val_loss" monitor as "loss"
            self.preds_iterf = pathjoin(model_dir,'preds_'+self.out_suffix+'.txt')
            self.model_iterf = pathjoin(model_dir,'model_'+self.out_suffix+'.h5')        

            # keep symlink to best model+preds
            self.preds_bestf = pathjoin(model_dir,'preds_'+self.best_suffix+'.txt')
            self.model_bestf = pathjoin(model_dir,'model_'+self.best_suffix+'.h5')        

            # only save latest set of mispredictions
            self.mispred_dataf = pathjoin(model_dir,'mispreds_pid%d.h5'%self.pid)        
            
        if initial_monitor:            
            self.update_monitor(initial_epoch,initial_monitor,verbose=0)
            
        self.debug_batch = 0
        self.debug_epoch = 1
            
    def update_monitor(self,epoch,monitor_value,verbose=1):
        val_cmp = self.val_func([self.val_best,monitor_value])
        msg = 'Epoch %05d: '%epoch
        new_best = False
        if val_cmp == monitor_value and val_cmp != self.val_best:
            if epoch >= self.epoch_best:
                new_best = True
                msg += 'New '
            self.val_prev = self.val_best
            self.epoch_prev = self.epoch_best
            self.val_best = monitor_value
            self.epoch_best = epoch
            
        msg += 'best %s value=%.6f'%(self.val_monitor,monitor_value)
        if epoch < self.warmup:
            msg += '(epoch < warmup epochs, not stored)'
            self.val_best = self.val_prev
            self.epoch_best = self.epoch_prev
        
        if verbose:
            print(msg)

        return new_best

    def update_data(self,validation_data,validation_labs,verbose=1):
        self.test_data = validation_data
        self.y_test = validation_labs
        if verbose:
            print('Updated validation checkpoint with '
                  '%d new test samples'%len(self.test_data))

    def update_ids(self,test_ids,verbose=1):
        """
        update_ids(self,test_ids)
        
        Summary: updates test data for model validation,
                 including test_ids for prediction file
        
        Arguments:
        - self: self
        - test_data: test_data
        - test_labs: test_labs
        - test_ids: test_ids
        
        Keyword Arguments:
        None
        
        Output:
        None        
        """
        
        self.test_ids = test_ids
        self.test_labs = [] # invalidate test_labs to update in on_epoch_end
        if verbose:
            print('Updated validation checkpoint with '
                  '%d new test_ids'%len(self.test_ids))

        #self.debug_batch = 0
        self.debug_epoch = 1

    def collect_predictions(self):
        return self.pred_labs

    def print_state(self,reset=True):
        if self.state_msg:
            print('\n'+self.state_msg+'\n')
            if reset:
                self.state_msg = None
            
    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.train_start_time = gettime()
        self.print_state()
            
    def on_train_end(self, logs=None):
        logs = logs or {}
        self.train_stop_time = gettime()
        self.print_state()

    # def on_batch_begin(self, batch, logs=None):
        # logs = logs or {}
        # self.batch_start_time = gettime()        
        # if self.debug_batch:
        #     print('\non_batch_begin:')
        #     print('dir(self): "%s"'%str((dir(self))))
        #     print('logs:',logs)
        #     print('X_batch.shape:',batch[0].shape)
        #     print('y_batch.shape:',batch[1].shape)            
        #     print('band_stats(X_batch):')        
        #     band_stats(batch[0],verbose=1)
        #     print('class_stats(y_batch):')                    
        #     class_stats(batch[1],verbose=1)

    # def on_batch_end(self, batch, logs=None):
        # logs = logs or {}
        # self.batch_stop_time = gettime()
        # if self.debug_batch:
        #     print('\non_batch_end:')
        #     print('dir(self): "%s"'%str((dir(self))))            
        #     print('logs:',logs)
        #     print('X_batch.shape:',batch[0].shape)
        #     print('y_batch.shape:',batch[1].shape)            
        #     print('band_stats(X_batch):')        
        #     band_stats(batch[0],verbose=1)
        #     print('class_stats(y_batch):')                    
        #     class_stats(batch[1],verbose=1)
        #     self.debug_batch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_start_time = gettime()
        self.print_state()
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_end_time  = gettime()
        etime = self.epoch_end_time-self.epoch_start_time
        state_msg = 'Epoch %05d: %.3f seconds processing time'%(epoch+1,etime)
        logs['cputime'] = etime
        n_data = len(self.test_data)
        if not self.validation_data and n_data==0:
            warn('no validation data for validation checkpoint!')
        elif (epoch % self.test_period) == 0:
            n_labs = len(self.test_labs)
            n_ids = len(self.test_ids)
            if n_labs==0:
                # self.test_labs,self.test_data = [],[]
                # for Xi,yi in self.validation_data:
                #     self.test_labs.append(np.argmax(yi))
                #     self.test_data.append(Xi)
                if len(self.y_test)==0:
                    if self.validation_data:
                        self.y_test = self.validation_data[1]
                    else:
                        warn('no validation labs for validation checkpoint!')
                        return
                
                self.test_labs = to_binary(self.y_test)
                n_labs = len(self.test_labs)
            elif n_ids!=0 and self.validation_data and \
                 (self.y_test != self.validation_data[1]).any():
                print('\nValidation data changed between current and previous '
                      'epoch, test_ids no longer valid!')
                self.test_ids = np.arange(n_labs)
                n_ids = n_labs

            if n_ids==0:
                self.test_ids = np.arange(n_labs)

            test_ids  = self.test_ids
            test_data = self.test_data if len(self.test_data)==n_labs else \
                        self.validation_data[0]
            test_labs = self.test_labs
            pred_dict = compute_predictions(self.model,test_data)
            self.pred_labs = pred_dict['pred_labs']
            self.pred_prob = pred_dict['pred_prob']
            self.pred_outs = pred_dict['pred_outs']
            if self.debug_epoch:
                test_labs_classes = class_stats(test_labs)
                pred_labs_classes = class_stats(self.pred_labs)
                print('\nvalidation_checkpoint epoch "%s" summary:'%str((epoch)))
                print('test_ids.shape: "%s"'%str((test_ids.shape)))
                print('test_data.shape: "%s"'%str((test_data.shape)))
                print('test_labs.shape: "%s"'%str((test_labs.shape)))
                print('test_labs classes (pos,neg): "%s"'%str((test_labs_classes)))
                print('pred_outs.shape: "%s"'%str((self.pred_outs.shape)))
                print('pred_labs.shape: "%s"'%str((self.pred_labs.shape)))
                print('pred_labs classes (pos,neg): "%s"'%str((pred_labs_classes)))
                print('unique pred_probs: %d of %d'%(len(np.unique(self.pred_prob)),
                                                     len(self.pred_prob)))
                self.debug_epoch += 1
                if self.debug_epoch > 2:
                    self.debug_epoch = 0

            val_epoch = np.float32(logs[self.val_monitor])
            self.val_log.append(val_epoch)
            mstr = ['%s=%9.6f'%(self.val_monitor,val_epoch)]
            pred_metrics = compute_metrics(test_labs,self.pred_labs)
            for m in metrics_sort:
                val = pred_metrics[m]
                logs['val_'+m] = val
                self.metrics[m].append(val)
                mstr.append('%s=%9.6f'%(m,val*100))
            state_msg += '\nValidation '+(', '.join(mstr))

            fmtdict = {'epoch':epoch,self.val_monitor:val_epoch}                    

            new_best = self.update_monitor(epoch,val_epoch,verbose=0)
            if epoch==0 or epoch < self.warmup:
                state_msg += '\nEpoch %d < warmup steps (=%d), no outputs written.'%(epoch,self.warmup)
            elif 'val_fscore' in logs and logs['val_fscore']==0:
                state_msg += '\nIgnoring val_fscore=0.0, no outputs written.'            
            elif new_best:
                # new best score, update best and report status
                state_msg += '\nNew best %s=%.6f'%(self.val_monitor,self.val_best)

                if self.save_model:
                    modelf = self.model_iterf.format(**fmtdict)
                    self.model.save(modelf)
                    update_symlink(modelf,self.model_bestf)
                    state_msg += '\nSaved best model to %s'%modelf

                if self.save_preds:
                    predsf = self.preds_iterf.format(**fmtdict)                        
                    write_predictions(predsf, test_ids, test_labs,
                                      self.pred_labs, self.pred_prob,
                                      pred_metrics, fnratfpr=0.01)
                    update_symlink(predsf,self.preds_bestf)
                    state_msg += '\nSaved best predictions to %s'%predsf

            elif (self.save_period > 0) and (epoch % self.save_period)==0:
                # periodic save
                state_msg += '\nPeriodic save triggered, %s=%.6f'%(self.val_monitor,val_epoch)                
                if self.save_model:
                    modelf = self.model_iterf.format(**fmtdict)
                    self.model.save(modelf)
                    state_msg += '\nSaved periodic model to %s'%modelf

                if self.save_preds:
                    predsf = self.preds_iterf.format(**fmtdict)                        
                    state_msg += '\nSaved periodic predictions to %s'%predsf
                    write_predictions(predsf, test_ids, test_labs,
                                      self.pred_labs, self.pred_prob,
                                      pred_metrics, fnratfpr=0.01)
                    
            else: # not new_best
                # current score doesn't beat the best, report status
                state_msg += '\nCurrent best %s=%.6f @ epoch %d'%(self.val_monitor,self.val_best,
                                                                  self.epoch_best)
                if self.epoch_best != 0 and val_epoch==0.0:
                    if self.epoch_vanish == self.max_vanish:
                        state_msg += 'Epoch %d: gradient vanished for max_vanish (=%d) consecutive epochs, terminating training'%(epoch,self.max_vanish)
                        self.model.stop_training = True
                    self.epoch_vanish = self.epoch_vanish + 1                    
                else:
                    self.epoch_vanish = 0
                    
        self.state_msg = state_msg
