from __future__ import absolute_import, division, print_function
import sys, os, time
gettime = time.time

from pylib import *

from tilepredictor_util import *

from keras.callbacks import Callback

class ValidationCheckpoint(Callback):
    def __init__(self,val_monitor='val_loss',model_dir=None, mode='auto',
                 period=1,warmup=5,save_best_model=True,save_best_preds=True,
                 initial_monitor=None,initial_epoch=0,pid=None,verbose=0):
        super(ValidationCheckpoint, self).__init__()
        
        self.metrics      = {}

        self.pid          = pid or os.getpid()
        self.test_data    = []
        self.test_labs    = []
        self.test_ids     = []

        self.pred_outs    = []
        self.pred_labs    = []
        self.pred_prob    = []

        self.statemsg     = None
        self.val_monitor  = val_monitor

        if mode == 'auto':
            if any([s in val_monitor for s in ['fscore','acc']]):
                self.val_mode = 'max'
            else:
                self.val_mode = 'min'
        else:
            self.val_mode = mode
        
        self.val_func     =  np.max if self.val_mode=='max' else np.min
        self.val_best     = -np.inf if self.val_mode=='max' else np.inf
        self.epoch_best   = initial_epoch if initial_monitor else 0
        self.period       = period
        self.warmup       = warmup

        self.out_suffix   = 'iter{epoch:d}_%s{%s:.6f}'%(self.val_monitor,
                                                        self.val_monitor)
        self.out_suffix   = self.out_suffix + '_pid%d'%self.pid
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
        if val_cmp==monitor_value and val_cmp != self.val_best:        
            if epoch > self.epoch_best:
                new_best = True
                msg += 'New '
            self.val_best = monitor_value
            self.epoch_best = epoch
            
        msg += 'best %s value=%.6f'%(self.val_monitor,monitor_value)
        if verbose:
            print(msg)
        return new_best
            
    def update_ids(self,test_ids,verbose=1):
        """
        update_data(self,test_data,test_labs,test_ids)
        
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
        
    def on_train_begin(self, logs=None):
        logs = logs or {}
                        
    def on_train_end(self, logs=None):
        logs = logs or {}

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
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

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
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
        if self.statemsg:
            print('\n'+self.statemsg+'\n')
            self.statemsg = None
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_end_time  = gettime()
        etime = self.epoch_end_time-self.epoch_start_time
        statemsg = 'Epoch %05d: %.3f seconds processing time'%(epoch+1,etime)
        logs['cputime'] = etime
        if self.validation_data and (epoch % self.period) == 0:
            n_labs = len(self.test_labs)
            n_ids = len(self.test_ids)
            if n_labs==0:
                # self.test_labs,self.test_data = [],[]
                # for Xi,yi in self.validation_data:
                #     self.test_labs.append(np.argmax(yi))
                #     self.test_data.append(Xi)
                self.y_test = self.validation_data[1]
                self.test_labs = to_binary(self.y_test)
            elif n_ids!=0 and (self.y_test != self.validation_data[1]).any():
                print('\nValidation data changed between current and previous '
                      'epoch, test_ids no longer valid!')
                self.test_ids = np.arange(n_labs)
                n_ids = n_labs

            if n_ids==0:
                self.test_ids = np.arange(len(self.test_labs))

            test_ids  = self.test_ids
            test_data = self.validation_data[0]
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
                        
            mstr = ['loss=%9.6f'%logs['val_loss']]
            pred_metrics = compute_metrics(test_labs,self.pred_labs)
            for m in metrics_sort:
                val = pred_metrics[m]
                logs['val_'+m] = val
                self.metrics.setdefault(m,[])
                self.metrics[m].append(val)
                mstr.append('%s=%9.6f'%(m,val*100))
            statemsg += '\nValidation '+(', '.join(mstr))

            val_epoch = np.float32(logs[self.val_monitor])
            
            if not self.update_monitor(epoch,val_epoch,verbose=0):
                # current score doesn't beat the best, report status
                statemsg += '\nCurrent best %s=%.6f @ epoch %d'%(self.val_monitor,self.val_best,
                                                                 self.epoch_best)
            else:
                # new best score, update best and report status
                statemsg += '\nNew best %s=%.6f'%(self.val_monitor,self.val_best)
                if self.save_preds or self.save_model:            
                    if epoch < self.warmup:
                        statemsg += '\nEpoch %d < warmup steps (=%d), no outputs written.'%(epoch,self.warmup)
                    elif 'val_fscore' in logs and logs['val_fscore']==0:
                        statemsg += '\nIgnoring val_fscore=0.0, no outputs written.'
                    else:
                        fmtdict = {'epoch':epoch,self.val_monitor:self.val_best}
                        if self.save_model:
                            modelf = self.model_iterf.format(**fmtdict)
                            self.model.save(modelf)
                            statemsg += '\nSaved best model to %s'%modelf
                        
                        if self.save_preds:
                            predsf = self.preds_iterf.format(**fmtdict)                        
                            statemsg += '\nSaved best predictions to %s'%predsf
                            write_predictions(predsf, test_ids, test_labs,
                                              self.pred_labs, self.pred_prob,
                                              pred_metrics, fnratfpr=0.01)
        self.statemsg = statemsg
