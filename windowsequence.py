import numpy as np

def keras_sequence():
    from keras.utils.data_utils import Sequence
    return Sequence

class WindowSequence(keras_sequence()):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.n_data = len(x_set)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, bj):
        bj_s,bj_e = bj*self.batch_size,(bj+1)*self.batch_size
        batch_x = self.x[min(self.n_data-1,bj_s):min(self.n_data,bj_e)]
        return batch_x
