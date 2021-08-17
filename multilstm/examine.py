import pickle
import numpy as np

#a = pickle.load(open("mnist_multilstm_cl_rnnprop/rp.l2l-0", 'rb'))
a = pickle.load(open("init-2.pkl", 'rb'))
b = pickle.load(open("mnist_multilstm_cl_rnnprop/rp.l2l-0", 'rb'))

for key in a:
    if (isinstance(a[key], dict)):
        for key2 in a[key]:
            
            print(key + ":"  + key2, (np.abs(a[key][key2] - b[key][key2])).mean())