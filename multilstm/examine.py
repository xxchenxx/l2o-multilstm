import pickle

a = pickle.load(open("mnist_multilstm_cl_rnnprop/rp.l2l-0", 'rb'))

b = pickle.load(open("mnist_multilstm_cl_rnnprop/rp.l2l-1000", 'rb'))

for key in a:
    if (isinstance(a[key], dict)):
        for key2 in a[key]:
            
            print(key + ":"  + key2, (a[key][key2] - b[key][key2]).abs().mean())