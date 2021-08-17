import pickle

a = pickle.load(open("mnist_multilstm_cl_rnnprop/rp.l2l-0", 'rb'))

b = pickle.load(open("mnist_multilstm_cl_rnnprop/rp.l2l-1000", 'rb'))

for key in a:
    print(a[key] - b[key])