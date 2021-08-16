import torch
import sys
import numpy as np
path = sys.argv[1]
mask_path = sys.argv[2]
output = sys.argv[3]
model = torch.load(path, map_location="cpu")
mask = torch.load(mask_path, map_location="cpu")

new_model = {}
for key in model:
    
    print(key)
    shape = model[key].numpy().shape
    print(shape)
    if len(shape) > 3:
        new_model['resnet/' + key] = np.transpose(model[key].numpy(), (2,3,1,0))
    else:
        new_model['resnet/' + key] = model[key].numpy()
        
new_mask = {}
for key in mask:
    new_mask['resnet/' + key] = np.transpose(mask[key].numpy(), (2,3,1,0))


result = {'weight': new_model, 'mask': new_mask}
import pickle
pickle.dump(result, open(output, 'wb'))