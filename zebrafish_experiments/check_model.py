import h5py

f = h5py.File("paperData/pretrained_models/humanMedian_trainepoch.11-0.426.h5", 'r')
print(f.attrs.get('model_config'))