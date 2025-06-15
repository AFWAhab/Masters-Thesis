import h5py, os
import numpy as np
import pandas as pd
from scipy import stats
from keras.models import Model, load_model

def get_predictions(datadir):
    MODEL_NAME = "bestparams_3_new_embryolabels_adam"

    file = h5py.File('zebrafish.hdf5', 'r')

    X_halflife, X_promoter, y, geneName = file['halflifeData'], file['promoter'], file['embryoLabels'], file['geneName']

    num_samples = X_halflife.shape[0]
    indices = np.random.default_rng(seed=0).permutation(num_samples)
    val_split = int(0.9 * num_samples)
    test_indexes = indices[val_split:]
    test_indexes.sort()

    X_testhalflife, X_testpromoter, y_test, geneName_test = X_halflife[test_indexes], X_promoter[test_indexes], y[test_indexes], geneName[test_indexes]

    #X_testhalflife, X_testpromoter, y_test_binary, geneName_test = testfile['data'], testfile['promoter'], testfile['detectionFlagInt'], testfile['geneName']
    #X_testhalflife, X_testpromoter, geneName_test = testfile['data'], testfile['promoter'], testfile['geneName']
    #y_test = testfile['embryoLabels']
    params = {'datadir': datadir, 'batchsize': 2 ** 7, 'leftpos': 3000, 'rightpos': 13500, 'activationFxn': 'relu',
              'numFiltersConv1': 2 ** 7, 'filterLenConv1': 6, 'dilRate1': 1,
              'maxPool1': 30,
              'numconvlayers': {'numFiltersConv2': 2 ** 5, 'filterLenConv2': 9, 'dilRate2': 1, 'maxPool2': 10,
                                'numconvlayers1': {'numconvlayers2': 'two'}},
              'dense1': 2 ** 6, 'dropout1': 0.00099,
              'numdenselayers': {'layers': 'two', 'dense2': 2, 'dropout2': 0.01546}}
    # 'numdenselayers': {'layers': 'one'}}

    best_file = os.path.join(params['datadir'], MODEL_NAME + ".keras")
    model = load_model(best_file)
    print('Loaded results from:', best_file)
    leftpos = int(params['leftpos'])
    rightpos = int(params['rightpos'])
    X_testpromoterSubseq = X_testpromoter[:, leftpos:rightpos, :]
    predictions_test = model.predict([X_testpromoterSubseq, X_testhalflife], batch_size=64).flatten()
    #predictions_test = model.predict([X_testpromoterSubseq], batch_size=64).flatten()
    #slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_test, y_test)
    #print('Test R^2 = %.3f' % r_value ** 2)
    df = pd.DataFrame(np.column_stack((geneName_test, predictions_test, y_test)), columns=['Gene', 'Pred', 'Actual'])

    print('Rows & Cols:', df.shape)
    df.to_csv(os.path.join(params['datadir'], MODEL_NAME + ".txt"), index=False, header=True, sep='\t')

#datadir = "zebrafish_training"
datadir = "zebrafish_training_NEW"
get_predictions(datadir)