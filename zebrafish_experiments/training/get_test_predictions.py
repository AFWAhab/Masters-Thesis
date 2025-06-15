import h5py, os
import numpy as np
import pandas as pd
from scipy import stats
from keras.models import Model, load_model

def get_predictions(datadir):
    testfile = h5py.File(os.path.join(datadir, 'zebrafish_test.hdf5'), 'r')
    X_testhalflife, X_testpromoter, y_test, geneName_test = testfile['data'], testfile['promoter'], testfile['detectionFlagInt'], \
        testfile['geneName']
    params = {'datadir': datadir, 'batchsize': 2 ** 5, 'leftpos': 7500, 'rightpos': 12500, 'activationFxn': 'relu',
              'numFiltersConv1': 2 ** 7, 'filterLenConv1': 6, 'dilRate1': 1,
              'maxPool1': 6,
              'numconvlayers': {'numFiltersConv2': 2 ** 8, 'filterLenConv2': 9, 'dilRate2': 1, 'maxPool2': 6,
                                'numconvlayers1': {'numFiltersConv3': 2 ** 9, 'filterLenConv3': 9, 'dilRate3': 1,
                                                   'maxPool3': 6,
                                                   'numconvlayers2': {'numFiltersConv4': 2 ** 9, 'filterLenConv4': 9,
                                                                      'dilRate4': 1, 'maxPool4': 4,
                                                                      'numconvlayers3': {'numconvlayers4': 'four'}}}},
              # 'numconvlayers': {'numFiltersConv2': 2 ** 5, 'filterLenConv2': 9, 'dilRate2': 1, 'maxPool2': 10,
              #                  'numconvlayers1': {'numconvlayers2': 'two'}},
              # 'numconvlayers': {'numconvlayers1': 'one'},
              'dense1': 2 ** 6, 'dropout1': 0.00099,
              'numdenselayers': {'layers': 'two', 'dense2': 2, 'dropout2': 0.01546}}
    # 'numdenselayers': {'layers': 'one'}}

    best_file = os.path.join(params['datadir'], 'human_hyperparameters/bestparams_7_iteration_binary.keras')
    model = load_model(best_file)
    print('Loaded results from:', best_file)
    leftpos = int(params['leftpos'])
    rightpos = int(params['rightpos'])
    X_testpromoterSubseq = X_testpromoter[:, leftpos:rightpos, :]
    #predictions_test = model.predict([X_testpromoterSubseq, X_testhalflife], batch_size=64).flatten()
    predictions_test = model.predict([X_testpromoterSubseq], batch_size=64).flatten()
    #slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_test, y_test)
    #print('Test R^2 = %.3f' % r_value ** 2)
    df = pd.DataFrame(np.column_stack((geneName_test, predictions_test, y_test)), columns=['Gene', 'Pred', 'Actual'])

    print('Rows & Cols:', df.shape)
    df.to_csv(os.path.join(params['datadir'], 'human_hyperparameters/predictions_7_iteration_binary_new.txt'), index=False, header=True, sep='\t')

datadir = "zebrafish_training"
get_predictions(datadir)