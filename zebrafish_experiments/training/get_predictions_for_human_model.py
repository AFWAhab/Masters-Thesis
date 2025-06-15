import h5py, os
import numpy as np
import pandas as pd
from scipy import stats
from keras.models import Model, load_model

def get_predictions(datadir):
    testfile = h5py.File(os.path.join(datadir, 'test (1).h5'), 'r')
    X_testhalflife, X_testpromoter, y_test, geneName_test = testfile['data'], testfile['promoter'], testfile['label'], testfile['geneName']
    params = {'datadir': datadir, 'batchsize': 2 ** 7, 'leftpos': 3000, 'rightpos': 13500, 'activationFxn': 'relu',
              'numFiltersConv1': 2 ** 7, 'filterLenConv1': 6, 'dilRate1': 1,
              'maxPool1': 30,
              'numconvlayers': {'numFiltersConv2': 2 ** 5, 'filterLenConv2': 9, 'dilRate2': 1, 'maxPool2': 10,
                                'numconvlayers1': {'numconvlayers2': 'two'}},
              'dense1': 2 ** 6, 'dropout1': 0.00099,
              'numdenselayers': {'layers': 'two', 'dense2': 2, 'dropout2': 0.01546}}

    # evaluate performance on test set using best learned model
    best_file = os.path.join(params['datadir'], 'bestparams_9_iteration_human_data.keras')
    model = load_model(best_file)
    print('Loaded results from:', best_file)
    leftpos = int(params['leftpos'])
    rightpos = int(params['rightpos'])
    X_testpromoterSubseq = X_testpromoter[:, leftpos:rightpos, :]
    predictions_test = model.predict([X_testpromoterSubseq, X_testhalflife], batch_size=64).flatten()
    #slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_test, y_test)
    #print('Test R^2 = %.3f' % r_value ** 2)
    df = pd.DataFrame(np.column_stack((geneName_test, predictions_test, y_test)), columns=['Gene', 'Pred', 'Actual'])

    print('Rows & Cols:', df.shape)
    df.to_csv(os.path.join(params['datadir'], 'bestparams_9_iteration_human_data.txt'), index=False, header=True, sep='\t')

datadir = "../paperData/pM10Kb_1KTest"
get_predictions(datadir)