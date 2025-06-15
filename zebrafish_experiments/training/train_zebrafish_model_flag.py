import h5py, os
import numpy as np
import pandas as pd
from scipy import stats
from IPython.display import Image

from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, AdamW
from keras.layers import *
from keras.metrics import BinaryAccuracy
from keras.losses import BinaryCrossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from hyperopt import hp, STATUS_OK

# Results presented in the paper are the best of 10 independent trials, choosing the one that minimizes
# validation binary cross entropy loss.
# These results are not exactly those shown in paper due to variability in performance

global X_trainhalflife, X_trainpromoter, y_train, X_validhalflife, X_validpromoter, y_valid, X_testhalflife, X_testpromoter, y_test, geneName_test, params, best_file
global MODEL_NAME

def main(datadir):
    global X_trainhalflife, X_trainpromoter, y_train, X_validhalflife, X_validpromoter, y_valid, X_testhalflife, X_testpromoter, y_test, geneName_test, params
    params['datadir'] = datadir
    file = h5py.File('zebrafish.hdf5', 'r')
    X_halflife, X_promoter, y, geneName = file['halflifeData'], file['promoter'], file['detectionFlagInt'], file['geneName']

    num_samples = X_halflife.shape[0]
    indices = np.random.default_rng(seed=0).permutation(num_samples)
    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)

    train_indexes = indices[:train_split]
    val_indexes = indices[train_split:val_split]
    test_indexes = indices[val_split:]
    train_indexes.sort()
    val_indexes.sort()
    test_indexes.sort()

    X_trainhalflife, X_trainpromoter, y_train, geneName_train = X_halflife[train_indexes], X_promoter[train_indexes], y[
        train_indexes], geneName[train_indexes]
    X_validhalflife, X_validpromoter, y_valid, geneName_valid = X_halflife[val_indexes], X_promoter[val_indexes], y[
        val_indexes], geneName[val_indexes]
    X_testhalflife, X_testpromoter, y_test, geneName_test = X_halflife[test_indexes], X_promoter[test_indexes], y[
        test_indexes], geneName[test_indexes]

    # best hyperparams learned
    params = {'datadir': datadir, 'batchsize': 2 ** 7, 'leftpos': 3000, 'rightpos': 13500, 'activationFxn': 'relu',
              'numFiltersConv1': 2 ** 7, 'filterLenConv1': 6, 'dilRate1': 1,
              'maxPool1': 30,
              'numconvlayers': {'numFiltersConv2': 2 ** 5, 'filterLenConv2': 9, 'dilRate2': 1, 'maxPool2': 10,
                                'numconvlayers1': {'numconvlayers2': 'two'}},
              'dense1': 2 ** 6, 'dropout1': 0.00099,
              'numdenselayers': {'layers': 'two', 'dense2': 2, 'dropout2': 0.01546}}
    print("Using best identified hyperparameters from architecture search, these are:")
    print(params)
    results = objective(params)
    print("Best Validation BCE = %.3f" % results['loss'])


params = {
    'tuneMode': 1,
    'batchsize': 2 ** hp.quniform('batchsize', 5, 7, 1),
    'leftpos': hp.quniform('leftpos', 0, 10000, 500),
    'rightpos': hp.quniform('rightpos', 10000, 20000, 500),
    'activationFxn': 'relu',
    'numFiltersConv1': 2 ** hp.quniform('numFiltersConv1', 4, 7, 1),
    'filterLenConv1': hp.quniform('filterLenConv1', 1, 10, 1),
    'dilRate1': hp.quniform('dilRate1', 1, 4, 1),
    'maxPool1': hp.quniform('maxPool1', 5, 100, 5),  # 5, 100, 5),
    'numconvlayers': hp.choice('numconvlayers', [
        {
            'numconvlayers1': 'one'
        },
        {
            'numFiltersConv2': 2 ** hp.quniform('numFiltersConv2', 4, 7, 1),
            'filterLenConv2': hp.quniform('filterLenConv2', 1, 10, 1),
            'dilRate2': hp.quniform('dilRate2', 1, 4, 1),
            'maxPool2': hp.quniform('maxPool2', 5, 100, 5),
            'numconvlayers1': hp.choice('numconvlayers1', [
                {
                    'numconvlayers2': 'two'
                },
                {
                    'numFiltersConv3': 2 ** hp.quniform('numFiltersConv3', 4, 7, 1),
                    'filterLenConv3': hp.quniform('filterLenConv3', 1, 10, 1),
                    'dilRate3': hp.quniform('dilRate3', 1, 4, 1),
                    'maxPool3': hp.quniform('maxPool3', 5, 100, 5),
                    'numconvlayers2': hp.choice('numconvlayers2', [
                        {
                            'numconvlayers3': 'three'
                        },
                        {
                            'numFiltersConv4': 2 ** hp.quniform('numFiltersConv4', 4, 7, 1),
                            'filterLenConv4': hp.quniform('filterLenConv4', 1, 10, 1),
                            'dilRate4': hp.quniform('dilRate4', 1, 4, 1),
                            'maxPool4': hp.quniform('maxPool4', 5, 100, 5),
                            'numconvlayers3': 'four'
                        }])
                }])
        }]),
    'dense1': 2 ** hp.quniform('dense1', 1, 8, 1),
    'dropout1': hp.uniform('dropout1', 0, 1),
    'numdenselayers': hp.choice('numdenselayers', [
        {
            'layers': 'one'
        },
        {
            'layers': 'two',
            'dense2': 2 ** hp.quniform('dense2', 1, 8, 1),
            'dropout2': hp.uniform('dropout2', 0, 1)
        }
    ])
}


def objective(params):
    global best_file
    leftpos = int(params['leftpos'])
    rightpos = int(params['rightpos'])
    activationFxn = params['activationFxn']
    global X_trainhalflife, y_train
    X_trainpromoterSubseq = X_trainpromoter[:, leftpos:rightpos, :]
    X_validpromoterSubseq = X_validpromoter[:, leftpos:rightpos, :]
    halflifedata = Input(shape=(X_trainhalflife.shape[1:]), name='halflife')
    input_promoter = Input(shape=X_trainpromoterSubseq.shape[1:], name='promoter')

    bce = 1
    # defined architecture with best hyperparameters
    x = Conv1D(int(params['numFiltersConv1']), int(params['filterLenConv1']), dilation_rate=int(params['dilRate1']),
               padding='same', kernel_initializer='glorot_normal',
               activation=activationFxn)(input_promoter)
    x = MaxPooling1D(int(params['maxPool1']))(x)

    if params['numconvlayers']['numconvlayers1'] != 'one':
        maxPool2 = int(params['numconvlayers']['maxPool2'])
        x = Conv1D(int(params['numconvlayers']['numFiltersConv2']), int(params['numconvlayers']['filterLenConv2']),
                   dilation_rate=int(params['numconvlayers']['dilRate2']), padding='same',
                   kernel_initializer='glorot_normal', activation=activationFxn)(x)  # [2, 3, 4, 5, 6, 7, 8, 9, 10]
        x = MaxPooling1D(maxPool2)(x)
        if params['numconvlayers']['numconvlayers1']['numconvlayers2'] != 'two':
            maxPool3 = int(params['numconvlayers']['numconvlayers1']['maxPool3'])
            x = Conv1D(int(params['numconvlayers']['numconvlayers1']['numFiltersConv3']),
                       int(params['numconvlayers']['numconvlayers1']['filterLenConv3']),
                       dilation_rate=int(params['numconvlayers']['numconvlayers1']['dilRate3']), padding='same',
                       kernel_initializer='glorot_normal', activation=activationFxn)(x)  # [2, 3, 4, 5]
            x = MaxPooling1D(maxPool3)(x)
            if params['numconvlayers']['numconvlayers1']['numconvlayers2']['numconvlayers3'] != 'three':
                maxPool4 = int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['maxPool4'])
                x = Conv1D(int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['numFiltersConv4']),
                           int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['filterLenConv4']),
                           dilation_rate=int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['dilRate4']),
                           padding='same', kernel_initializer='glorot_normal', activation=activationFxn)(
                    x)  # [2, 3, 4, 5]
                x = MaxPooling1D(maxPool4)(x)

    x = Flatten()(x)
    x = Concatenate()([x, halflifedata])
    x = Dense(int(params['dense1']))(x)
    x = Activation(activationFxn)(x)
    x = Dropout(params['dropout1'])(x)
    if params['numdenselayers']['layers'] == 'two':
        x = Dense(int(params['numdenselayers']['dense2']))(x)
        x = Activation(activationFxn)(x)
        x = Dropout(params['numdenselayers']['dropout2'])(x)
    main_output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_promoter, halflifedata], outputs=[main_output])
    #model = Model(inputs=[input_promoter], outputs=[main_output])
    bce_loss = BinaryCrossentropy(axis=0, from_logits=False)
    #model.compile(SGD(learning_rate=0.0005, momentum=0.9), loss=bce_loss, metrics=[BinaryAccuracy()]) # SGD optimizer from paper
    #model.compile(Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.90, epsilon=1e-08, weight_decay=0.0), loss=bce_loss, metrics=[BinaryAccuracy()]) # adam optimizer from paper
    #model.compile(AdamW(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay=0.004), loss=bce_loss, metrics=[BinaryAccuracy()]) # default AdamW optimizer
    model.compile(Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.2),loss=bce_loss, metrics=[BinaryAccuracy()])

    print(model.summary())
    modelfile = os.path.join(params['datadir'], 'plotted_model.png')
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=modelfile)

    # train model on training set and eval on 1K validation set
    MODEL_NAME = "bestparams_5_new_median_adam_binary_hf"
    check_cb = ModelCheckpoint(os.path.join(params['datadir'], MODEL_NAME + ".keras"), monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min')
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='min')
    result = model.fit([X_trainpromoterSubseq, X_trainhalflife], y_train, batch_size=int(params['batchsize']),
                       shuffle="batch", epochs=30,
                       validation_data=[[X_validpromoterSubseq, X_validhalflife], y_valid],
                       callbacks=[earlystop_cb, check_cb])
    bce_history = result.history['val_loss']
    bce = min(bce_history)

    # evaluate performance on test set using best learned model
    best_file = os.path.join(params['datadir'], MODEL_NAME + ".keras")
    model = load_model(best_file)
    print('Loaded results from:', best_file)
    X_testpromoterSubseq = X_testpromoter[:, leftpos:rightpos, :]
    predictions_test = model.predict([X_testpromoterSubseq, X_testhalflife], batch_size=64).flatten()
    m = BinaryAccuracy()
    m.update_state(y_test, predictions_test)
    print("Test accuracy:", m.result())
    #predictions_test = model.predict([X_testpromoterSubseq], batch_size=64).flatten()
    #slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_test, y_test)
    #print('Test R^2 = %.3f' % r_value ** 2)
    df = pd.DataFrame(np.column_stack((geneName_test, predictions_test, y_test)), columns=['Gene', 'Pred', 'Actual'])

    print('Rows & Cols:', df.shape)
    df.to_csv(os.path.join(params['datadir'], MODEL_NAME + ".txt"), index=False, header=True, sep='\t')

    return {'loss': bce, 'status': STATUS_OK}


datadir = "zebrafish_training_NEW_binary"
main(datadir=datadir)

# Matches FigS2A
Image(retina=True, filename=os.path.join(datadir, 'plotted_model.png'))