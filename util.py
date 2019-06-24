# coding: utf-8

'''
Miscellaneous utitilies
Raymond Kwok
rmwkwok at gmail.com
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.metrics import log_loss, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

class CSNSStylish(mpl.rc_context):
    def __init__(self, **kwargs):
        mpl.rc_context.__init__(self)
        sns.set(**kwargs)

def showSummary(data):
    '''
    Use with pandas dataframe.
    Show columns description, their datatype, number of missing data,
    number of unique values and what they are (if less than ten)
    in a table
    
    Parameters
    ----------
    data: pandas dataframe
    '''
    with pd.option_context('display.max_rows', 100, 
                           'display.max_columns', 100, 
                           'float_format', lambda x: '%.1f'%x):
    
        print('#rows, #columns = ', data.shape)

        lsUnique = lambda c: '' if len(c.unique()) > 10 else\
                             ','.join(c.unique().astype(str))
        
        # 4 sections of information
        tmp1 = data.describe().transpose()
        tmp2 = pd.DataFrame({
                    'datatype': data.dtypes, 
                    '#nonMissingData': data.count()})
        tmp3 = pd.DataFrame({
                    '#uniqueValues': data.nunique(), 
                    'uniqueValues': data.apply(lambda c: lsUnique(c), axis=0)})
        tmp4 = data.head(10).transpose()

        # give each a section name
        tmp1.columns = pd.MultiIndex.from_tuples([('numeric stats', c) for c in tmp1.columns])
        tmp2.columns = pd.MultiIndex.from_tuples([('general', c) for c in tmp2.columns])
        tmp3.columns = pd.MultiIndex.from_tuples([('categoric stats', c) for c in tmp3.columns])
        tmp4.columns = pd.MultiIndex.from_tuples([('example data', c) for c in tmp4.columns])

        # join the sections and display them
        display(tmp2.join(tmp1).join(tmp3).join(tmp4).fillna(''))

def showCorr(data, **kwargs):
    '''
    Use with pandas dataframe.
    Plot heatmap for data's correlation matrix.
    
    Parameters
    ----------
    data: pandas dataframe
    **kwargs: pass to sns.set(**kwargs)
    '''
    
    tmp = data.corr()
    with CSNSStylish(**kwargs):
        fig, ax = plt.subplots(1,1,figsize=(len(tmp),len(tmp)))
        sns.heatmap(tmp, cmap="RdBu_r",
                    xticklabels=tmp.columns.values, 
                    yticklabels=tmp.index.values, 
                    annot=tmp, fmt='0.1f', ax=ax)
        ax.set_title('correlation matrix')
    plt.show()
    
def randColor(n):
    '''
    Generate a list of color
    
    Parameters
    ----------
    n: number of colors
    
    Output:
    ----------
    list of color tuples of RGB
    '''
    
    colors = []
    np.random.seed(2)
    while len(colors) < n:
        cand = np.random.randint(0,255,3)
        if any([ch<5 for color in colors for ch in np.abs(color-cand)]):
            continue
        colors.append(cand)
    return ['#%02x%02x%02x'%tuple(color) for color in colors]

def rootmeansquared(ary):
    return np.sqrt(np.mean(np.square(ary)))

def oversampling(trn_x, trn_y, smote_ratio=0.7):
    '''
    SMOTENC and RandomOverSampler are used to oversample
    the minority class to balance the dataset.
    
    SMOTENC adds new data by intrapolation.
    RandomOverSampler adds data by repeating existing data
    
    Details: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
    
    Parameters
    ----------
    trn_x: 
        Dataset, excluding label
    
    trn_y: 
        Label
    
    smote_ratio: float, default 0.7
        consider sample gap between the majority and minority be n,
        smote_ratio * n will be filled up by SMOTENC, and the rest by RandomOverSampler
    
    Output
    ----------
    (oversampled_x, oversampled_y): upsampled dataset
    '''
    if smote_ratio < 1:
        ros_x, smotenc_x, ros_y, smotenc_y = train_test_split(trn_x, trn_y, test_size = smote_ratio, 
                                                              random_state = 3, shuffle = True, stratify=trn_y)
    
        ros_x_2, ros_y_2 = RandomOverSampler(random_state=10).fit_resample(ros_x, ros_y)
    
    else:
        
        smotenc_x = trn_x
        smotenc_y = trn_y
        
        ros_x_2 = np.empty((0, trn_x.shape[1]))
        ros_y_2 = np.empty((0, ))
        
    smotenc_x_2, smotenc_y_2 = SMOTENC(trn_x.nunique()==2, random_state=12, k_neighbors=5, n_jobs=4)\
                                .fit_resample(smotenc_x, smotenc_y)
    
    oversampled_x = np.concatenate((ros_x_2,smotenc_x_2))
    oversampled_y = np.concatenate((ros_y_2,smotenc_y_2))
    
    print('sample size from %d to %d'%(len(trn_x), len(oversampled_x)))
    print('label ratio from %.2f to %.2f'%(sum(trn_y==1)/sum(trn_y==0), (sum(oversampled_y==1)/sum(oversampled_y==0) )))
    
    return oversampled_x, oversampled_y


# Helper function
def bookkeeping(name, trn_x, trn_y, tst_x, tst_y, result, gs):
    '''
    bookkeeping model performance
    
    Parameters
    ----------
    name: str, model name
    gs: gridsearch object
    trn_x, trn_y, tst_x, tst_y: training and test sets
    result: dict, bookkeeping the result in this dictionary
    '''
    
    tmp = pd.DataFrame(gs.cv_results_)
    bestlb = tmp.sort_values('rank_test_f1').iloc[0]
    bestll = tmp.sort_values('rank_test_neg_log_loss').iloc[0]
    
    lbpred = gs.predict(tst_x)
    fpr, tpr, thres = roc_curve(tst_y, lbpred)
    
    result[name] = {'lb_best': bestlb['params'],
                    'lb_trn_f1'  : f1_score(trn_y, gs.predict(trn_x)),
                    'lb_vld_f1'  : bestlb['mean_test_f1'],
                    'lb_tst_f1'  : f1_score(tst_y, lbpred),
                    'lb_tst_acc' : accuracy_score(tst_y, lbpred),
                    'lb_tst_pre' : precision_score(tst_y, lbpred),
                    'lb_tst_recall' : recall_score(tst_y, lbpred),
                    'lb_tst_auroc'  : roc_auc_score(tst_y, lbpred),
                    'lb_tst_fpr' : fpr,
                    'lb_tst_tpr' : tpr,
                    'lb_tst_thres'  : thres,
                    
                    'll_best': bestll['params'],
                    'll_trn' : log_loss(trn_y, gs.predict_proba(trn_x)),
                    'll_vld' : -bestll['mean_test_neg_log_loss'],
                    'll_tst' : log_loss(tst_y, gs.predict_proba(tst_x)),
                     
                    'gs' : gs}
    
def model(cls, param_grid, trn_x, trn_y):
    '''
    5-Fold CV Grid Search
    
    Parameters
    ----------
    cls: classifier
    param_grid: dict, pass to GridSearchCV() 
    trn_x, trn_y: training set
    
    Output
    ----------
    gs: Grid Search Object
    '''
    gs = GridSearchCV(cls, 
                      param_grid = param_grid,
                      scoring    = ['f1','neg_log_loss'],
                      n_jobs     = 5,
                      cv         = StratifiedKFold(5),
                      verbose    = 1,
                      refit      = 'f1',
                      return_train_score = True,
                     )
    gs.fit(trn_x, trn_y)
    return gs

def model_dt(cls, param_grid, trn_x, trn_y, early_stopping_rounds=100):
    '''
    5-Fold CV Grid Search for decision tree
    Train set will be further split into train and 
    valid sets for early stopping
    
    Parameters
    ----------
    cls: classifier
    param_grid: dict, pass to GridSearchCV() 
    trn_x, trn_y: training set
    
    Output
    ----------
    gs: Grid Search Object
    '''
    gs = GridSearchCV(cls, 
                      param_grid = param_grid,
                      scoring    = ['f1','neg_log_loss'],
                      n_jobs     = 5,
                      cv         = StratifiedKFold(5),
                      verbose    = 1,
                      refit      = 'f1',
                      return_train_score = True,
                     )
    _trn_x, _vld_x, _trn_y, _vld_y = train_test_split(trn_x, 
                                                      trn_y, 
                                                      test_size = 0.2, 
                                                      random_state = 3, shuffle = True, 
                                                      stratify=trn_y)
    
    gs.fit(_trn_x, _trn_y, 
           eval_set    =[(_vld_x, _vld_y)], 
           eval_metric ='logloss', 
           early_stopping_rounds = early_stopping_rounds,
           verbose = False)
    
    return gs

# Helper function - permutation test
# it permute the features one by one and input to model for prediction
# the difference in score between the permuted and unpermuted result is calculated
# the importance becomes the ratio of the difference to unpermuted result.
def permutationtest(cls, metric, trn_x, trn_y):
    '''
    Permutation test
    
    Begin with calculating std metric (f1 for label prediction,
    logloss for probability prediction) value for the training set
    
    Then it chooses one feature at a time, permute it, 
    then put through the classifier for getting a new metric value
    
    permuted importance = (std metric - new metric)/abs(std metric)
    
    The permuted importance measures how worse it is after permutation
    
    Parameters
    ----------
    cls: classifier
    metric: str, {'f1','ll'}, use f1 for label prediction, 
            use ll for probability prediction
    trn_x, trn_y: training set
    
    Output
    ----------
    permutationImp: list, permutation importance
    '''
    
    def neg_ll(*args, **kwargs):
        return -log_loss(*args, **kwargs)
    
    repeat = 2
    ncol = trn_x.shape[1]
    permutationImp  = [0]*ncol
    
    np.random.seed(3)
    if metric == 'f1':
        predfunc = cls.predict
        evalfunc = f1_score
    elif metric == 'll':
        predfunc = cls.predict_proba
        evalfunc = neg_ll
    else:
        return None
        
    std = evalfunc(trn_y, predfunc(trn_x))
    for i in range(ncol):
        for j in range(repeat):
            _trn_x = trn_x.copy()
            _trn_x[:,i] = np.random.permutation(_trn_x[:,i])
            diff = std-evalfunc(trn_y, predfunc(_trn_x))
            permutationImp[i] += max(0, diff/np.abs(std))/repeat
                
    return permutationImp



























