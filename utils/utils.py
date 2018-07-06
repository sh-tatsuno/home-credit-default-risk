import pandas as pd
import numpy as np
import random
from numpy.random import choice

from lightgbm import LGBMClassifier
import lightgbm as lgbm
from tqdm import trange

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
import gc

# hyper parameter
seed =1001

# domain parameter
DIR = '/Users/sh-tatsuno/.kaggle/competitions/home-credit-default-risk/'
index_cols = ['SK_ID_CURR','TEST','TARGET']


def DataMerger(datalist,key):
    for i in trange(len(datalist)):
        if i==0:
            merged = datalist[i]
        else:
            merged = merged.merge(datalist[i], on=key,how='left')
    return merged

def convert_cat(data):
    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]

    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])

def FoldSubmit(all_data, index_cols, model = None, submit=None, return_clf = False, seed = 43):
    np.random.seed(seed)
    folds = KFold(n_splits=4, shuffle=True, random_state=seed)
    data = all_data[all_data.TEST==0]
    y = data.TARGET
    data = data.drop(index_cols,axis=1)
    test = all_data[all_data.TEST==1].drop(index_cols,axis=1)
    total_score=0
    clf_set=[]

    sub_preds = np.zeros(test.shape[0])

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
        trn_x, trn_y = data.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = data.iloc[val_idx], y.iloc[val_idx]

        if model == None:
            clf = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=30,
                colsample_bytree=.8,
                subsample=.9,
                max_depth= 7,
                reg_alpha=.1,
                reg_lambda=.1,
                min_split_gain=.01,
                min_child_weight=2,
                random_state=seed,
                silent=True,
                verbose=-1,
            )
        else:
            clf = model

        arrange_inds = np.r_[choice(np.where(val_y==0)[0], len(np.where(val_y==1)[0])), np.where(val_y==1)[0]]
        clf.fit(trn_x, trn_y,
                eval_set= [(trn_x, trn_y), (val_x.iloc[arrange_inds], val_y.iloc[arrange_inds])],
                eval_metric='auc', verbose=100, early_stopping_rounds=150
               )


        oof_preds = clf.predict_proba(val_x.iloc[arrange_inds], num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        auc_score = roc_auc_score(val_y.iloc[arrange_inds], oof_preds)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, auc_score))

        total_score += auc_score
        if return_clf: clf_set.append(clf)

    print('Total AUC : %.6f' % (total_score/folds.n_splits))

    if submit!=None:
        submit_data = pd.DataFrame(np.c_[all_data[all_data.TEST==1]['SK_ID_CURR'].values.astype('int32'),sub_preds],
             columns=['SK_ID_CURR','TARGET'])
        submit_data['SK_ID_CURR'] = submit_data['SK_ID_CURR'].astype('int')
        submit_data['TARGET'] = submit_data['TARGET'].astype('float32')
        submit_data.to_csv('csv/'+submit, index=None, float_format='%.8f')

    if return_clf: return clf_set

    return

def null_feat(df, selected=None):
    if selected==None:
        cols = df.columns
    else:
        cols = selected

    for col in cols:
        if (np.sum(df[col] != df[col]) > 0):
            df[col+'_isnull'] = 0
            df[col+'_isnull'] = df[col+'_isnull'].where(df[col] == df[col], 1)

    return

def null_rate(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_application_train_data

def eda_rate_bar(df, index_col):
    tp = type(df[index_col].iloc[0])
    df[df.TEST==0][index_col] = df[df.TEST==0][index_col].astype(tp)
    df[df.TEST==1][index_col]  = df[df.TEST==1][index_col].astype(tp)
    fig = plt.figure(figsize=(15,7))
    fig.add_subplot(131)
    df[df.TEST==0][index_col].value_counts().sort_index().plot.bar()
    plt.title(index_col+': train')
    fig.add_subplot(132)
    df[df.TEST==1][index_col].value_counts().sort_index().plot.bar()
    plt.title(index_col+': test')
    fig.add_subplot(133)
    (df[(df.TEST==0)&(df.TARGET==1)][index_col].value_counts() / df[df.TEST==0][index_col].value_counts()).sort_index().plot.bar()
    plt.title(index_col+': target rate')
    plt.show()

def feature_importances(cols, clf):
    return pd.DataFrame([cols,clf.feature_importances_.tolist()],
        columns=['feature','score']).sort_values('score', ascending=False)

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
