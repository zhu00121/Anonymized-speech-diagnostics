# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:54:45 2022

@author: Yi Zhu
@e-mail: Yi.Zhu@inrs.ca
"""

import numpy as np
import random
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, PredefinedSplit
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
from Utils import util

"""
Functions used to train and evaluate models.
"""

def train_model(X_train,y_train,
                model:str,
                clf_kwargs,
                random_state=26):
    
    """
    Given training data and labels, conduct model (shallow ones) training 
    with N-fold CV (or predefined split).
    """
    model_choice = ['svm','rf','dt']
    assert model in model_choice, " available models: svm/rf/dt "
    
    if model == 'svm':
        m = SVC(kernel = 'linear',probability=True)
        params = {'C':[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7]}
    
    elif model == 'rf':
        m = RandomForestClassifier()
        params = {'n_estimators': [5,10,30,50,100,200],
                 'max_depth': [2, 5, 10, 20]}
    
    if clf_kwargs['pds'] == 'no':
        """ N-fold stratefied cross-validation """
        cv = RepeatedStratifiedKFold(n_splits=clf_kwargs["n_splits"], 
                                     n_repeats=clf_kwargs["n_repeats"], 
                                     random_state=random_state)
    
    elif clf_kwargs['pds'] == 'yes':
        cv = PredefinedSplit(clf_kwargs['split_index'])
    
    # start training
    print('Training models...')
    clf = GridSearchCV(m, param_grid=params,cv=cv,scoring='roc_auc')
    clf.fit(X_train,y_train)
    print('Training finished. Best parameters are shown as follows:')
    print (clf.best_params_)

    return clf


def mean_confidence_interval(data):
    """
    Calculate mean and confidence intervals.
    """
    m = np.mean(data)
    std = np.std(data)
    h = 2*std
    return m, m-h, m+h


def model_predict(x_test,y_test,pt_clf):
    
    """
    Make predictions using a pretrained classifier and obtain evaluation metrics.
    """
    preds = pt_clf.predict(x_test)
    probs = pt_clf.predict_proba(x_test)
    UAR = sklearn.metrics.recall_score(y_test, preds, average='macro')
    ROC = sklearn.metrics.roc_auc_score(y_test, probs)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, probs, pos_label=1)
    report = {'UAR':UAR, 'ROC':ROC, 'TPR':tpr, 'TNR':1-fpr}
    
    return report

def eva_model(x_test,y_test,pt_clf,num_bs=1000,save_path=None):
    
    """
    Evaluate the model performance using N times Bootstrap on test data.
    """
    result = {'UAR':[], 'ROC':[], 'TPR':[], 'TNR':[],\
        'UAR_AVE':0, 'ROC_AVE':0, 'TPR_AVE':0, 'TNR_AVE':0}
        
    for bs in range(num_bs):

        idx = list(range(x_test.shape[0]))
        bs_idx = random.choices(idx,k=x_test.shape[0])
        x_test_bs = x_test[bs_idx,:]
        x_test_bs = y_test[bs_idx]
        result_bs = model_predict(x_test,y_test,pt_clf)
        result['UAR'].append(result_bs['UAR'])
        result['ROC'].append(result_bs['ROC'])
        result['TPR'].append(result_bs['TPR'])
        result['TNR'].append(result_bs['TNR'])
    
    result['UAR_AVE'] = mean_confidence_interval(result['UAR'])
    result['ROC_AVE'] = mean_confidence_interval(result['ROC'])
    result['TPR_AVE'] = mean_confidence_interval(result['TPR'])
    result['TNR_AVE'] = mean_confidence_interval(result['TNR'])

    # save results
    if save_path is not None:
        util.save_as_pkl(result,save_path)
        print('Results are save to: '+save_path)

    return result


def get_confmat(label,preds):

    """
    Plot confusion matrix.
    """
    cf_matrix = confusion_matrix(label,preds)

    group_names = ['True neg','False pos','False neg','True pos']
    group_percentages1 = ["{0:.2%}".format(value) for value in
                          cf_matrix[0]/np.sum(cf_matrix[0])]
    group_percentages2 = ["{0:.2%}".format(value) for value in
                          cf_matrix[1]/np.sum(cf_matrix[1])]

    group_percentages = group_percentages1+group_percentages2
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2,v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    yticklabels = ['negative','positive']
    xticklabels = ['negative','positive']
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',yticklabels=yticklabels,
                xticklabels=xticklabels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    return cf_matrix



def draw_roc(num_rep:int,results_list:list):
    
    c_fill = ['rgba(41, 128, 185, 0.1)','rgba(255, 155, 69, 0.1)','rgba(60,182,90,0.1)']
    c_line = ['rgba(41, 128, 185, 0.5)','rgba(255, 155, 69, 0.5)','rgba(60,182,90,0.5)']
    c_line_main = ['rgba(41, 128, 185, 1.0)','rgba(255, 155, 69, 1.0)','rgba(60,182,90,1)']
    c_grid  = 'rgba(189, 195, 199, 0.5)'
    c_annot = 'rgba(149, 165, 166, 0.5)'
    c_highlight = 'rgba(192, 57, 43, 1.0)'
    system_name = ['MSF','LP','Two-stage']
    
    fig = sp.make_subplots(rows=1,cols=1, vertical_spacing=0.5)
    
    for idx,results in enumerate(results_list):
        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        
        for i in range(num_rep):
            fpr           = results['fpr_list'][i]
            tpr           = results['tpr_list'][i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
            
        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std = 2*np.std(interp_tprs, axis=0)
        tpr_upper = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower = tpr_mean-tpr_std
        auc = np.mean(results['auc_list'])
        
        trace_1 = go.Scatter(x = fpr_mean,
                             y = tpr_upper,
                             line = dict(color=c_line[idx], width=1),
                             hoverinfo = "skip",
                             showlegend = False,
                             name = 'upper')
        
        trace_2 = go.Scatter(x = fpr_mean,
                             y = tpr_lower,
                             fill = 'tonexty',
                             fillcolor  = c_fill[idx],
                             line = dict(color=c_line[idx], width=1),
                             hoverinfo  = "skip",
                             showlegend = False,
                             name = 'lower')
        
        trace_3 = go.Scatter(x = fpr_mean,
                             y = tpr_mean,
                             line = dict(color=c_line_main[idx], width=3),
                             hoverinfo  = "skip",
                             showlegend = True,
                             name = f'%s: {auc:.3f}'%(system_name[idx]))
        
        fig.add_trace(trace_1)
        fig.add_trace(trace_2)
        fig.add_trace(trace_3)
        
        # fig = go.Figure([
        #     go.Scatter(
        #         x          = fpr_mean,
        #         y          = tpr_upper,
        #         line       = dict(color=c_line, width=1),
        #         hoverinfo  = "skip",
        #         showlegend = False,
        #         name       = 'upper'),
        #     go.Scatter(
        #         x          = fpr_mean,
        #         y          = tpr_lower,
        #         fill       = 'tonexty',
        #         fillcolor  = c_fill,
        #         line       = dict(color=c_line, width=1),
        #         hoverinfo  = "skip",
        #         showlegend = False,
        #         name       = 'lower'),
        #     go.Scatter(
        #         x          = fpr_mean,
        #         y          = tpr_mean,
        #         line       = dict(color=c_line_main, width=2),
        #         hoverinfo  = "skip",
        #         showlegend = True,
        #         name       = f'AUC: {auc:.3f}')
        # ])
        
    fig.add_shape(
        type ='line', 
        line =dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        template    = 'plotly_white', 
        title_x     = 0.5,
        xaxis_title = "False positive rate",
        yaxis_title = "True positive rate",
        width       = 800,
        height      = 800,
        legend      = dict(yanchor="bottom", 
                           xanchor="right", 
                           x=0.95,
                           y=0.01,
        ),
        font=dict(family="Arial",
                  size=20),
    )
    fig.update_yaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        scaleanchor = "x", 
        scaleratio  = 1,
        linecolor   = 'black')
    fig.update_xaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        constrain   = 'domain',
        linecolor   = 'black')
    
    
    fig.show()
    pio.write_image(fig,
                    r'YOUR_OWN_PATH',
                    format = 'jpeg',
                    width=800, 
                    height=800,
                    scale=6)
    
    return 0
