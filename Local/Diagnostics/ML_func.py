# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:54:45 2022

@author: Yi Zhu
@e-mail: Yi.Zhu@inrs.ca
"""

import os
import numpy as np
import random
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, PredefinedSplit
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score
import torch
from torch.utils.data import ConcatDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
from Utils import util
import covid_dataset
from tqdm import tqdm

"""
Functions used to train and evaluate models.
"""
#%% sklearn0based ones
def train_model(X_train,y_train,
                clf_kwargs,
                random_state=26):
    """
    Given training data and labels, conduct model (shallow ones) training 
    with N-fold CV (or predefined split).
    """
    model_choice = ['svm','pca-svm']
    assert clf_kwargs['model'] in model_choice, " available models: svm/pca-svm "
    
    if clf_kwargs['model'] == 'svm':
        pipe = Pipeline([("scale", StandardScaler()), \
                        ("svm", SVC(kernel = 'linear',probability=True))
                        ])

    elif clf_kwargs['model'] == 'pca-svm':
        pipe = Pipeline([("scale", StandardScaler()), \
                        ("pca", PCA(n_components=clf_kwargs['n_components'])), \
                        ("svm", SVC(kernel = 'linear',probability=True))
                        ])

    param_grid = dict(
                     svm__C = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7]
                     )
    
    if clf_kwargs['pds'] == 'no':
        """ N-fold stratefied cross-validation """
        cv = RepeatedStratifiedKFold(n_splits=clf_kwargs["n_splits"], 
                                     n_repeats=clf_kwargs["n_repeats"], 
                                     random_state=random_state)
    
    elif clf_kwargs['pds'] == 'yes':
        cv = PredefinedSplit(clf_kwargs['split_index'])
    
    # start training
    print('Training model...')
    clf = GridSearchCV(pipe, param_grid=param_grid,cv=cv,scoring='roc_auc')
    clf.fit(X_train,y_train)
    print('Training finished. Best parameters are shown as follows:')
    print (clf.best_params_)

    return clf


def model_predict(x_test,y_test,pt_clf,raw=False):
    
    """
    Make predictions using a pretrained classifier and obtain evaluation metrics.
    """
    preds = pt_clf.predict(x_test)
    probs = pt_clf.predict_proba(x_test)[:, 1]
    UAR = recall_score(y_test, preds, average='macro')
    ROC = roc_auc_score(y_test, probs)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, probs, pos_label=1)
    report = {'UAR':UAR, 'ROC':ROC, 'TPR':tpr, 'TNR':fpr}

    if raw:
        return probs,preds
    
    return report


# %% Pytorch-based ones
def train_loop(model, train_dataloader, device, optimizer, criterion):
    
    output = {'total_score': 0.,
              'total_loss': 0.}

    running_loss = 0 
    model.to(device)
    model.train()
    preds_all = []
    labs_all = []
    itr = 0
    for _, data in enumerate(train_dataloader):
        
        itr += 1
        inputs,labs=data['data'].to(device),data['label'].to(device)
        # BP
        optimizer.zero_grad()
        ot=model(inputs)
        loss=criterion(ot,labs)
        loss.backward()
        # AUC-ROC score
        preds = torch.sigmoid(ot.detach())
        # preds = ot.detach()
        preds = preds.cpu().numpy()
        labs = labs.cpu().numpy()
        preds_all += list(preds)
        labs_all += list(labs)
        optimizer.step()
        
        # Calculate loss
        running_loss += loss.item()*inputs.size(0)
        
    train_loss = running_loss/len(train_dataloader)
    train_score = roc_auc_score(labs_all,preds_all)
    output['total_score'] = train_score
    output['total_loss'] = train_loss
    print('Training Loss: %.3f | AUC-ROC score: %.3f'%(train_loss,train_score)) 
    
    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, criterion):

    output = {'total_score': 0.,
              'total_loss': 0.}

    running_loss=0
    model.to(device)
    model.eval()
    preds_all = []
    labs_all = []
    itr = 0
    with torch.no_grad():
        for _,data in enumerate(valid_dataloader):
            
            itr += 1
            
            inputs,labs=data['data'].to(device),data['label'].to(device)
            ot=model(inputs)
            loss=criterion(ot,labs)
            # AUC-ROC score
            preds = torch.sigmoid(ot.detach())
            # preds = ot.detach()
            preds = preds.cpu().numpy()
            labs = labs.cpu().numpy()
            preds_all += list(preds)
            labs_all += list(labs)
            # Calculate loss
            running_loss += loss.item()*inputs.size(0)
  
    val_loss = running_loss/len(valid_dataloader)
    val_score = roc_auc_score(labs_all,preds_all)
    output['total_score'] = val_score
    output['total_loss'] = val_loss
    
    print('Validation Loss: %.3f | AUC-ROC score: %.3f'%(val_loss,val_score)) 

    return output['total_score'], output['total_loss']


def set_seed(seed,device):

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_main(model,\
               device,
               optimizer,
               criterion,
               dataset,
               feat:str,
               batch_sizes:list,
               save_model_to:str,
               metadata_path:str,
               mode:str='held-out',
               filters=None):
    
    # sanity check
    assert mode in ['held-out','joint'], "Unknown training mode"
    assert dataset in ['CSS','DiCOVA2','Cambridge'], "Unknown dataset name"

    # prepare dataloaders
    train_set = covid_dataset._covid_dataset(dataset=dataset,split='train',feat=feat,metadata_path=metadata_path)
    test_set = covid_dataset._covid_dataset(dataset=dataset,split='test',feat=feat,metadata_path=metadata_path,filters=filters)
    if dataset == 'CSS' or dataset == "Cambridge":
        valid_set = covid_dataset._covid_dataset(dataset=dataset,split='valid',feat=feat,metadata_path=metadata_path)

    if mode == 'joint' and (dataset == 'CSS' or dataset == "Cambridge"):
        train_set = ConcatDataset([train_set, valid_set])
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_sizes[0], shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_sizes[0], shuffle=False)
    if dataset == 'CSS' or dataset == "Cambridge":
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_sizes[0], shuffle=True)

    # training strategies
    valid_score_best = 0
    patience = 5
    num_epochs = 40
    score = {'train':[],'valid':[],'test':[]}

    if mode == 'held-out':
        print('Use held-out validation set')
        for e in range(num_epochs):
            train_score,_ = train_loop(model, train_loader, device, optimizer, criterion)
            valid_score,_ = valid_loop(model, valid_loader, device, criterion)
            test_score = test_loop(model, test_loader, device, raw=False)

            # TODO: remove test scores
            print('epoch {}: train score={:.3f} valid score={:.3f} test score={:.3f}'\
                .format(e,train_score,valid_score,test_score))
            
            score['train'].append(train_score)
            score['valid'].append(valid_score)
            score['test'].append(test_score)
            
            if e >25:
                if valid_score > valid_score_best:
                    print('Best score: {}. Saving model...'.format(valid_score))
                    torch.save(model.state_dict(), os.path.join(save_model_to,'clf.pt'))
                    valid_score_best = valid_score
                else:
                    patience -= 1
                    print('Score did not improve! {} <= {}. Patience left: {}'.format(valid_score,
                                                                                    valid_score_best,
                                                                                    patience))
                if patience == 0:
                    print('patience reduced to 0. Training Finished.')
                    break

    elif mode == 'joint':
        print('Joining training and validation set')
        _std = []
        for e in range(num_epochs):
            train_score, _ = train_loop(model, train_loader, device, optimizer, criterion)
            test_score = test_loop(model, test_loader, device, raw=False)

            print('epoch {}: train score={:.3f} test score={:.3f}'.format(e,train_score,test_score))
            
            score['train'].append(train_score)
            score['test'].append(test_score)
            if e>=3:
                _std.append(np.std(score['train'][-3:]))
            # if train_score > .90:
            #     print('Training score reaches the threshold. Training stops.')
            #     print('Best score: {}. Saving model...'.format(test_score))
            #     torch.save(model.state_dict(), save_model_to)
            #     break
    
    return score['test'], _std


def test_loop(model, test_dataloader, device, raw=False):
    # predict
    y_pred = []
    y_prob = []
    lab_list = []
    model.eval()
    with torch.no_grad():
        for _,data in enumerate(test_dataloader):
            inputs,labs=data['data'].to(device),data['label'].to(device)
            ot = model(inputs)
            probs = torch.sigmoid(ot.detach())
            probs = probs.cpu().numpy()
            # convert probabilities to binary predictions
            preds = np.full_like(probs, 1)
            preds[probs<0.5] = 0
            y_prob += list(probs)
            y_pred += list(preds)
            lab_list += list(labs.cpu().numpy())
    
    y_prob = np.array(y_prob).squeeze()
    y_pred = np.array(y_pred).squeeze()
    lab_list = np.array(lab_list)
    test_score = roc_auc_score(lab_list, y_prob)

    # print('\ntest score -> ' + str(test_score*100) + '%')
    if raw:
        return y_prob, y_pred

    return test_score


# %% Universal functions used to evaluate all types of models
def mean_confidence_interval(data):
    """
    Calculate mean and confidence intervals.
    """
    m = np.mean(data)
    std = np.std(data)
    h = 2*std
    return m, m-h, m+h


def eva_model(pt_clf,framework,x_test=None,y_test=None,test_dataloader=None,device=None,num_bs=1000,save_path=None,print_path=None,notes=None):
    
    """
    Evaluate the model performance using N times Bootstrap on test data.
    """
    result = {'UAR':[], 'ROC':[],
        'UAR_AVE':None, 'ROC_AVE':None,
        'UAR_TRUE':None, 'ROC_TRUE':None}

    # select framework and get predictions
    if framework == 'sklearn':
        result_true = model_predict(x_test,y_test,pt_clf,raw=True)
    elif framework == 'pytorch':
        result_true = test_loop(pt_clf,test_dataloader,device,raw=True)
    
    # get true UAR and AUC-ROC scores
    result['UAR_TRUE'] = roc_auc_score(y_test,result_true[0])
    result['ROC_TRUE'] = recall_score(y_test,result_true[1],average='macro')

    # get CIs on UAR and AUC-ROC
    for bs in tqdm(range(num_bs)):

        num_sample = len(y_test)
        idx = list(range(num_sample))
        bs_idx = random.choices(idx,k=num_sample)
        bs_probs = result_true[0][bs_idx]
        bs_preds = result_true[1][bs_idx]
        result['UAR'].append(roc_auc_score(y_test,bs_preds))
        result['ROC'].append(recall_score(y_test,bs_probs,average='macro'))
        # result['TPR'].append(result_bs['TPR'])
        # result['TNR'].append(result_bs['TNR'])
    
    result['UAR_AVE'] = mean_confidence_interval(result['UAR'])
    result['ROC_AVE'] = mean_confidence_interval(result['ROC'])
    # result['TPR_AVE'] = mean_confidence_interval(result['TPR'])
    # result['TNR_AVE'] = mean_confidence_interval(result['TNR'])

    if print_path == None:
        print_path = os.path.join(os.path.split(save_path)[0],'summary.txt')

    print('-----')
    util.print_to_file(print_path,'-----')
    if notes is not None:
        util.print_to_file(print_path,notes)
    print('True AUC-ROC: %.3f'%(result['ROC_TRUE']))
    util.print_to_file(print_path,'True AUC-ROC: %.3f'%(result['ROC_TRUE']))
    print('True UAR: %.3f'%(result['UAR_TRUE']))
    util.print_to_file(print_path,'True UAR: %.3f'%(result['UAR_TRUE']))
    print('Average AUC-ROC: %.3f'%(result['ROC_AVE'][0]))
    util.print_to_file(print_path,'Average AUC-ROC: %.3f'%(result['ROC_AVE'][0]))
    print('CI on AUC-ROC: %.3f-%.3f'%(result['ROC_AVE'][1],result['ROC_AVE'][2]))
    util.print_to_file(print_path,'CI on AUC-ROC: %.3f-%.3f'%(result['ROC_AVE'][1],result['ROC_AVE'][2]))
    print('Average UAR: %.3f'%(result['UAR_AVE'][0]))
    util.print_to_file(print_path,'Average UAR: %.3f'%(result['UAR_AVE'][0]))
    print('CI on UAR: %.3f-%.3f'%(result['UAR_AVE'][1],result['UAR_AVE'][2]))
    util.print_to_file(print_path,'CI on UAR: %.3f-%.3f'%(result['UAR_AVE'][1],result['UAR_AVE'][2]))

    # save results
    if save_path is not None:
        util.save_as_pkl(save_path,result)
        print('Results are saved to '+save_path)

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


def minmax_scaler(data,axis:tuple):
    data_min = data.min(axis=axis, keepdims=True)
    data_max = data.max(axis=axis, keepdims=True)
    data = (data - data_min)/(data_max - data_min + 1e-15)
    
    return data


def standard_scaler(data,axis:tuple):
    data_mean = data.mean(axis=axis, keepdims=True)
    data_std = data.std(axis=axis, keepdims=True)
    data = (data - data_mean)/(data_std+1e-15)
    
    return data



def create_filter(hei=23,wid=8,mask_hei:tuple=None,mask_wid:tuple=None):
    
    f = np.zeros((hei,wid))
    
    h_low = mask_hei[0]
    h_upp = mask_hei[1]
    w_low = mask_wid[0]
    w_upp = mask_wid[1]

    f[h_low:h_upp,w_low:w_upp] = 1 # filtering region with 1
    
    return f


def filter_ndarray(x,hei=23,wid=8,mask_hei:tuple=None,mask_wid:tuple=None):
    
    assert (x.shape[-2]==hei) & (x.shape[-1]==wid), "mask size needs to be the same as input"
    f = create_filter(hei,wid,mask_hei,mask_wid)
    filtered_input = np.multiply(x,f)
    
    return filtered_input


def create_multi_filter(hei=23,wid=8,
                        filters:dict={}):
    
    num_filter = len(filters)
    f = np.zeros((hei,wid))
    
    for i in range(num_filter):
        filt = list(filters.values())[i]
        h_low = filt[0][0]
        h_upp = filt[0][1]
        w_low = filt[1][0]
        w_upp = filt[1][1]
        f[h_low:h_upp,w_low:w_upp] = 1 # filling region with 1
    
    return f
    

def multi_filter(x,hei=23,wid=8,
                 filters:dict={}):

    filtered_input = x.copy()
    f = create_multi_filter(hei=hei,wid=wid,filters=filters)
    filtered_input = np.multiply(x,f)
    
    return filtered_input