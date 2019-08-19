### Contains All Data Gathering 
### Feature Creation Functions 
### Plotting Functions
### SFI/MDI & Purged K Fold & Other Modeling functions Implementation Based 
### Advances in Financial Machine Learning - Marcos Lopez De Prado (2018)

import numpy as np
import matplotlib.pyplot as plt
from tiingo import TiingoClient
from pandas.io.json import json_normalize
from datetime import datetime, timedelta
import dateutil.parser
import time
import pandas as pd
import seaborn as sns
config = {}
config['session'] = True
config['api_key'] = '355d47f0f5a210d8c28e3ee8ee5a020a6b0ad539'
client = TiingoClient(config)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
pd.options.mode.chained_assignment = None

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as mpl

from itertools import combinations 
from copy import deepcopy
import itertools

import statsmodels.api as sm

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import itertools
import os
from ta.volatility import bollinger_hband_indicator, bollinger_lband_indicator
from ta.trend import aroon_down, aroon_up, macd, macd_diff, macd_signal

sns.set_style("darkgrid")
sns.set(rc={'figure.figsize':(20,15)})
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

## Data / Data Aggregation
def GetMkt(tickers, start_date, end_date, freq = 'daily', bmark_tick = 'SPY'): # Outputs Stacked Price Data & Returns
    if np.any(tickers == bmark_tick) == False:
        print("Benchmark Ticker is not specified")
        return
    Stock_data = pd.DataFrame([])
    for i in tickers:
        histData = client.get_ticker_price(i,
		                                    fmt='json',
		                                    startDate=start_date,
		                                    endDate=end_date,
		                                    frequency=freq)
        histData = json_normalize(histData)
        histData['clRtn'] = histData['adjClose'].pct_change() + 1
        histData['clRtn'] = np.log(histData['clRtn'].fillna(1))
        histData['Pxln'] = histData['clRtn'].cumsum().values


        histData.reset_index(inplace=True)
        histData['date'] = histData.date.apply(lambda x: x[:10])
        histData['date'] = histData.date.map(lambda x: datetime.strptime(x,'%Y-' '%m-' '%d'))
        histData['ticker'] = i

        if i != bmark_tick:
            dkeys = Stock_data[Stock_data.ticker == bmark_tick]['date']
            histData = histData[histData['date'].isin(dkeys)] 
            histData['relRtn'] = (histData['clRtn'].values - Stock_data[Stock_data['ticker'] == bmark_tick]['clRtn'].values) 
            histData['relPxln'] = histData['relRtn'].cumsum().values
        else:
            histData['relRtn'] = 0.0
            histData['relPxln'] = 0.0

        # Historical Return Series Time Frames
        histSer = [10, 21, 63, 126, 252]
        for s in histSer:
        	if len(str(s)) == 2:
        		_Ser = '0' + str(s)
        	else:
        		_Ser = str(s)
        	histData['clRtn'+'H'+_Ser] = histData['clRtn'].rolling(window=s, min_periods=s).sum()/s
        	histData['relRtn'+'H'+_Ser] = histData['relRtn'].rolling(window=s, min_periods=s).sum()/s
        	histData['clHv'+_Ser] = histData['clRtn'].rolling(window=s, min_periods=s).std()*np.sqrt(252)
        	histData['relHv'+_Ser] = histData['relRtn'].rolling(window=s, min_periods=s).std()*np.sqrt(252)


        print(i, histData.shape, histData.date.min())
        Stock_data = Stock_data.append(histData, ignore_index=True, sort=True)

    Stock_data['date'] = pd.to_datetime(Stock_data['date'])
    Stock_data['period'] = Stock_data['date']
    Stock_data.index = Stock_data.date
    
    print(Stock_data.shape, Stock_data.date.max(), Stock_data.date.min())
    
    return Stock_data

def ForwardTarget(df, tickers, rfld, f=21):
    sides = pd.DataFrame([])
    for t in tickers:
        side = []
        forwards = []
        df_t = df[df['ticker'] == t][['date', 'ticker']]
        for i in np.arange(0,df_t.shape[0]):
            f_ret = df[df['ticker'] == t][rfld][i+1:(i+f+1)].sum()
            forwards.append(f_ret)
            if f_ret > 0.0:
                side.append(1)
            else: 
                side.append(-1)     
        df_t['side'] = side
        df_t['forward'] = forwards
        scaler = MinMaxScaler()
        df_t['forwardmmS'] = scaler.fit_transform(df_t['forward'].values.reshape(-1,1))
        scaler = StandardScaler()
        df_t['forwardstdS'] = scaler.fit_transform(df_t['forward'].values.reshape(-1,1))
        sides = sides.append(df_t, ignore_index=True)
        print(t)

    sides = sides.set_index('date')
    sides['date'] = sides.index
    

    return sides

def getTimeDecay(tW,clfLastW=1.0):
    # apply piecewise-linear decay to observed uniqueness
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:
        slope=(1.0-clfLastW)/clfW.iloc[-1]
    else:
        slope=1.0/((clfLastW+1)*clfW.iloc[-1])
    const=1.0-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    return clfW

#def AggregateFeatures(px, tickers, rfld, dict_args):

# Model Evaluation Data Functions

def featImpMDI(fit, featNames):
    #feat importance based on IS mean impurity reduction
    df0={i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan)
    imp = pd.concat({'mean':df0.mean(), 'std':df0.std()*df0.shape[0]**-.5}, axis=1)
    imp/=imp['mean'].sum()
    return imp

def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy', 'f1']:
        raise Exception('wrong scoring method '+scoring)
    from sklearn.metrics import log_loss,accuracy_score, f1_score
    cvGen=PurgedKFold(n_splits=cv, 
                      t1=t1, 
                      pctEmbargo=pctEmbargo)
    scr0,scr,scr1=pd.Series(), pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0,y0,w0=X.iloc[train,:], y.iloc[train], sample_weight.iloc[train]
        X1,y1,w1=X.iloc[test,:], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0,
                      y=y0,
                      sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1)
            scr.loc[i] = -log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
            prob=fit.predict_proba(X0)
            scr0.loc[i] = -log_loss(y0,prob,sample_weight=w0.values,labels=clf.classes_)
        if scoring=='accuracy':
            pred=fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1,pred,sample_weight=w1.values)
            pred=fit.predict(X0)
            scr.loc[i] = accuracy_score(y0,pred,sample_weight=w0.values)
        if scoring=='f1':
            pred=fit.predict(X1)
            scr0.loc[i] = f1_score(y1,pred,sample_weight=w1.values)
            pred=fit.predict(X0)
            scr.loc[i] = f1_score(y0,pred,sample_weight=w0.values)
            
        for j in X.columns:
            X1_=X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # permutation of a single column
            if scoring=='neg_log_loss':
                prob=fit.predict_proba(X1_)
                scr1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
            if scoring=='accuracy':
                pred=fit.predict(X1_)
                scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
            if scoring=='f1':
                pred=fit.predict(X1_)
                scr1.loc[i,j]=f1_score(y1,pred,sample_weight=w1.values)
    imp=(-scr1).add(scr0, axis=0)
    if scoring=='neg_log_loss':
        imp=imp/-scr1
    else:
        imp=imp/(1.0-scr1)
    imp=pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-.5}, axis=1)
    return imp, scr0.mean(), scr.mean()

def ImpSFI(featNames, clf, trnsX, Y, wts, scoring, t1=None, cv_splits = 4):
    imp=pd.DataFrame(columns=['mean', 'std'])
    cvGen=PurgedKFold(n_splits=cv_splits, 
                      t1=t1, 
                      pctEmbargo=.01)
    for featName in featNames:
        print(featName)
        df0=cvScore(clf,
                    X=trnsX[[featName]], 
                    y=Y, 
                    sample_weight=wts,
                    scoring=scoring,
                    cvGen=cvGen)
        imp.loc[featName, 'mean']=df0.mean()
        imp.loc[featName, 'std']=df0.std()*df0.shape[0]**-.5
    return imp

from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    '''
    Extend Kfold to work with labels that span intervals
    the train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle = False). 
    w/o training examples in between
    '''
    def __init__(self, n_splits = 3, t1=None, pctEmbargo=0.0):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pandas series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo
    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[(i[0],i[-1] +1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices = indices[i:j]
            train_indices = np.concatenate((indices[(indices > j+mbrg)],
                                            indices[(indices < i-mbrg)]))
            yield train_indices, test_indices

def cvScore(clf,X,y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=None):
    if scoring not in ['neg_log_loss', 'accuracy', 'f1']:
        raise Exception('wrong scoring method')
    from sklearn.metrics import log_loss,accuracy_score, f1_score
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv, 
                          t1=t1, 
                          pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[train,:],
                    y=y.iloc[train], 
                    sample_weight=sample_weight.iloc[train].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test],
                             prob,
                             sample_weight=sample_weight.iloc[test].values,
                             labels=clf.classes_)
        if scoring=='accuracy':
            pred=fit.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test], 
                                  pred, 
                                  sample_weight=sample_weight.iloc[test].values)
        if scoring=='f1':
            pred=fit.predict(X.iloc[test,:])
            score_=f1_score(y.iloc[test], 
                                  pred, 
                                  sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)


def cvForecast(clf,X,y, sample_weight, t1=None, cv=None, cvGen=None, pctEmbargo=None):
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv, 
                          t1=t1, 
                          pctEmbargo=pctEmbargo) # purged
    pred=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[train,:],
                    y=y.iloc[train], 
                    sample_weight=sample_weight.iloc[train].values)
        pred_ = fit.predict(X.iloc[test,:])
        pred += list(pred_)
    return pd.Series(pred, index=t1.index)

# Simple Long/Short Model Backtest

def mCombPurgedBacktest(X, Y, Y_r, wts, bTestRetWindow, 
    clf,  clfs=None, X_s = None, 
    k_n = [2, 35], purg_i = .02 , t_cost = 5, test_sims = 0, leverage = 1.0, 
    tcostScore ='LS'):
    if tcostScore not in ['LS', 'I']:
        raise Exception('wrong transaction cost scoring method')
    # Simple Backtest for +1/-1 Side only Model
    start = time.time()
    trs = pd.DataFrame([], columns= ['Ann-Return-Net-OOS', 
                                             'Ann-Vol-OOS', 
                                             'Sharpe-OOS',
                                             'Ann-Return-Net-IS', 
                                             'Ann-Vol-IS', 
                                             'Sharpe-IS',
                                             'TCost',
                                             'TC-Score', 
                                             'Trialn', 
                                             'Sim_n',
                                             'TestYears',
                                             'bTestRetWindow'])
    K = k_n[0]
    N = k_n[1]
    K_comb = np.array(list(combinations(np.arange(0,N), K)))
    N_indx = np.array_split(Y.index, N)
    P_indx_ = []

    if type(purg_i) == float:
        purg_i = int(Y.shape[0]*.02)

    # Goes through the combinations & purges where test / train sets line up
    for i in np.arange(0,len(K_comb)):
        N_indx_ = deepcopy(N_indx)
        for i_ in np.arange(0,N):
            if ((K_comb[i] != i_).all() & (i_ != N) & (K_comb[i] == i_+1).any()):
                N_indx_[i_] = N_indx_[i_][:-purg_i]
            if ((K_comb[i] != i_).all() & (i_ != 0) & (K_comb[i] == i_-1).any()):
                N_indx_[i_] = N_indx_[i_][purg_i:]        
        P_indx_.append(N_indx_)
        
    print(np.array(P_indx_).shape[0], 'Paths')
    TestYears = np.round(len(Y.index)/N*K/252/Y_r.ticker.unique().shape[0],2)
    print('Test Set Obs in Years', TestYears)
    print('TOTAL DATA in Years', np.round(len(Y.index)/252/Y_r.ticker.unique().shape[0],2))
    for i in np.arange(0,len(K_comb)): 
        l1 = [P_indx_[i][i_] for i_ in K_comb[i]]
        l1 = [y for x in l1 for y in x]

        l2 = [P_indx_[i][i_] for i_ in np.setdiff1d(np.arange(0,N), K_comb[i])]
        l2 = [y for x in l2 for y in x]

        X_test = X[X.index.isin(l1)]
        Y_test = Y[Y.index.isin(l1)]

        X_train = X[X.index.isin(l2)]
        Y_train = Y[Y.index.isin(l2)]

        clf.fit(X_train, Y_train, sample_weight=wts[wts.index.isin(l2)])
        pred = clf.predict(X_test)
        insample = clf.predict(X_train)

        #Train Size Model if Specifed
        if clfs != None:
                Ys_train = pd.Series(np.where(Y_train == clf.predict(X_train), 1, 0),
                                        index = Y_train.index)
                clfs.fit(X_s[X_s.index.isin(l2)],  Ys_train, sample_weight=wts[wts.index.isin(l2)])
                pred_s = clfs.predict(X_s[X_s.index.isin(l1)])
                insample_s = clfs.predict(X_s[X_s.index.isin(l2)])
        else:
                pred_s = np.full(Y_test.shape,1)
                insample_s = np.full(Y_train.shape,1)

        insample = pd.DataFrame(insample, index = Y_train.index, columns = {'side'})
        insample[bTestRetWindow] = list(Y_r[Y_r.period.isin(l2)][bTestRetWindow])
        insample['ticker'] = list(Y_r[Y_r.period.isin(l2)]['ticker'])
        insample['size'] = insample_s
        insample['IS-Perf'] = (insample[bTestRetWindow] * insample['side'] * insample['size']) + 1
        insample =insample.reset_index().sort_values(by=['ticker', 'period']).reset_index(drop=True)
        
        if tcostScore =='LS':
            _tcst = list()
            for tk in insample['ticker'].unique():
                tmp_ = insample[insample['ticker'] == tk]
                for ind in tmp_.index:
                    if tmp_.loc[(ind-1):(ind)].shape[0] < 2:
                        _tcst.append(float('NaN'))
                    else:
                        _tcst.append(tmp_.loc[(ind-1)]['side']*tmp_.loc[(ind-1)]['size'] - tmp_.loc[(ind)]['side']*tmp_.loc[(ind)]['size'])
            insample['T_cost'] = np.abs(_tcst)*t_cost

        if tcostScore == 'I':
            insample['T_cost'] = t_cost*2*insample['size']

        insample['IS-NetPerf'] = insample['IS-Perf']-insample['T_cost']/10000

        pred = pd.DataFrame(pred, index = Y_test.index, columns = {'side'})
        pred[bTestRetWindow] = list(Y_r[Y_r.period.isin(l1)][bTestRetWindow])
        pred['ticker'] = list(Y_r[Y_r.period.isin(l1)]['ticker'])
        pred['size'] = pred_s
        pred['OOS-Perf'] = (pred[bTestRetWindow] * pred['side'] * pred['size']) + 1
        pred = pred.reset_index().sort_values(by=['ticker', 'period']).reset_index(drop=True)
        
        if tcostScore =='LS':
            _tcst = list()
            for tk in pred['ticker'].unique():
                tmp_ = pred[pred['ticker'] == tk]
                for ind in tmp_.index:
                    if tmp_.loc[(ind-1):(ind)].shape[0] < 2:
                        _tcst.append(float('NaN'))
                    else:
                        _tcst.append(tmp_.loc[(ind-1)]['side']*tmp_.loc[(ind-1)]['size'] - tmp_.loc[(ind)]['side']*tmp_.loc[(ind)]['size'])
            pred['T_cost'] = np.abs(_tcst)*t_cost

        if tcostScore == 'I':
            pred['T_cost'] = t_cost*2*pred['size']

        pred['OOS-NetPerf'] = pred['OOS-Perf']-pred['T_cost']/10000

        tick_wt = leverage/Y_r.ticker.unique().shape[0]
        piv_p = ((pred.pivot(index='period', columns='ticker')['OOS-NetPerf']-1)*(tick_wt))
        piv_i = ((insample.pivot(index='period', columns='ticker')['IS-NetPerf']-1)*(tick_wt))
        
        piv_p_annret = ((piv_p.sum(axis=1).mean()+1)**(252/int(bTestRetWindow[-3:])))-1
        piv_p_vol = (piv_p.sum(axis=1)+1).std()*np.sqrt((252/int(bTestRetWindow[-3:])))
        piv_i_annret = ((piv_i.sum(axis=1).mean()+1)**(252/int(bTestRetWindow[-3:])))-1
        piv_i_vol = (piv_i.sum(axis=1)+1).std()*np.sqrt((252/int(bTestRetWindow[-3:])))

        trs.loc[len(trs)+1] = [piv_p_annret, 
                                 piv_p_vol, 
                                 piv_p_annret/piv_p_vol,
                                 piv_i_annret,
                                 piv_i_vol,
                                 piv_i_annret/piv_i_vol,
                                 t_cost, 
                                 tcostScore,
                                 i, 
                                 test_sims,
                                 TestYears,
                                 bTestRetWindow]
        
        if i % 100 == 0:
                elapsed = time.time() - start
                print(i," Time elapsed", int(elapsed/60))
           
    trs = trs.reset_index()
    trs = trs.drop(['index'], axis=1)


    return trs
   
# Regime Splitting MDA/MDI evaluation

def RegimeEvaluation(mdata, cols, tickers, bc):
    MDI = pd.DataFrame([])
    MDA = pd.DataFrame([])
    for tick in tickers:
        regime = [tick+'forward', 'MeanRev-3Mrealized'+tick]
        df_ = mdata[regime]
        df_['year'] = mdata.index.map(lambda x: datetime.strftime(x,'%Y'))
        dfg = df_.groupby(['year']).agg({regime[0] : 'mean', regime[1] : 'mean'})

        hRy = np.array(dfg[dfg[regime[0]] >= dfg[regime[0]].quantile(.75)].index)
        lRy = np.array(dfg[dfg[regime[0]] <= dfg[regime[0]].quantile(.25)].index)
        mRy = np.array(dfg[(dfg[regime[0]] > dfg[regime[0]].quantile(.25)) & (dfg[regime[0]] < dfg[regime[0]].quantile(.75))].index)

        hRy1 = np.array(dfg[dfg[regime[1]] >= dfg[regime[1]].quantile(.75)].index)
        lRy1 = np.array(dfg[dfg[regime[1]] <= dfg[regime[1]].quantile(.25)].index)
        mRy1 = np.array(dfg[(dfg[regime[1]] > dfg[regime[1]].quantile(.25)) & (dfg[regime[1]] < dfg[regime[1]].quantile(.75))].index)

        cols =  ['MeanRev-3Mrealized',
                       'MeanRev-rescaled-',
                       'MeanRev-rescaled-EWMA2-',
                       'MeanRev-rs_ratio-',
                       'Trend-absBma-',
                       'Trend-hBma-',
                       'Trend-rs-rsm-',
                       'Trend-MA-Slope200-',
                       'Trend-MA-Slope100-',
                       'Trend-MA-Slope50-']
        cols = [s + tick for s in cols]
        X = mdata.loc[:, cols]
        Y = mdata[tick+'side']
        X['year'] = df_['year']

        X_h = X[X['year'].isin(hRy)].drop(['year'], axis = 1)
        X_l = X[X['year'].isin(lRy)].drop(['year'], axis = 1)
        X_m = X[X['year'].isin(mRy)].drop(['year'], axis = 1)

        # Forward Returns Regime
        # High
        
        Y_train = Y.loc[X_h.index]
        wa = (mdata.loc[X_h.index, tick+'forward'] - 1 ).abs()
        wts = wa*1/(wa.sum())
        wts = getTimeDecay(wts,clfLastW=0.0)
        bc.fit(X_h,
               Y_train, 
               sample_weight=wts.loc[X_h.index])

        MDI_ = featImpMDI(bc, X_h.columns)
        MDI_['Type'] = 'hForwardRet'
        MDI_['RType'] = 'returns'
        MDI_['Feature'] = MDI_.index
        MDI_['tick'] = tick
        MDI = MDI.append(MDI_, ignore_index=True, sort=False) 
        
        MDA_, OOS, OOB = featImpMDA(bc, 
                               X=X_h, 
                               y=Y_train, 
                               cv=4, 
                               sample_weight = wts.loc[X_h.index], 
                               t1 = pd.Series(Y_train.index, index=Y_train.index, name = 't1'), 
                               pctEmbargo = .01, 
                               scoring='accuracy')
        
        MDA_['OOS'] = OOS
        MDA_['OOB'] = OOB
        MDA_['Type'] = 'hForwardRet'
        MDA_['RType'] = 'returns'
        MDA_['Feature'] = MDA_.index
        MDA_['tick'] = tick
        MDA = MDA.append(MDA_, ignore_index=True, sort=False)     

        # Medium

        Y_train = Y.loc[X_m.index]
        wa = (mdata.loc[X_m.index, tick+'forward'] - 1 ).abs()
        wts = wa*1/(wa.sum())
        wts = getTimeDecay(wts,clfLastW=0.0)
        bc.fit(X_m,
               Y_train, 
               sample_weight=wts.loc[X_m.index])

        MDI_ = featImpMDI(bc, X_m.columns)
        MDI_['Type'] = 'mForwardRet'
        MDI_['RType'] = 'returns'
        MDI_['Feature'] = MDI_.index
        MDI_['tick'] = tick
        MDI = MDI.append(MDI_, ignore_index=True, sort=False) 
        
        
        MDA_, OOS, OOB = featImpMDA(bc, 
                                   X=X_m, 
                                   y=Y_train, 
                                   cv=4, 
                                   sample_weight = wts.loc[X_m.index], 
                                   t1 = pd.Series(Y_train.index, index=Y_train.index, name = 't1'), 
                                   pctEmbargo = .01, 
                                   scoring='accuracy')
        
        MDA_['OOS'] = OOS
        MDA_['OOB'] = OOB
        MDA_['Type'] = 'mForwardRet'
        MDA_['RType'] = 'returns'
        MDA_['Feature'] = MDA_.index
        MDA_['tick'] = tick
        MDA = MDA.append(MDA_, ignore_index=True, sort=False)

        # Low

        Y_train = Y.loc[X_l.index]
        wa = (mdata.loc[X_l.index, tick+'forward'] - 1 ).abs()
        wts = wa*1/(wa.sum())
        wts = getTimeDecay(wts,clfLastW=0.0)
        bc.fit(X_l,
               Y_train, 
               sample_weight=wts.loc[X_l.index])

        MDI_ = featImpMDI(bc, X_l.columns)
        MDI_['Type'] = 'lForwardRet'
        MDI_['RType'] = 'returns'
        MDI_['Feature'] = MDI_.index
        MDI_['tick'] = tick
        MDI = MDI.append(MDI_, ignore_index=True, sort=False)

        MDA_, OOS, OOB = featImpMDA(bc, 
                                   X=X_l, 
                                   y=Y_train, 
                                   cv=4, 
                                   sample_weight = wts.loc[X_l.index], 
                                   t1 = pd.Series(Y_train.index, index=Y_train.index, name = 't1'), 
                                   pctEmbargo = .01, 
                                   scoring='accuracy')
        
        MDA_['OOS'] = OOS
        MDA_['OOB'] = OOB
        MDA_['Type'] = 'lForwardRet'
        MDA_['RType'] = 'returns'
        MDA_['Feature'] = MDA_.index
        MDA_['tick'] = tick
        MDA = MDA.append(MDA_, ignore_index=True, sort=False)
        
        

        # Vol Regimes
        ##################
        ####################

        X_h = X[X['year'].isin(hRy1)].drop(['year'], axis = 1)
        X_l = X[X['year'].isin(lRy1)].drop(['year'], axis = 1)
        X_m = X[X['year'].isin(mRy1)].drop(['year'], axis = 1)

        # High
        
        Y_train = Y.loc[X_h.index]
        wa = (mdata.loc[X_h.index, tick+'forward'] - 1 ).abs()
        wts = wa*1/(wa.sum())
        wts = getTimeDecay(wts,clfLastW=0.0)
        bc.fit(X_h,
               Y_train, 
               sample_weight=wts.loc[X_h.index])

        MDI_ = featImpMDI(bc, X_h.columns)
        MDI_['Type'] = 'hrvol'
        MDI_['RType'] = 'rvol'
        MDI_['Feature'] = MDI_.index
        MDI_['tick'] = tick
        MDI = MDI.append(MDI_, ignore_index=True, sort=False)
        
        MDA_, OOS, OOB = featImpMDA(bc, 
                                   X=X_h, 
                                   y=Y_train, 
                                   cv=4, 
                                   sample_weight = wts.loc[X_h.index], 
                                   t1 = pd.Series(Y_train.index, index=Y_train.index, name = 't1'), 
                                   pctEmbargo = .01, 
                                   scoring='accuracy')
        
        MDA_['OOS'] = OOS
        MDA_['OOB'] = OOB
        MDA_['Type'] = 'hrvol'
        MDA_['RType'] = 'rvol'
        MDA_['Feature'] = MDA_.index
        MDA_['tick'] = tick
        MDA = MDA.append(MDA_, ignore_index=True, sort=False)

        # Medium
        
        Y_train = Y.loc[X_m.index]
        wa = (mdata.loc[X_m.index, tick+'forward'] - 1 ).abs()
        wts = wa*1/(wa.sum())
        wts = getTimeDecay(wts,clfLastW=0.0)
        bc.fit(X_m,
               Y_train, 
               sample_weight=wts.loc[X_m.index])

        MDI_ = featImpMDI(bc, X_m.columns)
        MDI_['Type'] = 'mrvol'
        MDI_['RType'] = 'rvol'
        MDI_['Feature'] = MDI_.index
        MDI_['tick'] = tick
        MDI = MDI.append(MDI_, ignore_index=True, sort=False) 
        
        MDA_, OOS, OOB = featImpMDA(bc, 
                                   X=X_m, 
                                   y=Y_train, 
                                   cv=4, 
                                   sample_weight = wts.loc[X_m.index], 
                                   t1 = pd.Series(Y_train.index, index=Y_train.index, name = 't1'), 
                                   pctEmbargo = .01, 
                                   scoring='accuracy')
        
        MDA_['OOS'] = OOS
        MDA_['OOB'] = OOB
        MDA_['Type'] = 'mrvol'
        MDA_['RType'] = 'rvol'
        MDA_['Feature'] = MDA_.index
        MDA_['tick'] = tick
        MDA = MDA.append(MDA_, ignore_index=True, sort=False)

        # Low
        
        Y_train = Y.loc[X_l.index]
        wa = (mdata.loc[X_l.index, tick+'forward'] - 1 ).abs()
        wts = wa*1/(wa.sum())
        wts = getTimeDecay(wts,clfLastW=0.0)
        bc.fit(X_l,
               Y_train, 
               sample_weight=wts.loc[X_l.index])
        MDI_ = featImpMDI(bc, X_l.columns)
        MDI_['Type'] = 'lrvol'
        MDI_['RType'] = 'rvol'
        MDI_['Feature'] = MDI_.index
        MDI_['tick'] = tick
        MDI = MDI.append(MDI_, ignore_index=True, sort=False)
        
        MDA_, OOS, OOB = featImpMDA(bc, 
                                   X=X_l, 
                                   y=Y_train, 
                                   cv=4, 
                                   sample_weight = wts.loc[X_l.index], 
                                   t1 = pd.Series(Y_train.index, index=Y_train.index, name = 't1'), 
                                   pctEmbargo = .01, 
                                   scoring='accuracy')
        
        MDA_['OOS'] = OOS
        MDA_['OOB'] = OOB
        MDA_['Type'] = 'lrvol'
        MDA_['RType'] = 'rvol'
        MDA_['Feature'] = MDA_.index
        MDA_['tick'] = tick
        MDA = MDA.append(MDA_, ignore_index=True, sort=False)
        
        MDI['Feature'] = MDI['Feature'].str.replace(tick, '', regex=False)
        MDA['Feature'] = MDI['Feature'].str.replace(tick, '', regex=False)
        
        X = X.drop(['year'], axis = 1)
        print(tick, ' Done')
        
    MDI['Feat-Type'] = MDI['Feature'].str.contains('Trend')
    MDI['Feat-Type'] = np.where(MDI['Feat-Type'] == True, 'Trend', 'MeanReversion')
    MDA['Feat-Type'] = MDA['Feature'].str.contains('Trend')
    MDA['Feat-Type'] = np.where(MDA['Feat-Type'] == True, 'Trend', 'MeanReversion')

    print(MDI.shape, MDA.shape) 
    MDI = MDI.rename(index=str, columns={'mean': "Mean Decrease in Impurity",
                                         'Type': "Forward Return & Realized Vol Regimes"})
    MDA = MDA.rename(index=str, columns={'mean': "Mean Decrease in Accuracy",
                                         'Type': "Forward Return & Realized Vol Regimes"})

    return MDI, MDA

## Features

# RRG functions
def rs_ratio(prices_df, benchmark, window=12):
    rs_df = pd.DataFrame([], index = prices_df.index.unique())
    for series in prices_df:
        rs = (prices_df[series].divide(benchmark)) * 100
        rs_window = rs.ewm(span=window).mean()
        rs_diff = rs_window.diff()/rs_window
        rs_ratio = 100 + ((rs_window - rs_window.mean()) / rs_window.std())
        rs_momentum = 100 + ((rs_diff - rs_diff.mean()) / rs_diff.std())
        rs_df[series[-3:]+'rs_ratio'] = rs_ratio
        rs_df[series[-3:]+'rs_momentum'] = rs_momentum
    rs_df.dropna(axis=0, how='any', inplace=True) 
    return rs_df

def rs_rank(returns_df, FastMa = 8, SlowMa = 30, span0 = 100):
    
    ### Method Two Attempts to Measure Normalized vs Benchmark RS & RS Momentum
    # Calculate both a 8 and 30 week SMA
    # Take MACD of above
    # Create RS Ratio using Normalized MACD Values, Use Z Score Normalization 
    # Calculate 16 week SMA of RS Ratio
    # Create RS Momentum of above, use 4 week SMA 
    # Plot RS Ratio vs. RS Momentum
    #.rolling(window=FastMa, min_periods=FastMa)
    
    
    df_fast = returns_df.rolling(window=FastMa, min_periods=FastMa).agg(lambda x : x.prod())
    df_slow = returns_df.rolling(window=SlowMa, min_periods=FastMa).agg(lambda x : x.prod())
    
    MACD = (df_slow - df_fast)
    
    norm = MACD.std() 
    
    RS_norm = MACD.divide(norm.values, axis=1) # Normalized to 1 Y Vol of MACD Rel Returns
    
    RS = 100 + RS_norm.ewm(span=FastMa).mean()
    
    RSM = 100 + (RS.diff() / RS.diff().std()).ewm(span=FastMa).mean() 

    RS.columns = [col.replace('rreturns','rs_ratio') for col in list(RS.columns)]

    RSM.columns = [col.replace('rreturns','rs_momentum') for col in list(RSM.columns)]
    
    rs_df = RSM.merge(RS,left_index=True, right_index=True)
    
    rs_df.dropna(axis=0, how='any', inplace=True)
    
    print(returns_df.shape, rs_df.shape)
    
    return rs_df

# Fractional Differentiation 

def getWeights_FFD(d,thres):
    w, k =[1.], 1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_) < thres: break
        w.append(w_) 
        k+=1
    return np.array(w[::-1]).reshape(-1,1)

def fracDiff_FFD(series, d, thres=1e-5):
    # Constant width window (new solution)
    w = getWeights_FFD(d,thres)
    width = len(w)-1 
    df = {}
    seriesF = series.fillna(method='ffill').dropna() 
    df_ = pd.Series()
    for iloc1 in range(width, seriesF.shape[0]):
        loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
        #print(iloc1, width, seriesF.shape[0])
        if not np.isfinite(series.loc[loc1]) : 
            continue 
        df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0]
    df=df_.copy(deep=True)
    df.name = series.name
    return df
    
def plotMinFFD(df1, thres = .01):
    out=pd.DataFrame(columns=['adfStat','pVal', 'lags', 'nObs', '95% conf', 'corr', 'Diff'])
    for d in np.linspace(0,1,11):
        df2 = fracDiff_FFD(df1,d,thres = thres)
        corr=np.corrcoef(df1.loc[df2.index], df2)[0,1]
        df2=adfuller(df2, maxlag=1, regression='c', autolag = None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr] + [d]
    return out

# Technicals
def RSI(close, window_length = 20, MA = 'S'):
    delta = close.diff()[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    if MA == 'E':
        roll_up = up.ewm(span=window_length).mean()
        roll_down = down.abs().ewm(span=window_length).mean()
        RS1 = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS1))
        
    if MA == 'S':
        roll_up = up.rolling(window=window_length, min_periods=1).mean()
        roll_down = down.abs().rolling(window=window_length, min_periods=1).mean()
        RS1 = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS1))
        
    RSI_S = RSI.diff()
    RSI_S = RSI_S.rolling(window=window_length, min_periods=5).mean()
        
    return RSI[window_length:], RSI_S[window_length:]

def CrossOver(MA1, MA2, wwindow = 5):
    ind = [1]
    window_ = 0
    for i in MA1.index[1:]:
        if MA1[i] > MA2[i]:
            if (ind[-1] != 1) & (window_ < wwindow):
                window_ += 1
                ind.append(ind[-1])
            else:
                window_ = 0
                ind.append(1)
        else:
            if (ind[-1] != -1) & (window_ < wwindow):
                window_ += 1
                ind.append(ind[-1])
            else:
                window_ = 0
                ind.append(-1)
                
    signal = pd.Series(ind, MA1.index)
    
    return signal

## Plotting

def PlotReturnLines(df, tickers):
    # Benchmark ticker must be ticker[0]
    # Plots Relative Returns & Cummulative Return Plots
    data = []
    for i in tickers[1:]:
        trace3 = go.Scatter(x = df.index,
                            y = df['rreturns-'+i].values,
                            mode = 'lines',
                            name = i,
                            marker=dict(colorscale='Viridis',
                                        size = 10,
                                        opacity = .9)) 
        data.append(trace3)

    layout = go.Layout(
        title = 'Relative to SPY Returns',
        titlefont=dict(family='Balto, sans-serif', size=30, color='black'),
        xaxis=dict(
            title='C Rel Returns',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')),
        yaxis=dict(
            title='Date',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')))
    fig = go.Figure(data=data, layout=layout)

    data = []
    for i in tickers:
        tick = i
        trace3 = go.Scatter(x = df.index,
                            y = df['creturns-'+i].cumprod().values,
                            mode = 'lines',
                            name = 'C Returns ' + tick) 

        data.append(trace3)

    layout = go.Layout(
        title = 'Cummulative Returns',
        titlefont=dict(family='Balto, sans-serif', size=30, color='black'),
        xaxis=dict(
            title='C Returns',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')),
        yaxis=dict(
            title='Date',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')))
    fig1 = go.Figure(data=data, layout=layout)

    return fig, fig1

def RotationPlot(rsr, tickers):
    rs_slice = rsr[-26:]
    data = []
    ann = []
    Cols = ['rs_ratio-', 'rs_momentum-']
    for i in tickers[1:]:
        trace3 = go.Scatter(x = rs_slice[Cols[0]+i].values,
                            y = rs_slice[Cols[1]+i].values,
                            mode = 'lines+markers',
                            name = i,
                            text = rs_slice.index,
                            marker=dict(colorscale='Viridis',
                                        size = 10,
                                        opacity = .9)) 
        data.append(trace3)
        ann1 = dict(
                x=rs_slice[Cols[0]+i][-1],
                y=rs_slice[Cols[1]+i][-1],
                xref='x',
                yref='y',
                text=i,
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-20)
        ann2 = dict(
                x=rs_slice[Cols[0]+i][0],
                y=rs_slice[Cols[1]+i][0],
                xref='x',
                yref='y',
                text='',
                showarrow=True,
                arrowhead=5,
                ax=0,
                ay=-10)
        ann.append(ann1)
        ann.append(ann2)

        
    layout = go.Layout(
        title = 'Relative MACD Weekly - Indices vs SPY',
        titlefont=dict(family='Balto, sans-serif', size=30, color='black'),
        hovermode= 'closest',
        shapes=[{'line': {'color': 'rgb(180, 180, 180)', 'width': 1.5, 'dash': 'dashdot'},
                           'type': 'line',
                           'x0': 100,
                           'x1': 100,
                           'y0': 97,
                           'y1': 103},
               {'line': {'color': 'rgb(180, 180, 180)', 'width': 1.5, 'dash': 'dashdot'},
                           'type': 'line',
                           'x0': 98,
                           'x1': 103,
                           'y0': 100,
                           'y1': 100}],
        xaxis=dict(
            title='RS Ratio',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')),
        yaxis=dict(
            title='RS Momentum',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')))

    fig = go.Figure(data=data, layout=layout)

    fig['layout'].update(annotations = ann) 

    return fig  

def QuadrantPlot(df):
    data = []

    for i in np.arange(1,5,1):

        trace3 = go.Scatter(x = df['rs_ratio-'][df['Q-'] == i],
                                y = df['rreturns-'][df['Q-'] == i],
                                mode = 'markers',
                                name = str(i),
                                marker=dict(colorscale='Viridis',
                                            size = 10,
                                            opacity = .9)) 
        data.append(trace3)
        
    layout = go.Layout(
        title = 'Relative Ratio by Quadrant vs 1w Forward Rel SPY Returns Scatter',
        titlefont=dict(family='Balto, sans-serif', size=30, color='black'),
        hovermode= 'closest',
        xaxis=dict(
            title='RS Ratio',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')),
        yaxis=dict(
            title='Forward 1 Week Returns',
            showline=True,
            titlefont=dict(
                family='Balto, sans-serif',
                size=18,
                color='black')))

    fig = go.Figure(data=data, layout=layout)

    return fig

def PlotMultiClassROC(Y, pred, c_names):
    # Creates Multi-Class 

    n_classes = c_names.shape[0]
    if n_classes > 2:
        Y_b = label_binarize(Y, classes=c_names)
        P_b = label_binarize(pred, classes=c_names)
    else:
        Y_b = np.concatenate((label_binarize(Y,classes=c_names),
                             np.abs((label_binarize(Y,classes=c_names)-1))),
                             axis = 1)
        P_b = np.concatenate((label_binarize(pred,classes=c_names),
                             np.abs((label_binarize(pred,classes=c_names)-1))),
                             axis = 1)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_b[:, i], P_b[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_b.ravel(), P_b.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(c_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")

def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plotFeatImportances(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    #plot mean importance bars with std
    mpl.figure(figsize=(10,10))
    imp=imp.sort_values('mean', ascending=True)
    ax=imp['mean'].plot(kind='barh', color='b', alpha = .25, xerr=imp['std'], error_kw={'ecolor':'r'})
    if method=='MDI':
        mpl.xlim([0,imp.sum(axis=1).max()])
        mpl.axvline(1./imp.shape[0],linewidth=1,color='r',linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i,j in zip(ax.patches, imp.index) :ax.text(i.get_width()/2,
                                                   i.get_y()+i.get_height()/2,
                                                   j,
                                                   ha='center',
                                                   va='center',
                                                   color='black')
    mpl.title('tag='+str(tag)+' | simNum='+str(simNum)+' | oob='+str(round(oob,4))+ ' | oos='+str(round(oos,4)))
    mpl.savefig(pathOut+'featImportance_'+str(simNum)+str(tag)+'.png', dpi=100)
    plt.show()
    mpl.clf()
    mpl.close()
    return


# OHLC & High / Low Vol Functions

def ParkinsonsN(x, f):
    xHL = np.log(x['adjHigh']/x['adjLow'])
    parkNs = []
    for i in np.arange(0,x.shape[0]):
        if (i-f) < 0:
            parkNs.append(np.nan)
            continue
        xHL_r = xHL[(i-f):i]
        parkN = np.sqrt(np.power(np.sum(1/(4*np.log(2))*xHL_r),2)/xHL_r.shape[0])
        v = np.std(x['creturns'])*np.sqrt(252/f)
        parkNs.append(parkN/v) 
    return parkNs

## Dict for feature names

tick_ft = dict({
                'IWM': ['MeanRev-rescaled-IWM',
                        'MeanRev-rs_ratio-IWM',
                        'Trend-hBma-IWM',
                        'Trend-absBma-IWM',
                        'Trend-MA-Slope50-IWM',
                        'Trend-MA-Slope100-IWM'],
                'XLY': ['MeanRev-rescaled-XLY', 
                        'Trend-MA-Slope200-XLY',
                        'Trend-MA-Slope50-XLY',
                        'Trend-absBma-XLE'],
                'XLP': ['MeanRev-rs_ratio-XLP',
                        'MeanRev-3MrealizedXLP',
                        'MeanRev-rescaled-EWMA2-XLP',
                        'Trend-MA-Slope200-XLP',
                        'Trend-hBma-XLP'],
                'XLE': ['MeanRev-rs_ratio-XLE',
                        'Trend-MA-Slope200-XLE',
                        'Trend-hBma-XLE',
                        'Trend-absBma-XLE'],
                'XLF': ['MeanRev-3MrealizedXLF',
                        'MeanRev-rescaled-XLF',
                        'Trend-MA-Slope50-XLF',
                        'Trend-absBma-XLF'],
                'XLV': ['MeanRev-3MrealizedXLV',
                        'MeanRev-rescaled-XLV',
                        'Trend-absBma-XLV', 
                        'Trend-rs-rsm-XLV'],
                'XLI': ['MeanRev-rs_ratio-XLI', 
                        'MeanRev-3MrealizedXLI', 
                        'MeanRev-rescaled-XLI',
                        'Trend-MA-Slope50-XLI'],
                'XLB': ['MeanRev-3MrealizedXLB',
                        'MeanRev-rs_ratio-XLB',
                        'Trend-MA-Slope200-XLB', 
                        'Trend-hBma-XLB'],
                'XLK': ['MeanRev-rescaled-XLK',
                        'Trend-MA-Slope100-XLK',
                        'Trend-MA-Slope200-XLK',
                        'Trend-absBma-XLK'],
                'XLU': ['MeanRev-3MrealizedXLU',
                        'MeanRev-rescaled-XLU',
                        'MeanRev-rescaled-EWMA2-XLU',
                        'Trend-MA-Slope200-XLU']})

full_ft =  ['MeanRev-3Mrealized',
                   'MeanRev-rescaled-',
                   'MeanRev-rescaled-EWMA2-',
                   'MeanRev-rs_ratio-',
                   'Trend-absBma-',
                   'Trend-hBma-',
                   'Trend-rs-rsm-',
                   'Trend-MA-Slope200-',
                   'Trend-MA-Slope100-',
                   'Trend-MA-Slope50-']


