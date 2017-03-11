# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:28:42 2017

@author: Issac
"""

from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import ExtraTreesRegressor as ET
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import pandas as pd
import xgboost as xgb

class Stacking(object):
    
    def __init__(self):
        self.bst = None
        self.gb = None
        self.rf = None
        self.et = None       
        self.lr = None       
        self.rd = None
    
    def fit(self,X,y):
        kf = KFold(n_splits=2,random_state=0, shuffle=True)
        train_index, test_index = kf.split(X).next()
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        print 'In XGB'
        self.bst = model_XGB(X_train,y_train)
        print 'In GB'
        self.gb = model_GB(X_train,y_train)
        print 'In RF'
        self.rf = model_RF(X_train,y_train)
        print 'In ET'
        self.et = model_ET(X_train,y_train)
        print 'In Ridge'
        self.rd = model_Ridge(X_train,y_train)
        print 'In Stacking'
        X_layer2 = stacking_get_X_layer2(self.bst,self.gb,self.rf,self.et,self.rd,X_test)
        X_layer2.reset_index(drop=True)
        y_test.reset_index(drop=True)
        self.lr = model_XGB(X_layer2,y_test)
        
    def predict(self,X):
        X_layer2 = stacking_get_X_layer2(self.bst,self.gb,self.rf,self.et,self.rd,X)
        X_layer2 = xgb.DMatrix(X_layer2)
        y = self.lr.predict(X_layer2)
        return y
        

def model_GB(X,y):
    gb = GB(n_estimators=300)
    gb.fit(X,y)
    return gb
    

def model_RF(X,y):
    rf = RF(n_jobs=3,n_estimators=200)
    rf.fit(X,y)
    return rf
    
    
def model_ET(X,y):
    et = ET(n_jobs=3,n_estimators=200)
    et.fit(X,y)
    return et
    

def model_XGB(X,y):
    params = {'objective':'reg:linear','eta':0.1}
    dtrain = xgb.DMatrix( X, label=y )
    bst = xgb.train(params,dtrain,num_boost_round=1000)
    return bst
    
        
def stacking_get_X_layer2(bst,gb,rf,et,rd,X):
    dX = xgb.DMatrix(X)
    y_bst = pd.DataFrame( bst.predict(dX),index=X.index,columns=['y_bst'] )
    y_gb = pd.DataFrame( gb.predict(X),index=X.index,columns=['y_gb'] )
    y_rf = pd.DataFrame( rf.predict(X),index=X.index,columns=['y_rf'] )
    y_et = pd.DataFrame( et.predict(X),index=X.index,columns=['y_et'] )
    y_rd = pd.DataFrame( rd.predict(X),index=X.index,columns=['y_rd'] )
    X_layer2 =  pd.concat([y_bst,y_gb,y_rf,y_et,y_rd], axis=1, join_axes=[y_rf.index])
    return X_layer2

def model_Ridge(X,y):
    clf = Ridge(alpha=3.0,normalize=True)
    clf.fit(X,y) 
    return clf
    
    