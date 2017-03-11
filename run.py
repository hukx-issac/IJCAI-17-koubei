# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 18:47:05 2017

@author: Issac
"""

import os
import pandas as pd
from time import time
from model import Stacking,model_XGB
from preprocess import get_predict_result, mulExtractFeature
from send_email import sendResultByEmail

def main(cpu=3):
    # 数据根目录
    dataRoot = os.getcwd()+os.sep+'data'
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'
    
#==============================================================================
#     # 数据路径
#     dataPath_view = os.path.join(dataRoot,'user_view.txt')
#     dataPath_extra_view = os.path.join(dataRoot,'extra_user_view.txt')
#     dataPath_pay = os.path.join(dataRoot,'user_pay.txt')
#     dataPath_info = os.path.join(dataRoot,'shop_info.txt')
#     
#     # 读取数据
#     data_info = pd.read_csv(dataPath_info,header=None,names=['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name'])
#     data_pay = pd.read_csv(dataPath_pay,header=None,names=['user_id','shop_id','time_stamp'])
#     data_view = pd.read_csv(dataPath_view,header=None,names=['user_id','shop_id','time_stamp'])
#     data_extra_view = pd.read_csv(dataPath_extra_view,header=None,names=['user_id','shop_id','time_stamp'])
#     
#     del dataRoot,dataPath_view,dataPath_pay,dataPath_info
#     data_view = pd.concat([data_view,data_extra_view])
#         
#     extractFeature(data_info,data_pay,data_view)
#     predict_set()
#==============================================================================
#    mulExtractFeature()
    if cpu == 1:
        data_train = pd.read_csv(os.path.join(dataRoot_cache,'train_set.csv'))
    else:
        data_train = []
        for i in range(cpu):
            data_train.append(pd.read_csv(os.path.join(dataRoot_cache,'train_set%s.csv'%i)))
        data_train = pd.concat(data_train)
        data_train.reset_index(drop=True,inplace=True)
    data_predict = pd.read_csv(os.path.join(dataRoot_cache,'predict_set.csv'))
    
    X = data_train[range(0,98)]

    model_flow = Stacking()
    model_flow.fit(X,data_train[[-1]])
    result = get_predict_result(model_flow,data_predict)
    
#==============================================================================
#     model_flow = model_XGB(X,data_train[[-1]])
#     model_view = model_XGB(data_train_view[range(0,100)],data_train_view[[-1]])
#     a=pd.DataFrame(model_flow.get_fscore().items(),columns=['feature','importance']).sort_values('importance', ascending=False)
#     b=pd.DataFrame(model_view.get_fscore().items(),columns=['feature','importance']).sort_values('importance', ascending=False)
#     
#     X_predict = xgb.DMatrix(data_train[range(0,133)])
#     result = model_flow.predict(X_predict)
#      
#==============================================================================
    
if __name__ == '__main__':
    start = time()
    main()
    end = time()
    delta = end - start
    m, s = divmod(delta, 60)
    h, m = divmod(m, 60)
    content = '''
    程序运行时间:%02d小时%02d分%02d秒.\n
    F1 value for train set: {:.4f}.\n  
    best params:\n%s.
    '''.format(1) % (h,m,s,'****')
    print content
#    att_paths = [os.getcwd() + os.sep + 'output.csv']
    sendResultByEmail(content)
#    os.system('shutdown -s -t 60')    #关机  

