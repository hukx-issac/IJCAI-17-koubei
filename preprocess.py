# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 19:01:01 2017

@author: Issac
"""

import os
import pandas as pd
import datetime
import numpy as np
from tsfresh import extract_features
from multiprocessing import Process
from timeseries import stationarityTest, find_optimal_model, arimaModelCheck

dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'
startdate = '2016-03-01'

def extractFeature(feature_shop_basic,process_no,data_info=None,data_pay=None,data_view=None):
#==============================================================================
#     # 提取基本特征
#     date_feature()
#     weather_feathre()
#     tsfresh_feature()
#     transfer_time(data_pay);transfer_time(data_view)
#     count_consumer_flow(data_info,data_pay)
#     count_consumer_view(data_info,data_view)
#     shop_basic_feature(data_info)
#     predict_set(date='2016-11-01')
#     average_two_month_feature()
#==============================================================================
    
    consumer_flow = pd.read_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv'),index_col=0)
    consumer_view = pd.read_csv(os.path.join(dataRoot_cache,'consumer_view.csv'),index_col=0)    
    feature_date = pd.read_csv(os.path.join(dataRoot_cache,'feature_date.csv'),index_col=0)
    feature_weather = pd.read_csv(os.path.join(dataRoot_cache,'all_city_weather.csv'),index_col=0)
    timeseries_flow = pd.read_csv(os.path.join(dataRoot_cache,'timeseries_view.csv'),index_col=0)
    average_two_month = pd.read_csv(os.path.join(dataRoot_cache,'average_two_month.csv'),index_col=0)
#    feature_shop_basic = pd.read_csv(os.path.join(dataRoot_cache,'feature_shop_basic.csv'),index_col=0)
    
    consumer_view = consumer_view[188:]
    consumer_flow = consumer_flow[188:]
    
    train_set = []
    
    consumer_flow.index=map(pd.tslib.Timestamp,map(str,consumer_flow.index))
    consumer_view.index=map(pd.tslib.Timestamp,map(str,consumer_view.index))
    feature_date.index=map(pd.tslib.Timestamp,map(str,feature_date.index))
    feature_weather.index=map(pd.tslib.Timestamp,map(str,feature_weather.index))
    timeseries_flow.index=map(pd.tslib.Timestamp,map(str,timeseries_flow.index))
    average_two_month.index=map(pd.tslib.Timestamp,map(str,average_two_month.index))
    for shop_id in feature_shop_basic.index:
        shop_f = feature_shop_basic.loc[shop_id]
        flow_f = consumer_flow[[str(shop_id)]]
        view_f = consumer_view[[str(shop_id)]]
        to_0_day = -1
        washData(flow_f)
        for date in dateRange(startdate,'2016-10-31'):
            opening_today = opening(date,flow_f,shop_id)           
            if opening_today == 0:
                to_0_day = 0
                continue;
            temp = shop_f[:57].append(feature_date.loc[date])    # 商店基本特征、日期特征
            temp['rain'] = feature_weather.loc[date][shop_f[-1]]
            temp['timeseries'] = timeseries_flow[str(shop_id)].loc[date]
            
            lastmonth =  get_last_month_day(date)   # 与上月同一天相差天数
            yesterday = date - datetime.timedelta(1)
            before_yesterday = date - datetime.timedelta(2)
            
            
            last1weekday_flow = get_before_nday_folw(date,7,flow_f)  
            last2weekday_flow = get_before_nday_folw(date,14,flow_f)
            last3weekday_flow = get_before_nday_folw(date,21,flow_f)
            lastmonth_flow = get_before_nday_folw(date,lastmonth,flow_f)            
            before_yesterday_flow = get_before_nday_folw(date,2,flow_f) 
            yesterday_flow = get_before_nday_folw(date,1,flow_f)
            
            before_yesterday_view = view_f.loc[before_yesterday][str(shop_id)]
            yesterday_view = view_f.loc[yesterday][str(shop_id)]          
            if before_yesterday_view == 0:
                before_yesterday_view = 1.0
            k_yesterday = (float(yesterday_flow)/before_yesterday_view)*yesterday_view    # (昨日流量/前日浏览量)*昨日浏览量
            diff_flow = 2*yesterday_flow - before_yesterday_flow    # 2*昨日流量 - 前日流量
            average = yesterday_flow*0.7 + before_yesterday_flow*0.3
            
            if to_0_day != -1:
                to_0_day += 1
            
            today_flow = flow_f.loc[date][str(shop_id)]  # 今日流量
            if np.isnan(today_flow):
                today_flow = 0
            
            other_feature = []
            other_feature.append(lastmonth_flow)    # 上月同一天值  
            other_feature.append(last1weekday_flow)    # 上一周同一天流量
            other_feature.append( (last1weekday_flow+last2weekday_flow+last3weekday_flow)/3.0 )    # 前三周同一星期平均                            
            other_feature.append( average_n_day(flow_f,date,15) ) # 前15天平均
            other_feature.append( average_n_day(flow_f,date,7) ) # 前7天平均
            other_feature.append( average_n_day(flow_f,date,3) ) # 前3天平均
            other_feature.append(average)   # 昨天和前天加权平均
            
            other_feature.append(before_yesterday_flow - get_before_nday_folw(date,9,flow_f))    # 前天 - 前天上一周
            other_feature.append(yesterday_flow - get_before_nday_folw(date,8,flow_f))    # 昨天 - 昨天上一周           
            other_feature.append(last1weekday_flow - get_before_nday_folw(date,6,flow_f))    # 上一周流量 - 上一周第二天
            
            other_feature.append(before_yesterday_flow)    # 前日流量
            other_feature.append(yesterday_flow)    # 昨日流量
            
            other_feature.append( before_yesterday_view )  # 前日浏览量
            other_feature.append( yesterday_view ) # 昨日浏览量
            other_feature.append(k_yesterday)   # (昨日流量/前日浏览量)*昨日浏览量
            other_feature.append(diff_flow)  # 2*昨日流量 - 前日流量
            
            
            other_feature.append(to_0_day)  # 距离上一次null多少天
            other_feature.append((date - pd.tslib.Timestamp('2016-03-01 00:00:00')).days)  # 距训练集第一天多少天
            other_feature.append(average_two_month.loc[date][str(shop_id)])  # 距训练集第一天多少天
            
            other_feature.append(today_flow)   
            temp1 = temp.append( pd.Series(other_feature,index=['lastmonth_flow','last1weekday_flow','average_3_week','average_15_day','average_7_day','average_3_day',
                                                                'average','beforeyesterday_lastweekbeforeyesterday','yesterday_lastweekyesterday','lastweek_lastweektomorrow',                                                              
                                                                'before_yesterday_flow','yesterday_flow','before_yesterday_view','yesterday_view','k_yesterday','diff_flow','to_0_day','to_begin_day','average_two_month','y_flow']) )           
            train_set.append(temp1)
    train_set = pd.DataFrame(train_set)

    train_set.to_csv(os.path.join(dataRoot_cache,'train_set%s.csv'%process_no),index=False)   
    


def to_0_day_feature():
    consumer_flow = pd.read_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv'),index_col=0)
    consumer_flow.index=map(pd.tslib.Timestamp,map(str,consumer_flow.index))
    to_0_day_p = []
    for shop_id in range(1,2001):
        to_0_day = -1
        flow_f = consumer_flow[[str(shop_id)]]
        for date in dateRange(startdate,'2016-10-31'):
            opening_today = opening(date,flow_f,shop_id)           
            if opening_today == 0:
                to_0_day = 0
                continue;
            if to_0_day != -1:
                to_0_day += 1
            if date == pd.tslib.Timestamp('2016-10-31 00:00:00'):
                to_0_day_p.append(to_0_day)
    to_0_day_p = pd.DataFrame(to_0_day_p)
    to_0_day_p.index = range(1,2001)
    to_0_day_p = to_0_day_p.T
    to_0_day_p.to_csv(os.path.join(dataRoot_cache,'to_0_day_p.csv'),index=False)   
            


def average_two_month_feature():    
    consumer_flow = pd.read_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv'),index_col=0)
    consumer_flow = consumer_flow[188:]
    consumer_flow.index=map(pd.tslib.Timestamp,map(str,consumer_flow.index))
    average_two_month = {}
    for shop_id in range(1,2001):
        flow_f = consumer_flow[[str(shop_id)]]
        temp = []
        for date in dateRange(startdate,'2016-10-31'):
            temp.append(averageTwoMonth(flow_f,date))
        average_two_month[shop_id] = temp
    average_two_month = pd.DataFrame(average_two_month,index=dateRange(startdate,'2016-10-31') ) 
    average_two_month.to_csv(os.path.join(dataRoot_cache,'average_two_month.csv'))   
        
        
    

def averageTwoMonth(flow_f,date):
    onedaypaylist = [get_before_nday_folw_null(date,n,flow_f) for n in [56,49,42,35,28,21,14,7]]
    onedaypaylist = filter(lambda x: x is not None, onedaypaylist)                 
    listNum = len(onedaypaylist)
    cmplist = []
    #3个以上的数据预测，3个以下的求平均值
    if listNum > 3:
        end1 = onedaypaylist[listNum-1]
        end2 = onedaypaylist[listNum-2]
        narry = np.array(onedaypaylist)
        onedaypaylist.remove(narry.max())
        onedaypaylist.remove(narry.min())
        mean = np.mean(onedaypaylist)
        for payNum in onedaypaylist:
            meanvar = np.abs(payNum - mean) / mean
            if meanvar < 0.15:
                cmplist.append(payNum)
        cmppayarry = np.array(cmplist)
        if not cmppayarry.any() :
            cmpmean = (end1 + end2) / 2
        else:
            cmpmean = (np.mean(cmppayarry) + 1.02*end1 + 1.02*end2) / 3
    elif 0<listNum<3:
        cmpmean = np.mean(onedaypaylist)
    else:
        return flow_f.mean()[0]
    return cmpmean


def get_before_nday_folw_null(date,n,flow_f):
    before_nday = date - datetime.timedelta(n)
    nday_flow = flow_f.loc[before_nday][0]
    if np.isnan(nday_flow):
        return None
    else:
        return nday_flow
    
    
def washData(flow_f):
    des_flow = flow_f.describe()
    IQR = (des_flow.loc['75%'] - des_flow.loc['25%'])
    up = des_flow.loc['75%'] + 1.5*IQR
    low = des_flow.loc['25%'] - 1.5*IQR
    special = [pd.tslib.Timestamp('2016-03-08 00:00:00'),pd.tslib.Timestamp('2016-04-01 00:00:00'),pd.tslib.Timestamp('2016-04-02 00:00:00'),pd.tslib.Timestamp('2016-04-03 00:00:00'),pd.tslib.Timestamp('2016-04-04 00:00:00'),pd.tslib.Timestamp('2016-04-05 00:00:00'),
               pd.tslib.Timestamp('2016-04-29 00:00:00'),pd.tslib.Timestamp('2016-04-30 00:00:00'),pd.tslib.Timestamp('2016-05-01 00:00:00'),pd.tslib.Timestamp('2016-05-02 00:00:00'),pd.tslib.Timestamp('2016-05-03 00:00:00'),pd.tslib.Timestamp('2016-06-01 00:00:00'),
               pd.tslib.Timestamp('2016-06-08 00:00:00'),pd.tslib.Timestamp('2016-06-09 00:00:00'),pd.tslib.Timestamp('2016-06-10 00:00:00'),pd.tslib.Timestamp('2016-06-11 00:00:00'),pd.tslib.Timestamp('2016-06-12 00:00:00'),pd.tslib.Timestamp('2016-08-09 00:00:00'),
               pd.tslib.Timestamp('2016-09-14 00:00:00'),pd.tslib.Timestamp('2016-09-15 00:00:00'),pd.tslib.Timestamp('2016-09-16 00:00:00'),pd.tslib.Timestamp('2016-09-17 00:00:00'),pd.tslib.Timestamp('2016-09-18 00:00:00'),pd.tslib.Timestamp('2016-09-30 00:00:00'),
               pd.tslib.Timestamp('2016-10-01 00:00:00'),pd.tslib.Timestamp('2016-10-02 00:00:00'),pd.tslib.Timestamp('2016-10-03 00:00:00'),pd.tslib.Timestamp('2016-10-04 00:00:00'),pd.tslib.Timestamp('2016-10-05 00:00:00'),pd.tslib.Timestamp('2016-10-06 00:00:00'),
               pd.tslib.Timestamp('2016-10-07 00:00:00'),pd.tslib.Timestamp('2016-10-08 00:00:00'),pd.tslib.Timestamp('2016-10-09 00:00:00'),pd.tslib.Timestamp('2016-11-11 00:00:00'),
               pd.tslib.Timestamp('2016-01-01 00:00:00'),pd.tslib.Timestamp('2016-01-02 00:00:00'),pd.tslib.Timestamp('2016-01-03 00:00:00'),pd.tslib.Timestamp('2016-01-04 00:00:00'),pd.tslib.Timestamp('2016-02-06 00:00:00'),pd.tslib.Timestamp('2016-02-07 00:00:00'),
               pd.tslib.Timestamp('2016-02-08 00:00:00'),pd.tslib.Timestamp('2016-02-09 00:00:00'),pd.tslib.Timestamp('2016-02-10 00:00:00'),pd.tslib.Timestamp('2016-02-11 00:00:00'),pd.tslib.Timestamp('2016-02-12 00:00:00'),pd.tslib.Timestamp('2016-02-13 00:00:00'),
               pd.tslib.Timestamp('2016-02-14 00:00:00'),pd.tslib.Timestamp('2016-02-22 00:00:00')]
    a = flow_f[flow_f>up].dropna()
    b = flow_f[flow_f<low].dropna()
    for date in a.index:
        if date in special:
            a.drop([date],axis=0,inplace=True)
    flow_f.loc[a.index] = des_flow.loc['mean'][0]
    flow_f.loc[b.index] = des_flow.loc['mean'][0]
 
    
def predict_view():     
    consumer_view = pd.read_csv(os.path.join(dataRoot_cache,'consumer_view.csv'),index_col=0)
    consumer_view = consumer_view[310:]
    consumer_view.index=map(pd.tslib.Timestamp,map(str,consumer_view.index))
    for date in dateRange('2016-11-01','2016-11-14'):
         b1d = date - datetime.timedelta(7)
         b2d = date - datetime.timedelta(14)
         b3d = date - datetime.timedelta(21)
         b1d_view = consumer_view.loc[b1d]
         b2d_view = consumer_view.loc[b2d]
         b3d_view = consumer_view.loc[b3d]
         today_view = (b1d_view+b2d_view+b3d_view)/3.0
         today_view.name = date
         consumer_view = consumer_view.append(today_view)      
    consumer_view.to_csv(os.path.join(dataRoot_cache,'predict_view.csv'))       

    
def mulExtractFeature(data_info=None,data_pay=None,data_view=None,cpu=3):
    feature_shop_basic = pd.read_csv(os.path.join(dataRoot_cache,'feature_shop_basic.csv'),index_col=0)
    if cpu == 1:
        extractFeature(feature_shop_basic,'all')
    else:       
        d = int(feature_shop_basic.shape[0]/cpu)
        processes = []
        for i in range(cpu):
            if i == (cpu-1):
                p = Process(target=extractFeature,\
                            args=(feature_shop_basic[d*i:],i))
            else:
                p = Process(target=extractFeature,\
                            args=(feature_shop_basic[d*i:d*(i+1)],i))
            processes.append(p)
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()    
    
    
def get_before_nday_folw(date,n,flow_f):
    before_nday = date - datetime.timedelta(n)
    nday_flow = flow_f.loc[before_nday][0]
    if np.isnan(nday_flow):
        return flow_f.mean()[0]
    else:
        return nday_flow
    
def average_n_day(flow_f,date,n):
    temp = flow_f.fillna(flow_f.mean()[0])
    a = temp.loc[date -  datetime.timedelta(n):date -  datetime.timedelta(1)]
    return a.mean()[0]
        
def get_last_month_day(date):
    if date.month == 1:
        year = date.year - 1
        month = 12
    else:
        year = date.year
        month = date.month-1
    days_in_lastmonth = pd.tslib.Timestamp(year,month,1).daysinmonth
    if date.day > days_in_lastmonth:
        return (date - pd.tslib.Timestamp(year,month,days_in_lastmonth)).days
    else:
        return (date - pd.tslib.Timestamp(year,month,date.day)).days
    
       
def opening(date,flow_f,shop_id):   #   判断是否正常营业
    yesterday = date - datetime.timedelta(1)
    tomorrow = date + datetime.timedelta(1)    
    today_flow = flow_f.loc[date].isnull()[str(shop_id)]  # 今日流量是否为空    
    if date ==  pd.tslib.Timestamp('2016-10-31 00:00:00'):
        return 1
    elif today_flow:
        yesterday_flow = flow_f.loc[yesterday].isnull()[str(shop_id)]   # 昨日流量是否为空    
        tomorrow_flow = flow_f.loc[tomorrow].isnull()[str(shop_id)]  # 明日流量是否为空    
        if (tomorrow_flow) or (yesterday_flow):
            return 0
        else:
            return 1
    else:
        return 1
    
    
def predict_set():   
    to_0_day_p = pd.read_csv(os.path.join(dataRoot_cache,'to_0_day_p.csv'))
    feature_date = pd.read_csv(os.path.join(dataRoot_cache,'feature_date.csv'),index_col=0)
    feature_shop_basic = pd.read_csv(os.path.join(dataRoot_cache,'feature_shop_basic.csv'),index_col=0)
    feature_weather = pd.read_csv(os.path.join(dataRoot_cache,'all_city_weather.csv'),index_col=0)
    timeseries_flow = pd.read_csv(os.path.join(dataRoot_cache,'timeseries_flow.csv'),index_col=0)
    timeseries_flow.index=map(pd.tslib.Timestamp,map(str,timeseries_flow.index))
    feature_date.index=map(pd.tslib.Timestamp,map(str,feature_date.index))
    feature_weather.index=map(pd.tslib.Timestamp,map(str,feature_weather.index))
    predict_set = []
    
    for date in dateRange('2016-11-01','2016-11-14'):
        for shop_id in range(1,2001):
            other_feature = []
            shop_f = feature_shop_basic.loc[shop_id]
            temp = shop_f[:57].append(feature_date.loc[date])
            temp['rain'] = feature_weather.loc[date][shop_f[-1]]
            temp['timeseries'] = timeseries_flow[str(shop_id)].loc[date]
            last = to_0_day_p.loc[0][str(shop_id)]
            if last == -1:
                to_0_day = last
            else:
                to_0_day = last +  (date - pd.tslib.Timestamp('2016-10-31 00:00:00')).days
            to_begin_day = (date - pd.tslib.Timestamp('2016-03-01 00:00:00')).days
            other_feature = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,to_0_day,to_begin_day,None]
            temp = temp.append( pd.Series(other_feature,index=['lastmonth_flow','last1weekday_flow','average_3_week','average_15_day','average_7_day','average_3_day',
                                                                'average','beforeyesterday_lastweekbeforeyesterday','yesterday_lastweekyesterday','lastweek_lastweektomorrow',                                                              
                                                                'before_yesterday_flow','yesterday_flow','before_yesterday_view','yesterday_view','k_yesterday','diff_flow','to_0_day','to_begin_day','average_two_month']) )
            predict_set.append(temp)
    predict_set = pd.DataFrame(predict_set)
    predict_set.to_csv(os.path.join(dataRoot_cache,'predict_set.csv'),index=False)

    
def timeseries_feature():
    data_flow = pd.read_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv'),index_col=0)
    y_predicted = pd.DataFrame()
    data_flow = data_flow.fillna(data_flow.describe().loc['mean'])
    data_flow.index=map(pd.tslib.Timestamp,map(str,data_flow.index))
  
    for id in data_flow.columns:
        flow = data_flow[id]
        # 平稳性检验
        (d_flow,p_flow) = stationarityTest(flow)  
        if d_flow > 20 :
            d_flow = 20
            print id
        p_flow, q_flow = find_optimal_model(flow,3,3,d_flow)
        arima_flow = arimaModelCheck(flow,p_flow, q_flow,d_flow)
        if d_flow == 0 :
            y_predicted[id] = arima_flow.predict(startdate,'20161114')
        else:
            y_predicted[id] = arima_flow.predict(startdate,'20161114',typ='levels')
        y_predicted.to_csv(os.path.join(dataRoot_cache,'timeseries_flow.csv'))
            
            
# 商家基本特征       
def shop_basic_feature(data_info):
    data_info['cate_3_name'].fillna(data_info['cate_2_name'],inplace=True)
    ts_col = ['y_flow__mean_second_derivate_central','y_flow__longest_strike_below_mean','y_flow__percentage_of_reoccurring_datapoints_to_all_datapoints',
              'y_flow__mean_abs_change_quantiles__qh_1.0__ql_0.8','y_flow__number_cwt_peaks__n_5','y_flow__number_cwt_peaks__n_1','y_flow__fft_coefficient__coeff_8','y_flow__time_reversal_asymmetry_statistic__lag_3','y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_12__w_2',
              'y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_6__w_2','y_flow__mean_abs_change_quantiles__qh_0.6__ql_0.2','y_flow__mean_abs_change_quantiles__qh_0.8__ql_0.0','y_flow__autocorrelation__lag_8','y_flow__spkt_welch_density__coeff_8','y_flow__sum_of_reoccurring_values','y_flow__fft_coefficient__coeff_3','y_flow__ar_coefficient__k_10__coeff_1',
              'y_flow__fft_coefficient__coeff_2','y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_1__w_2','y_flow__time_reversal_asymmetry_statistic__lag_1','y_flow__quantile__q_0.9','y_flow__ar_coefficient__k_10__coeff_0','y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_2__w_2',
              'y_flow__mean_change','y_flow__sum_of_reoccurring_data_points','y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_9__w_2','y_flow__autocorrelation__lag_2','y_flow__skewness','y_flow__last_location_of_maximum','y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_13__w_2','y_flow__ar_coefficient__k_10__coeff_2',
              'y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_14__w_2','y_flow__median','y_flow__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_0__w_20','y_flow__mean_abs_change_quantiles__qh_1.0__ql_0.4','y_flow__time_reversal_asymmetry_statistic__lag_2','y_flow__autocorrelation__lag_4','y_flow__longest_strike_above_mean']
    feature_shop_basic = []
    consumer_flow_null = pd.read_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv'),index_col=0)
    consumer_view_null = pd.read_csv(os.path.join(dataRoot_cache,'consumer_view.csv'),index_col=0)
    shop_user_feature = pd.read_csv(os.path.join(dataRoot_cache,'shop_user_feature.csv'),index_col=0)
    tsfresh_feature = pd.read_csv(os.path.join(dataRoot_cache,'tsfresh_feature_all.csv'),index_col=0)
    tsfresh_feature = tsfresh_feature[ts_col]
    h2p = pd.read_csv(os.path.join(os.getcwd()+os.sep+'data'+os.sep+'city_weather','pingyin.csv'))
    consumer_flow_null = consumer_flow_null.iloc[244:]
    consumer_view_null = consumer_view_null.iloc[244:]
    flow_des = consumer_flow_null.describe()
    view_des = consumer_view_null.describe()
    location_num = data_info.groupby('location_id').count()
    for shop in data_info.itertuples(index=False):
        shop_user_f = shop_user_feature.loc[shop[0]]
        flow_f = consumer_flow_null[[str(shop[0])]]
        view_f = consumer_view_null[[str(shop[0])]]
        temp = []
        temp.append(shop[0])  # shop_id
        temp.append(shop[3])  # 人均消费
        temp.append(shop[4])  # 评分
        temp.append(shop[5])  # 评论数
        temp.append(shop[6])  # 门店等级
        temp.append(shop_user_f[0])  # rate_vp_v
        temp.append(shop_user_f[1])  # rate_vp_p
        temp.append(shop_user_f[2])  # rate_pay_two        
#        temp.append(city_level(shop[1]))   # 所在城市分类 
        temp.append(flow_des[str(shop[0])]['mean']) # 商家日流量均值
        temp.append(flow_des[str(shop[0])]['std']) # 商家日流量方差
        temp.append(view_des[str(shop[0])]['mean']) # 商家日浏览均值
        temp.append(view_des[str(shop[0])]['std']) # 商家日浏览方差
        temp.append(flow_des[str(shop[0])]['max'])
        temp.append(flow_des[str(shop[0])]['min'])
        temp.append(flow_f.sum()[str(shop[0])])
        temp.append(view_des[str(shop[0])]['max'])
        temp.append(flow_f[flow_f[str(shop[0])]>0].shape[0])       # 商家正常营业天数 
        k = (view_f.sum()/flow_f.sum())[str(shop[0])]  # 浏览总量/流量总量
        temp.append(k)
        temp.append(cate(shop[7]))     
        temp.append(location_num.loc[shop[2]]['shop_id'])
#        region = h2p[h2p['city_c']==shop[1]].iloc[0]['diqu']
#        temp.append(region) 
        city_pingyin = h2p[h2p['city_c']==shop[1]].iloc[0]['city_p']
        temp.append( city_pingyin)
        feature_shop_basic.append(temp)
    feature_shop_basic = pd.DataFrame(feature_shop_basic,columns=['shop_id','per_pay','score','comment_cnt','shop_level','rate_vp_v','rate_vp_p','rate_pay_two','mean_flow',
                                                                  'std_flow','mean_view','std_view','max_flow','min_flow','sum_flow','max_view','opening_day','k','cate','location_num','city_pingyin'])
    feature_shop_basic.set_index('shop_id',inplace=True)
#    des_shop = feature_shop_basic.describe()
#    feature_shop_basic.fillna(des_shop.loc['mean'],inplace=True)
    feature_shop_basic = feature_shop_basic.join(tsfresh_feature)
    b = feature_shop_basic.pop('city_pingyin')
    feature_shop_basic.insert(57,'city_pingyin',b)
#    feature_shop_basic = oneHot(feature_shop_basic)  
#    feature_shop_basic.to_csv(os.path.join(dataRoot_cache,'feature_shop_basic.csv'))
    nearstPolation(feature_shop_basic)


def oneHot(originData):
    data = pd.DataFrame(index=originData.index)
    deal_col = ['cate','region']
    for col, col_data in originData.iteritems():
        if col in deal_col:
            col_data = pd.get_dummies(col_data,prefix=col)
        data = data.join(col_data)
    return data
    
    
def cate(c):
    if c == '美食':
        return 1
    else:
        return 0


    
# 生成预测结果
def get_predict_result(model_flow,data_predict):
    consumer_flow = pd.read_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv'),index_col=0)
    consumer_flow.index=map(pd.tslib.Timestamp,map(str,consumer_flow.index))
    consumer_view = pd.read_csv(os.path.join(dataRoot_cache,'predict_view.csv'),index_col=0)
    consumer_view.index=map(pd.tslib.Timestamp,map(str,consumer_view.index))
    consumer_flow = consumer_flow[188:]   
    consumer_flow.fillna(consumer_flow.mean(),inplace=True)
    result_flow = consumer_flow.copy()
                                     
    for i,date in enumerate( dateRange('2016-11-01','2016-11-14') ):
        X_predict = data_predict[i*2000:(i+1)*2000]
        X_predict.reset_index(drop=True,inplace=True)
        average_two_month = []
        for i in range(1,2001):
            average_two_month.append(averageTwoMonth(result_flow[[str(i)]],date))
        
        lastmonth =  date - datetime.timedelta(get_last_month_day(date))
        last1weekday = date - datetime.timedelta(7)
        last2weekday = date - datetime.timedelta(14)
        last3weekday = date - datetime.timedelta(21)  
        
        average_15_day = result_flow.loc[date -  datetime.timedelta(15):date -  datetime.timedelta(1)].mean()
        average_7_day = result_flow.loc[date -  datetime.timedelta(7):date -  datetime.timedelta(1)].mean()
        average_3_day = result_flow.loc[date -  datetime.timedelta(3):date -  datetime.timedelta(1)].mean()
        
        
        yesterday = date - datetime.timedelta(1)
        before_yesterday = date - datetime.timedelta(2)
        before_yesterday_flow = result_flow.loc[before_yesterday]        
        yesterday_flow = result_flow.loc[yesterday]
        
        lastmonth_flow = result_flow.loc[lastmonth]
        last1weekday_flow = result_flow.loc[last1weekday]
        last2weekday_flow = result_flow.loc[last2weekday]  
        last3weekday_flow = result_flow.loc[last3weekday]
        beforeyesterday_lastweekbeforeyesterday = before_yesterday_flow - result_flow.loc[before_yesterday - datetime.timedelta(7)]
        yesterday_lastweekyesterday = yesterday_flow - result_flow.loc[yesterday - datetime.timedelta(7)]
           
        before_yesterday_view = consumer_view.loc[yesterday]   
        yesterday_view = consumer_view.loc[yesterday]   
        
        lastweek_lastweektomorrow = last1weekday_flow - result_flow.loc[last1weekday + datetime.timedelta(1)]
        lastweek_view = consumer_view.loc[last1weekday]
        before_yesterday_flow.reset_index(drop=True,inplace=True)
        before_yesterday_view.reset_index(drop=True,inplace=True)
        yesterday_flow.reset_index(drop=True,inplace=True)
        yesterday_view.reset_index(drop=True,inplace=True)        
        last1weekday_flow.reset_index(drop=True,inplace=True)        
        last2weekday_flow.reset_index(drop=True,inplace=True)        
        last3weekday_flow.reset_index(drop=True,inplace=True)        
        lastmonth_flow.reset_index(drop=True,inplace=True)              
        average_15_day.reset_index(drop=True,inplace=True)        
        average_7_day.reset_index(drop=True,inplace=True)        
        average_3_day.reset_index(drop=True,inplace=True)        
        beforeyesterday_lastweekbeforeyesterday.reset_index(drop=True,inplace=True)        
        yesterday_lastweekyesterday.reset_index(drop=True,inplace=True)        
        lastweek_lastweektomorrow.reset_index(drop=True,inplace=True)        
        lastweek_view.reset_index(drop=True,inplace=True)             
        
        X_predict['lastmonth_flow'].fillna(lastmonth_flow,inplace=True)
        X_predict['last1weekday_flow'].fillna(last1weekday_flow,inplace=True)
        X_predict['average_3_week'].fillna((last1weekday_flow + last2weekday_flow + last3weekday_flow)/3.0,inplace=True)
        X_predict['average_15_day'].fillna(average_15_day,inplace=True)
        X_predict['average_7_day'].fillna(average_7_day,inplace=True)
        X_predict['average_3_day'].fillna(average_3_day,inplace=True)
        
        X_predict['beforeyesterday_lastweekbeforeyesterday'].fillna(beforeyesterday_lastweekbeforeyesterday,inplace=True)
        X_predict['yesterday_lastweekyesterday'].fillna(yesterday_lastweekyesterday,inplace=True)
        
        X_predict['lastweek_lastweektomorrow'].fillna(lastweek_lastweektomorrow,inplace=True)
        X_predict['before_yesterday_flow'].fillna(before_yesterday_flow,inplace=True)
        X_predict['yesterday_flow'].fillna(yesterday_flow,inplace=True)
        X_predict['before_yesterday_view'].fillna(before_yesterday_view,inplace=True)
        X_predict['yesterday_view'].fillna(yesterday_view,inplace=True)
        a = X_predict['before_yesterday_view'].copy()
        a.replace(0,1,inplace=True)
        X_predict['k_yesterday'] = (X_predict['yesterday_flow'] / map(float,a))*X_predict['yesterday_view']    # (昨日流量/前日浏览量)*昨日浏览量
        X_predict['diff_flow'] = 2.0 * X_predict['yesterday_flow'] - X_predict['before_yesterday_flow']   # 2*昨日流量 - 前日流量        
        X_predict['average'] = X_predict['yesterday_flow']*0.7 + X_predict['before_yesterday_flow']*0.3
        X_predict['average_two_month'] = average_two_month
         
        y_predict_flow = model_flow.predict(X_predict)
        y_predict_flow = map(adjust_result,map(round,y_predict_flow))
        y_predict_flow = pd.DataFrame(y_predict_flow,columns =[date] ,index=map(str,range(1,2001))).T
      
        result_flow = result_flow.append(y_predict_flow)
    result_flow = result_flow.T[range(301,315)]
    result_flow.to_csv(os.path.join(os.getcwd(),'output.csv'),header=None)
    return result_flow
   
def adjust_result(p):
    if p < 0:
        return 0
    else:
        return int(p)
        
        
# 计算历史客流量            
def count_consumer_flow(data_info,data_pay):
    dates = dateRange('2015-07-01','2016-10-31')
    consumer_flow = pd.DataFrame(index=dates)
    for shop in data_info.itertuples(index=False):
        shop_pay = data_pay[data_pay['shop_id']==shop[0]]   # 单个商家账单
        consumer_flow[shop[0]] = shop_pay.groupby('time_stamp').count()['shop_id'] # 某商家某日客流量
    consumer_flow.to_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv')) 
    consumer_flow = consumer_flow.fillna(0)
    consumer_flow.to_csv(os.path.join(dataRoot_cache,'consumer_flow.csv'))    

# 计算历史浏览次数            
def count_consumer_view(data_info,data_view):
    dates = dateRange('2015-07-01','2016-10-31')
    consumer_view = pd.DataFrame(index=dates)
    for shop in data_info.itertuples(index=False):
        shop_view = data_view[data_view['shop_id']==shop[0]]   # 单个商家浏览行为
        consumer_view[shop[0]] = shop_view.groupby('time_stamp').count()['shop_id'] # 某商家某日浏览量  
    consumer_view.to_csv(os.path.join(dataRoot_cache,'consumer_view_null.csv'))     
    consumer_view=consumer_view.fillna(0)
    consumer_view.to_csv(os.path.join(dataRoot_cache,'consumer_view.csv'))    
                        
def dateRange(beginDate,endDate):
    dates = []
    dt = datetime.datetime.strptime(beginDate,"%Y-%m-%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(pd.tslib.Timestamp(dt))
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y-%m-%d")
    return dates
        
def transfer_time(data):
    data['time_stamp'] = map(lambda x : pd.tslib.Timestamp(x[0:10]),data['time_stamp'])

    
def weather_feathre():
    dates = dateRange('2015-07-01','2016-11-14')
    root = os.getcwd()+os.sep+'data'+os.sep+'city_weather'
    h2p = pd.read_csv(os.path.join(root,'pingyin.csv'))
    all_city_weather = pd.DataFrame(index = dates)
    for city in h2p['city_p']:
        city_weather = pd.read_csv(os.path.join(root,city),header=None,index_col=0)
        all_city_weather[city] = pd.Series(map(rain,city_weather[3][:503]),index=dates)
    all_city_weather.to_csv(os.path.join(dataRoot_cache,'all_city_weather.csv'))
        
def rain(hanzi):
    if ('雨' in hanzi) or ('雪' in hanzi):
        return 1
    else:
        return 0        


# 日期特征
def date_feature():
    dates = dateRange('2016-01-01','2016-11-14')
    feature_date = []
    for date in dates:
        temp = [0,0,0,0,0,0,0, 0,0,0, 0,0,0,0,0,0,0,0,0,0]   # 10休假前最后一天上班、11休假第一天 、12休假最后一天、13休假、14休假后第一天上班、15补班、16特殊日子、七八月暑假 、31天、上半年
        temp[date.weekday()] = 1   #星期特征
        if date.day < 11:
             temp[7] = 1
        elif date.day>20:
             temp[9] = 1
        else:
             temp[8] = 1
        if (date > pd.tslib.Timestamp('2016-06-20 00:00:00')) and (date < pd.tslib.Timestamp('2016-09-10 00:00:00')):
            temp[17] = 1
        if (date.month in [1,3,5,7,8,10,12]):
            temp[18] = 1
        if (date.month in [1,2,3,4,5,6]):
            temp[19] = 1
        feature_date.append(temp)
    feature_date = pd.DataFrame(feature_date,index=dates,columns=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','first','mid','last','last_day_work','fist_vacation','last_vacation','vacation','first_day_work','not_vacation','festival','summer_vacation','day_31','up_year'])
    feature_date.loc[pd.tslib.Timestamp('2016-01-01 00:00:00')] = [0,0,0,0,1,0,0,1,0,0, 0,1,0,1,0,0,1,0,1,1]  # 元旦
    feature_date.loc[pd.tslib.Timestamp('2016-01-02 00:00:00')] = [0,0,0,0,0,1,0,1,0,0, 0,0,0,1,0,0,0,0,1,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-01-03 00:00:00')] = [0,0,0,0,0,0,1,1,0,0, 0,0,1,1,0,0,0,0,1,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-01-04 00:00:00')] = [1,0,0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,1,1]  
                     
    feature_date.loc[pd.tslib.Timestamp('2016-02-06 00:00:00')] = [0,0,0,0,0,1,0,1,0,0, 1,0,0,0,0,1,0,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-07 00:00:00')] = [0,0,0,0,0,0,1,1,0,0, 0,1,0,1,0,0,1,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-08 00:00:00')] = [1,0,0,0,0,0,0,1,0,0, 0,0,0,1,0,0,1,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-09 00:00:00')] = [0,1,0,0,0,0,0,1,0,0, 0,0,0,1,0,0,1,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-10 00:00:00')] = [0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,1,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-11 00:00:00')] = [0,0,0,1,0,0,0,1,0,0, 0,0,0,1,0,0,1,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-12 00:00:00')] = [0,0,0,0,1,0,0,1,0,0, 0,0,0,1,0,0,1,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-13 00:00:00')] = [0,0,0,0,0,1,0,1,0,0, 0,0,1,1,0,0,1,0,0,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-02-14 00:00:00')] = [0,0,0,0,0,0,1,1,0,0, 0,0,0,0,1,1,1,0,0,1]
                                           
    feature_date.loc[pd.tslib.Timestamp('2016-02-22 00:00:00')]['festival'] = 1  # 元宵
    
    feature_date.loc[pd.tslib.Timestamp('2016-03-08 00:00:00')]['festival'] = 1  # 妇女节
    feature_date.loc[pd.tslib.Timestamp('2016-04-01 00:00:00')] = [0,0,0,0,1,0,0,1,0,0, 1,0,0,0,0,0,0,0,0,1]  # 清明
    feature_date.loc[pd.tslib.Timestamp('2016-04-02 00:00:00')] = [0,0,0,0,0,1,0,1,0,0, 0,1,0,1,0,0,0,0,0,1]
    feature_date.loc[pd.tslib.Timestamp('2016-04-03 00:00:00')] = [0,0,0,0,0,0,1,1,0,0, 0,0,0,1,0,0,0,0,0,1]
    feature_date.loc[pd.tslib.Timestamp('2016-04-04 00:00:00')] = [1,0,0,0,0,0,0,1,0,0, 0,0,1,1,0,0,1,0,0,1]
    feature_date.loc[pd.tslib.Timestamp('2016-04-05 00:00:00')] = [0,1,0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,0,1]
                     
    feature_date.loc[pd.tslib.Timestamp('2016-04-29 00:00:00')] = [0,0,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,0,1]  # 五一
    feature_date.loc[pd.tslib.Timestamp('2016-04-30 00:00:00')] = [0,0,0,0,0,1,0,0,0,1, 0,1,0,1,0,0,0,0,0,1] 
    feature_date.loc[pd.tslib.Timestamp('2016-05-01 00:00:00')] = [0,0,0,0,0,0,1,1,0,0, 0,0,0,1,0,0,1,0,1,1]  
    feature_date.loc[pd.tslib.Timestamp('2016-05-02 00:00:00')] = [1,0,0,0,0,0,0,1,0,0, 0,0,1,1,0,0,0,0,1,1] 
    feature_date.loc[pd.tslib.Timestamp('2016-05-03 00:00:00')] = [0,1,0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0,1,1]
                     
    feature_date.loc[pd.tslib.Timestamp('2016-06-01 00:00:00')]['festival'] = 1  # 六一
    
    feature_date.loc[pd.tslib.Timestamp('2016-06-08 00:00:00')] = [0,0,1,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,0,0,1]  # 端午
    feature_date.loc[pd.tslib.Timestamp('2016-06-09 00:00:00')] = [0,0,0,1,0,0,0,1,0,0, 0,1,0,1,0,0,1,0,0,1]
    feature_date.loc[pd.tslib.Timestamp('2016-06-10 00:00:00')] = [0,0,0,0,1,0,0,1,0,0, 0,0,0,1,0,0,0,0,0,1]
    feature_date.loc[pd.tslib.Timestamp('2016-06-11 00:00:00')] = [0,0,0,0,0,1,0,0,1,0, 0,0,1,1,0,0,0,0,0,1]
    feature_date.loc[pd.tslib.Timestamp('2016-06-12 00:00:00')] = [0,0,0,0,0,0,1,0,1,0, 0,0,0,0,1,1,0,0,0,1]
                     
    feature_date.loc[pd.tslib.Timestamp('2016-08-09 00:00:00')] = [0,1,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,1,1,1,0]  # 七夕
    
    feature_date.loc[pd.tslib.Timestamp('2016-09-14 00:00:00')] = [0,0,1,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,0,0]  #中秋
    feature_date.loc[pd.tslib.Timestamp('2016-09-15 00:00:00')] = [0,0,0,1,0,0,0,0,1,0, 0,1,0,1,0,0,1,0,0,0]
    feature_date.loc[pd.tslib.Timestamp('2016-09-16 00:00:00')] = [0,0,0,0,1,0,0,0,1,0, 0,0,0,1,0,0,0,0,0,0]
    feature_date.loc[pd.tslib.Timestamp('2016-09-17 00:00:00')] = [0,0,0,0,0,1,0,0,1,0, 0,0,1,1,0,0,0,0,0,0]
    feature_date.loc[pd.tslib.Timestamp('2016-09-18 00:00:00')] = [0,0,0,0,0,0,1,0,1,0, 0,0,0,0,1,1,0,0,0,0]
                     
    feature_date.loc[pd.tslib.Timestamp('2016-09-30 00:00:00')] = [0,0,0,0,1,0,0,0,0,1, 1,0,0,0,0,0,0,0,0,0]  # 十一
    feature_date.loc[pd.tslib.Timestamp('2016-10-01 00:00:00')] = [0,0,0,0,0,1,0,1,0,0, 0,1,0,1,0,0,1,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-02 00:00:00')] = [0,0,0,0,0,0,1,1,0,0, 0,0,0,1,0,0,0,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-03 00:00:00')] = [1,0,0,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-04 00:00:00')] = [0,1,0,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-05 00:00:00')] = [0,0,1,0,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-06 00:00:00')] = [0,0,0,1,0,0,0,1,0,0, 0,0,0,1,0,0,0,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-07 00:00:00')] = [0,0,0,0,1,0,0,1,0,0, 0,0,1,1,0,0,0,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-08 00:00:00')] = [0,0,0,0,0,1,0,1,0,0, 0,0,0,0,1,1,0,0,1,0]
    feature_date.loc[pd.tslib.Timestamp('2016-10-09 00:00:00')] = [0,0,0,0,0,0,1,1,0,0, 0,0,0,0,0,1,0,0,1,0]
    
    feature_date.loc[pd.tslib.Timestamp('2016-11-11 00:00:00')]['festival'] = 1  # 11
    feature_date.to_csv(os.path.join(dataRoot_cache,'feature_date.csv'))

    
def city_level(city):
    city_1 = ['北京','上海','广州','深圳','天津','杭州','西安','成都','重庆','武汉','南京','济南','青岛','大连','宁波','厦门','苏州','无锡''哈尔滨','沈阳',
              '长春','长沙','福州','郑州','石家庄','佛山','东莞','烟台','太原','合肥','南昌','南宁','昆明','温州','淄博','唐山']
    if city in city_1:
        return 1
    else: 
        return 0
        
def shop_user_feature(data_info,data_view,data_pay):  
    data_pay = data_pay[data_pay['time_stamp']>pd.tslib.Timestamp('2016-02-29 00:00:00')]
    data_view = data_view[data_view['time_stamp']>pd.tslib.Timestamp('2016-02-29 00:00:00')]
    rate_view_pay= []
    for shop in data_info.itertuples(index=False):
        temp = []
        shop_pay = data_pay[data_pay['shop_id']==shop[0]]   # 单个商家账单
        shop_view = data_view[data_view['shop_id']==shop[0]]   # 单个商家浏览行为
        user_id_view = list(set(list(shop_view['user_id'])))
        user_id_pay = list(set(list(shop_pay['user_id'])))
        pay_user = shop_pay.groupby('user_id').count()['shop_id']
        pay_two = pay_user[pay_user>1]
        pay_two_num = pay_two.shape[0]
        count = 1
        for id in user_id_view:            
            if id in user_id_pay:
                count += 1
        temp.append(shop_view.shape[0]/float(count))  # 浏览人数的比例 / 浏览并购买的人数
        temp.append(shop_pay.shape[0]/float(count))  # 购买人数的比例 / 浏览并购买的人数
        temp.append(float(pay_two_num)/shop_pay.shape[0])  # 二次购买人数占总购买人数比例
        rate_view_pay.append(temp)
                
    rate_view_pay = pd.DataFrame(rate_view_pay,index = data_info['shop_id'],columns=['rate_vp_v','rate_vp_p','rate_pay_two'])
    rate_view_pay.to_csv(os.path.join(dataRoot_cache,'shop_user_feature.csv'))
    
    
def tsfresh_feature():
    data_flow = pd.read_csv(os.path.join(dataRoot_cache,'consumer_flow_null.csv'),index_col=0)
    data_flow = data_flow.iloc[244:]
    data_flow = data_flow.fillna(data_flow.describe().loc['mean'])
    data_flow.reset_index(inplace=True,drop=True)
    data_flow = data_flow.stack(dropna=False)
    data_flow = pd.DataFrame(data_flow)
    data_flow = data_flow.swaplevel()
    data_flow.reset_index(inplace=True)
    data_flow.columns=['shop_id','time','y_flow']
    features_filtered_direct = extract_features(data_flow,column_id='shop_id', column_sort='time')
    
    
def clcDistance(vec1,vec2):
    index = range(3,57)
    index.insert(0,0)
    vec1 = vec1[index]
    vec2 = vec2[index]
    dis = np.linalg.norm(vec1 - vec2)
    return dis
    
def nearstPolation(feature_shop_basic):
#    feature_shop_basic = pd.read_csv(os.path.join(dataRoot_cache,'feature_shop_basic.csv'),index_col=0)
    null_index = list(feature_shop_basic[feature_shop_basic['comment_cnt'].isnull()].index)
    notnull_index = list(feature_shop_basic[feature_shop_basic['comment_cnt'].notnull()].index)
    for i in null_index:
        dis = []
        for j in notnull_index:
            dis.append( clcDistance(feature_shop_basic.loc[i],feature_shop_basic.loc[j]) )
        dis = pd.DataFrame(dis,columns=['distance'],index = notnull_index)
        dis = dis.sort_values('distance')
        data_nearest = feature_shop_basic.loc[dis.index[:5]]
        des = data_nearest.describe().loc['mean']
        feature_shop_basic.loc[i] = feature_shop_basic.loc[i].fillna(des)
    feature_shop_basic.to_csv(os.path.join(dataRoot_cache,'feature_shop_basic.csv'))
            

