#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

import pandas as pd
import time

def rate(s,val):
    return np.sum(s==val)/len(s)

    
def q1(df):
    mode = df['INCIDENT_TYPE_DESC'].mode()[0]
    slct = df['INCIDENT_TYPE_DESC']==mode
    print("question 1:", np.sum(slct)/len(df))


def q2(df):
    fire = df[df['INCIDENT_TYPE_DESC']=='111 - Building fire']['UNITS_ONSCENE'].mean()
    smoke = df[df['INCIDENT_TYPE_DESC']=='651 - Smoke scare, odor of smoke']['UNITS_ONSCENE'].mean()
    print("question 2: fire/smoke:", fire/smoke )


    
def q3(df):
    manhattan = rate(df[df['BOROUGH_DESC']=='1 - Manhattan']['INCIDENT_TYPE_DESC'],
                     '710 - Malicious, mischievous false call, other')
    staten    = rate(df[df['BOROUGH_DESC']=='3 - Staten Island']['INCIDENT_TYPE_DESC'],
                     '710 - Malicious, mischievous false call, other')
    # print(staten, manhattan, staten/manhattan)
    print("question 3: false calls in staten vs manhattan: ", staten/manhattan)

def q4(df):
    df2 = df[df['INCIDENT_TYPE_DESC']=='111 - Building fire']
    quart3rd_111 = (df2['ARRIVAL_DATE_TIME']-df2['INCIDENT_DATE_TIME']).quantile(0.75)
    print("question 4: 3rd quart response time to 111: ", quart3rd_111)

def q5(df):
    df['hour'] = df['INCIDENT_DATE_TIME'].dt.hour
    gdfs = df.groupby(by=['hour','INCIDENT_TYPE_DESC']).size()
    fract_cooking = gdfs[:,'113 - Cooking fire, confined to container']/gdfs.sum(level=[0])
    a = fract_cooking.values.argmax()
    print("question5: when are people cooking: hour={}, fraction_cooking_fires={:.3f}".format(a,fract_cooking[a]))

    
def q6(df):
    df2 = df[df['INCIDENT_TYPE_DESC']=='111 - Building fire']
    df2._is_copy = None
    dfc = df2.groupby('ZIP_CODE').size().to_frame(name='fires')
    # print(dfc)
    census = pd.read_csv('2010+Census+Population+By+Zipcode+(ZCTA).csv', dtype={'Zip Code ZCTA':str})
    fires_zip = census.merge(dfc, how='right', left_on='Zip Code ZCTA', right_on='ZIP_CODE')
    fires_zip.dropna(inplace=True)
    # print(fires_zip)
    from scipy.stats import linregress
    slope,inter, r, p, err = linregress(fires_zip['2010 Census Population'],fires_zip['fires'])
    # print(slope, inter, r, p, err)
    print("question 6: r^2 for population vs fires:", r**2)
    # plt.figure()
    # plt.plot(fires_zip['2010 Census Population'], fires_zip['fires'], 'o')
    # plt.xscale('log')
    # plt.show()

def q7(df):
    df2 = df[pd.notnull(df['CO_DETECTOR_PRESENT_DESC'])]
    df2._is_copy = None
    df2['incident_duration']= df2['LAST_UNIT_CLEARED_DATE_TIME']-df2['INCIDENT_DATE_TIME']
    # print(df2[df2['incident_duration']<'00:{}:00'.format(20)])
    bins = np.linspace(20,70,6, dtype=int)
    bins_cen = (bins[1:]+bins[:-1])/2.0
    fract_co = np.zeros(len(bins_cen))
    for i in range(0,len(bins_cen)):
        slct1  = df2['incident_duration']>'00:{:d}:00'.format(bins[i])
        slct2  =  df2['incident_duration']<'00:{:d}:00'.format(bins[i+1])
        slct = slct1 & slct2
        # print(np.sum(df2[slct]['CO_DETECTOR_PRESENT_DESC']=='Yes'), np.sum(slct))
        fract_co[i] = np.sum(df2[slct]['CO_DETECTOR_PRESENT_DESC']=='Yes')/np.sum(slct)
    clean = np.isfinite(fract_co)
    bins_cen = bins_cen[clean]
    fract_co = fract_co[clean]
    # print(bins_cen)
    # print(fract_co)
    from scipy.stats import linregress
    slope, inter, _, _, _ =linregress(bins_cen, fract_co)
    # print(slope, inter)
    print("question 7: predicted fract co dec at 39min duration: ", slope*39 + inter)

def q8(df):
    df2 = df[pd.notnull(df['CO_DETECTOR_PRESENT_DESC'])]
    df2._is_copy = None
    df2['incident_duration']= df2['LAST_UNIT_CLEARED_DATE_TIME']-df2['INCIDENT_DATE_TIME']
    df2['longer_60'] = df2['incident_duration']>'00:60:00'
    dfs = df2.groupby(by=['CO_DETECTOR_PRESENT_DESC', 'longer_60']).size()
    df3 = dfs.to_frame()
    # print(dfs)
    #print(dfs.sum(level=[0]))
    # print('\n')
    #print(dfs.sum(level=[1]))
    longer_60 =dfs.sum(level=[1])
    co_pres = dfs.sum(level=[0])
    expected_60 =longer_60/longer_60.sum()
    Xsqr = 0.0

    for co_pres_i in ['Yes', 'No']:
        for longer_60_i in [True, False]:
            a = expected_60[longer_60_i]
            b = co_pres[co_pres_i]
            expected = a*b
            obs      = dfs[co_pres_i][longer_60_i]
            Xsqr += (expected - obs)**2/expected
            # print()
            # print(a)
            # print(b)
            # print(co_pres_i, longer_60_i)
            # print(expected, obs)
            # print(Xsqr)
            # print()
    print("question 8: X^2 for longer incident if no co dect: ", Xsqr)
    
    
def ny_fire():
    t1 = time.time()
    usecols=['INCIDENT_TYPE_DESC', 'INCIDENT_DATE_TIME',
             'ARRIVAL_DATE_TIME', 'TOTAL_INCIDENT_DURATION',
             'ZIP_CODE', 'BOROUGH_DESC',
             'CO_DETECTOR_PRESENT_DESC','UNITS_ONSCENE',
             'LAST_UNIT_CLEARED_DATE_TIME' ]
    col_dtype={'INCIDENT_TYPE_DESC':str,
               'INCIDENT_DATE_TIME':str,
               'ARRIVAL_DATE_TIME':str,
               'TOTAL_INCIDENT_DURATION':str,
               'ZIP_CODE':str,
               'BOROUGH_DESC':str,
               'CO_DETECTOR_PRESENT_DESC':str,
               'UNITS_ONSCENE':float,
               'LAST_UNIT_CLEARED_DATE_TIME':str }

    # dtypes = {'ARRIVAL_DATE_TIME': pd.}
    df = pd.read_csv('Incidents_Responded_to_by_Fire_Companies.csv', usecols=usecols, dtype=col_dtype)
    df['ZIP_CODE'] = df['ZIP_CODE'].str[:8]
    df['LAST_UNIT_CLEARED_DATE_TIME'] = pd.to_datetime(df['LAST_UNIT_CLEARED_DATE_TIME'])
    df['INCIDENT_DATE_TIME'] = pd.to_datetime(df['INCIDENT_DATE_TIME'])
    df['ARRIVAL_DATE_TIME'] = pd.to_datetime(df['ARRIVAL_DATE_TIME'])
    print(df['ZIP_CODE'])
    print(df.keys())
    q1(df)
    q2(df)
    q3(df)
    q4(df)
    q5(df)
    q6(df)
    q7(df)
    q8(df)
    print("time: ", time.time()-t1)


if __name__ == "__main__":
    ny_fire()




