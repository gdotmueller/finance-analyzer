# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 22:24:24 2017

@author: muller
"""
import sys
import codecs
import copy
import numpy as np
import pandas as pd
from bankCategorizer import bankCategorizer

LABEL, DATE, EUR, TEXT = range(0, 4)

konten = {'AT0814200xxxx': 'easyKreditkarte', "AT8714200xxxx": "easyGiro", "paypal": "Paypal"}


def main():
    files = []
    for arg in range(1, len(sys.argv)):
        files.append(sys.argv[arg])

    easyb = bankCategorizer()
    easyb.loadForest()
    finance_data = []
    for fin in files:
        finance_data.append(writeAnalyzeLog(easyb, fin))
        print konten[finance_data[-1][1]]
    x = 0


def writeAnalyzeLog(easyb, inputFileName):
    if inputFileName.lower().find('paypal') > -1:
        datatest, konto = easyb.import_paypal_train(inputFileName)
    else:
        datatest, konto = easyb.import_easybank_train(inputFileName)
    result, prob = easyb.classifyData(copy.deepcopy(datatest))
    perfForest = easyb.getForestPerformance(datatest[LABEL], result, prob)

    # build data Frame for analysis
    data_analyze = pd.DataFrame(data={"groundtruth": datatest[LABEL],
                                      "result": result,
                                      "score": prob,
                                      "summe": datatest[EUR]})
    # append Date to data Frame
    data_analyze = pd.concat([data_analyze, datatest[DATE]], axis=1)
    unique_cat = list(set(data_analyze['result'].tolist()))

    cat_data = pd.DataFrame(columns=("category",
                                     "numRecs",
                                     "sum_euro",
                                     "avgDaysFreq",
                                     "avgSumRec",
                                     "avgSumMonth",
                                     "period"))
    cnt = 0
    for cat in unique_cat:
        sum_cat = data_analyze.loc[data_analyze['result'] == cat, 'summe'].sum()
        pd_tmp_cat = data_analyze.loc[data_analyze['result'] == cat]

        newLine = pd.DataFrame(data={"category": cat,
                                     "numRecs": pd_tmp_cat.shape[0],
                                     "sum_euro": sum_cat,
                                     "avgDaysFreq": getAVGDayFrequency(pd_tmp_cat),
                                     "avgSumRec": sum_cat / pd_tmp_cat.shape[0],
                                     "avgSumMonth": sum_cat / getNumberMonths(pd_tmp_cat),
                                     "period": getNumberMonths(pd_tmp_cat)}, index=[cnt])
        cat_data = cat_data.append(newLine)
        cnt += 1


        # monthly analyzes
    #        pd_cat = pd_tmp_cat.copy()
    #        pd_cat = pd_cat.sort_values("Timestamp")
    #        unique_months = list(set(pd_cat['Month/Year'].tolist()))
    #        for month in unique_months:
    #            print month
    #            sum_cat_month = pd_cat.loc[data_analyze['Month/Year']==month, 'summe'].sum()
    #            print "Summe EURO " + str(sum_cat_month)
    #            pd_tmp_cat_m = pd_cat.loc[data_analyze['Month/Year']==month]
    #            avg_freq_days_m = getAVGDayFrequency(pd_tmp_cat_m)
    #            num_recs_in_cat_m = pd_tmp_cat_m.shape[0]
    #            avg_sum_per_rec_m = sum_cat/num_recs_in_cat

    return cat_data, konto


def writeTableOverview():
    f = codecs.open('umsatz_analyse.txt', 'w', encoding="ISO-8859-1")
    f.write(
        "Category\tabs Sum in Period[EUR]\tPeriod [months]\tavg Sum per Month[EUR]\t# of records in period\tFrequency [days]\tavg Sum per item[EUR]\n")


def getNumberMonths(panda_df):
    """returns number of consecutive months from earliest to latest date
    (e.g. earliest date = Jan 2016, latest date = May 2017,
    but no dates from Sep 2016 -> returned number of months is
    ALWAYS from earliest date to latest date - in this case 17 months)
    Output is a float number because number of days is divided
    by an average month constant.

    Parameters
    ----------
    panda_df : array_like
        pandas data frame with at least one column called 'Timestamp'
        which consists of pandas Timestamps

    Returns
    -------
    period : float
        returns number of months as a float number
    """
    days_in_month_constant = 30.436875
    timedelta = max(panda_df['Timestamp']) - min(panda_df['Timestamp'])
    if max(panda_df['Timestamp']) == min(panda_df['Timestamp']):
        period = 1
    else:
        period = timedelta.days / days_in_month_constant
    return period


def getAVGDayFrequency(panda_df):
    """returns the average frequency in days from a list of items with timestamps

    Parameters
    ----------
    panda_df : array_like
        pandas data frame with at least one column called 'Timestamp'
        which consists of pandas Timestamps

    Returns
    -------
    period : float
        returns number of days
    """
    df_sort = panda_df.sort_values("Timestamp")
    df_sort['diff'] = (df_sort['Timestamp'] -
                       df_sort['Timestamp'].shift()).fillna(0)
    avg_frequency = float(df_sort['diff'].sum().days) / df_sort['diff'].size
    return avg_frequency


main()
