# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 22:24:24 2017

@author: muller
"""

import sys, codecs,copy
import numpy as np
import pandas as pd
from bankCategorizer import bankCategorizer

LABEL, DATE, EUR, TEXT = range(0,4)

def main(): 
    if len(sys.argv)==2:            
      filename=sys.argv[1]  
      
    easyb = bankCategorizer() 
    easyb.loadForest()                            
    datatest,konto=easyb.import_easybank_train(filename)
    result, prob = easyb.classifyData(copy.deepcopy(datatest))
                   

    #create test report 
    output = pd.DataFrame( data={"groundtruth":datatest[LABEL], "result":result, "score":prob} )    
    cnt_correct=0
    for i in range(output.size/3):
        if output['groundtruth'][i] == output['result'][i]:
            cnt_correct+=1
    print str(float(cnt_correct)/(output.size/3.0)*100) + " % korrekt"
    # Use pandas to write the comma-separated output file
    output.to_csv( "test_report.csv", index=False, quoting=3 ) 
    
    # create categorized umsatzliste
    f=codecs.open('umsatzliste.txt', 'w', encoding="ISO-8859-1")
    f.write("Result\tKonto\t\Text\tDate1\tDate2\t\Sum\t\Cur\tScore\n")   
    for rec in range(len(prob)):
      if prob[rec]<0.5:   
         result[rec]=u'' 
      buf = "%s\tkonto\t%s\t%s\t%s\t%f\tEuro\t%f\n" % (result[rec], datatest[TEXT][rec], datatest[1]['Timestamp'][rec].isoformat(), datatest[1]['Timestamp'][rec].isoformat(), datatest[EUR][rec],prob[rec])
      f.write(buf)    
    f.close()
    
    #show sum for each category
    data_analyze = pd.DataFrame( data={"groundtruth":datatest[LABEL], "result":result, "score":prob, "summe":datatest[EUR]} )
    data_analyze=  pd.concat([data_analyze, datatest[DATE]], axis=1)
    unique_cat = list(set(data_analyze['result'].tolist()))
    f=codecs.open('umsatz_analyse.txt', 'w', encoding="ISO-8859-1")    
    f.write("Category\tabs Sum in Period[EUR]\tPeriod [months]\tavg Sum per Month[EUR]\t# of records in period\tFrequency [days]\tavg Sum per item[EUR]\n")         
    for cat in unique_cat:
    #if 1:
        #cat='Supermarkt'
        print cat
        sum_cat = data_analyze.loc[data_analyze['result']==cat, 'summe'].sum()
        print "Summe EURO " + str(sum_cat) 
        pd_tmp_cat = data_analyze.loc[data_analyze['result']==cat]        
        #would it be better to use months from whole data_analyze?         
        num_months = getNumberMonths(pd_tmp_cat)
        avg_freq_days = getAVGDayFrequency(pd_tmp_cat)
        num_recs_in_cat = pd_tmp_cat.shape[0]
        avg_sum_per_rec = sum_cat/num_recs_in_cat
        avg_sum_per_month = sum_cat / num_months
                        
        buf = "%s\t%f\t%f\t%f\t%f\t%f\t%f\t\n" % (cat, sum_cat, num_months, avg_sum_per_month, num_recs_in_cat, avg_freq_days, avg_sum_per_rec)
        f.write(buf)    
        
                
        pd_cat = pd_tmp_cat.copy()
        pd_cat = pd_cat.sort_values("Timestamp")
        unique_months = list(set(pd_cat['Month/Year'].tolist()))
        for month in unique_months:
            print month
            sum_cat_month = pd_cat.loc[data_analyze['Month/Year']==month, 'summe'].sum()
            print "Summe EURO " + str(sum_cat_month) 
            pd_tmp_cat_m = pd_cat.loc[data_analyze['Month/Year']==month]                    
            avg_freq_days_m = getAVGDayFrequency(pd_tmp_cat_m)
            num_recs_in_cat_m = pd_tmp_cat_m.shape[0]
            avg_sum_per_rec_m = sum_cat/num_recs_in_cat            
    f.close()       
        
def getNumberMonths(panda_df):    
    days_in_month_constant = 30.436875
    timedelta=max(panda_df['Timestamp'])-min(panda_df['Timestamp'])
    if max(panda_df['Timestamp'])==min(panda_df['Timestamp']):
        period=1
    else:
        period = timedelta.days/days_in_month_constant    
    return period
    
def getAVGDayFrequency(panda_df):    
    df_sort=panda_df.sort_values("Timestamp")
    df_sort['diff'] = (df_sort['Timestamp']-df_sort['Timestamp'].shift()).fillna(0)
    avg_frequency=float(df_sort['diff'].sum().days)/df_sort['diff'].size
    return avg_frequency
    
    
main()    
    