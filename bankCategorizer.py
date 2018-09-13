# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 22:52:55 2017

@author: muller
"""
import numpy as np
import codecs
from datetime import datetime
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import random
import pickle

LABEL, DATE, EUR, TEXT = range(0,4)

class bankCategorizer():

    def trainData(self, importData):
        importData=self.clean_buchtext(importData)
        train_data_features, vectorizer = self.createBagOfWords(importData[TEXT])
        train_data_features = self.add_custom_features(train_data_features, importData[EUR])
        forest = self.trainRandomForest(train_data_features, importData[LABEL])
        return forest, vectorizer, train_data_features

    def classifyData(self, importedData):
        importedData=self.clean_buchtext(importedData)
        test_data_features = self.vectorizer.transform(importedData[TEXT]).toarray()
        test_data_features = self.add_custom_features(test_data_features, importedData[EUR])
        result,prob=self.testRandomForest(self.forest,test_data_features)
        prob=np.amax(prob,1)
        return result, prob

    def saveForest(self, forest, vectorizer):
        # ********* SAVE RANDOM FOREST **************
        filename_vec = 'vectorizer.sav'
        pickle.dump(vectorizer, open(filename_vec, 'wb'))
        filename = 'randForest.sav'
        pickle.dump(forest, open(filename, 'wb'))
        print "\nClassifier saved as:\n" + 'vectorizer.sav\n' + 'randForest.sav'
        #*********************************************
        return 0

    def loadForest(self):
        # ********* SAVE RANDOM FOREST **************
        filename_vec = 'vectorizer.sav'
        filename = 'randForest.sav'

        vectorizer = pickle.load(open(filename_vec, 'rb'))
        forest = pickle.load(open(filename, 'rb'))
        #*********************************************
        self.forest=forest
        self.vectorizer=vectorizer

        return forest, vectorizer

    def add_custom_features(self, features, summe):
        features[0].size
        tmp = np.zeros((features.shape[0],features.shape[1]+2))
        tmp[:,:-2] = features
        for i in range(features.shape[0]):
            if summe[i]<0:
                tmp[i][-1]=1
            if summe[i]<200:
                tmp[i][-2]=1
        return tmp


    def testRandomForest(self, forest, features):
        # Use the random forest to make sentiment label predictions
        result = forest.predict(features)
        prob = forest.predict_proba(features)
        return result,prob


    def trainRandomForest(self, train_data_features, labels):
        print "Training the random forest..."
        from sklearn.ensemble import RandomForestClassifier

        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier(n_estimators = 100)

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        forest = forest.fit( train_data_features, labels )
        return forest


    def createBagOfWords(self, words):
        print "Creating the bag of words..."
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer(analyzer = "word",   \
                                     tokenizer = None,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features = 1000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        train_data_features = vectorizer.fit_transform(words)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        train_data_features = train_data_features.toarray()

        # Take a look at the words in the vocabulary
        vocab = vectorizer.get_feature_names()

        # Sum up the counts of each vocabulary word
        dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set

        return train_data_features, vectorizer



    #clean up buchungstext by replacing or deleting data which is no feature
    def clean_buchtext(self, data):
        for d in range(len(data[TEXT])):
            data[TEXT][d]=re.sub('Kapsch[A-Za-z0-9]+','KAPSCH',data[TEXT][d])
            data[TEXT][d]=re.sub('S6EE[0-9][0-9][0-9][0-9]','S6EEINLAND',data[TEXT][d]) #replace bankomat ID
            data[TEXT][d]=re.sub('\d+\.\d+\.20[0-9][0-9]','',data[TEXT][d]) #delete dates
            data[TEXT][d]=re.sub('[0-9][0-9]\.[0-9][0-9]\..+UM.+[0-9][0-9]\.[0-9][0-9]','zeitdatum',data[TEXT][d])
            #data[TEXT][d]=re.sub('\d+',' ',data[TEXT][d]) #delete all numbers!
            data[TEXT][d]=re.sub('[0-9]','n',data[TEXT][d]) #delete all numbers!
            data[TEXT][d]=re.sub('[A-Z]','a',data[TEXT][d]) #delete all numbers!
            data[TEXT][d]= " ".join(data[TEXT][d].split())  # delete all whitespaces  tabs etc... and replace with whitespace
        return data



    def import_paypal_train(self, filename, remove_empty_labels=False):
        print 'Importing ' + filename + ' ...\n'
        LABEL, DAT, TIME, ZZONE, NAME, TYP, STATUS, WAEHRUNG, BETRAG, BELEGNUMMER, GUTHABEN, EMPTY = range(0,12)  #data columns in CSV FILE
        f=codecs.open(filename,'r', encoding="ISO-8859-1")
        data=f.read()
        f.close()

        data=data.replace('"','')
        data=data.splitlines()
        del data[0]
        for rec in range(len(data)):
            data[rec]=data[rec].split('\t')
        if remove_empty_labels==True:
            tmp_data=[]
            for rec in range(len(data)):
                 if len(data[rec][0])>0:
                    tmp_data.append(data[rec])
            data=list(tmp_data)
        random.shuffle(data)

        columns=zip(*data)

        #create new data array
        data=[]

        columns[BETRAG]=list(columns[BETRAG])
        columns[DAT]=list(columns[DAT])
        #format SUM and DATE
        for s in range(len(columns[BETRAG])):
                columns[BETRAG][s]=columns[BETRAG][s].replace('+','')
                if ''.join(columns[BETRAG]).find(',') >-1:
                    columns[BETRAG][s]=columns[BETRAG][s].replace('.','').replace(',','.')
                #columns[DAT1][s]=datetime.strptime(columns[DAT1][s], '%d.%m.%Y')
                date_tmp=datetime.strptime(columns[DAT][s], '%d.%m.%Y')
                columns[DAT][s] = np.datetime64("%d-%02d-%02d" % (date_tmp.year,date_tmp.month,date_tmp.day))

        #create panda data frame
        df_tmp = pd.DataFrame({'Timestamp': pd.DatetimeIndex(columns[DAT])})
        df_tmp['Month/Year'] = df_tmp['Timestamp'].apply(lambda x: "%d/%d" % (x.month, x.year))

        data.append(list(columns[LABEL]))
        data.append(df_tmp)
        data.append(np.array(map(float, columns[BETRAG])))
        data.append(list(columns[NAME]))

        #Print Unique Categories
        unique_categories = list(set(data[LABEL]))
        print str(len(unique_categories)) + ' Unique Categories: '
        print '***************'
        for cat in unique_categories:
            print cat
        print '***************\n'
        return data, 'paypal'


    def import_easybank_train(self, filename, remove_empty_labels=False):
        print 'Importing ' + filename + ' ...\n'
        LABEL, KTO, TXT, DAT1,DAT2, SUM, CUR = range(0,7)  #data columns in CSV FILE
        f=codecs.open(filename,'r', encoding="ISO-8859-1")
        data=f.read()
        f.close()

        data=data.splitlines()
        del data[0]
        for rec in range(len(data)):
            data[rec]=data[rec].split('\t')
        if remove_empty_labels==True:
            tmp_data=[]
            for rec in range(len(data)):
                 if len(data[rec][0])>0:
                    tmp_data.append(data[rec])
            data=list(tmp_data)
        random.shuffle(data)

        columns=zip(*data)

        #create new data array
        data=[]

        columns[SUM]=list(columns[SUM])
        columns[DAT1]=list(columns[DAT1])
        #format SUM and DATE
        for s in range(len(columns[SUM])):
                columns[SUM][s]=columns[SUM][s].replace('+','')
                if ''.join(columns[SUM]).find(',') >-1:
                    columns[SUM][s]=columns[SUM][s].replace('.','').replace(',','.')
                #columns[DAT1][s]=datetime.strptime(columns[DAT1][s], '%d.%m.%Y')
                date_tmp=datetime.strptime(columns[DAT1][s], '%d.%m.%Y')
                columns[DAT1][s] = np.datetime64("%d-%02d-%02d" % (date_tmp.year,date_tmp.month,date_tmp.day))

        #create panda data frame
        df_tmp = pd.DataFrame({'Timestamp': pd.DatetimeIndex(columns[DAT1])})
        df_tmp['Month/Year'] = df_tmp['Timestamp'].apply(lambda x: "%d/%d" % (x.month, x.year))

        data.append(list(columns[LABEL]))
        data.append(df_tmp)
        data.append(np.array(map(float, columns[SUM])))
        data.append(list(columns[TXT]))

        #Print Unique Categories
        unique_categories = list(set(data[LABEL]))
        print str(len(unique_categories)) + ' Unique Categories: '
        print '***************'
        for cat in unique_categories:
            print cat
        print '***************\n'
        return data, columns[KTO][0]

    def getForestPerformance(self, labels, groundtruth, probability):
        # create test report
        output = pd.DataFrame(data={"groundtruth": labels,
                                    "result": groundtruth,
                                    "score": probability})
        cnt_correct = 0
        for i in range(output.size/3):
            if output['groundtruth'][i] == output['result'][i]:
                cnt_correct += 1
        print str(float(cnt_correct)/(output.size/3.0)*100) + " % korrekt"
        return float(cnt_correct)/(output.size/3.0)*100