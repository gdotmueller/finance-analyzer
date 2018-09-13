# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:52:13 2017

@author: mul
"""
import sys
from sklearn.model_selection import cross_val_score
from bankCategorizer import bankCategorizer

LABEL, DATE, EUR, TEXT = range(0, 4)


def main():
    if len(sys.argv) == 2:
        filename = sys.argv[1]

    easyb = bankCategorizer()
    # import data and filter records with empty GT labels
    data, konto = easyb.import_easybank_train(filename, True)
    forest, vectorizer, feature_vectors = easyb.trainData(data)

    print "\n\nTop 10 / Features sorted by their score:\n----------------------------"
    feature_names = sorted(zip(map(lambda x: round(x, 4),
                                   forest.feature_importances_),
                                   vectorizer.get_feature_names()),
                                   reverse=True)
    for feat in range(0, 10):
        print feature_names[feat][1] + " / " + str(feature_names[feat][0])

    print "\nPerformance of Random Forest: "
    crossvalid_perf = cross_val_score(forest, feature_vectors, data[LABEL])
    for c in crossvalid_perf:
        print str(c*100) + ' % correct'
    easyb.saveForest(forest, vectorizer)
    print "Finished!"

main()
