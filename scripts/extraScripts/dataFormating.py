#!/usr/bin/env python
# coding: utf-8

from sktime.transformers.compose import ColumnConcatenator
from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.classifiers.dictionary_based.boss import BOSSEnsemble
from sktime.classifiers.compose import ColumnEnsembleClassifier
from sktime.classifiers.shapelet_based import ShapeletTransformClassifier
from sktime.datasets import load_basic_motions
from sktime.pipeline import Pipeline

from sktime.classifiers.distance_based import ProximityForest
from sktime.classifiers.distance_based import KNeighborsTimeSeriesClassifier

import numpy as np
import pandas as pd


#cleaning up whatever happened in loading data 
#feed in the data 
from sktime.utils.load_data import from_long_to_nested
import time 
import sys 
import json

#setting constants 
classifier={
    'TSF_CLF':0,
    'KNN_CLF':1,
    'PF_CLF':2,
}


#concatenates ....
#make this assumption cleare 
#write better columns 
#error messages
#define constants when refactoring(readability)

#extracting the data from the csv as a dataframe and reformatting it 

def reformatData(target, file_name):
    print("reformatting the data...")
    raw_df= pd.read_csv(file_name)
    
    #collapses the time cols 
    long_table_df= raw_df.melt(id_vars=["event", "name","start time", "end time","channel"], 
            var_name="anindex", 
            value_name="value")

    sorted_long_table_df=long_table_df.sort_values(by=['event','name','start time','channel'], axis=0)

    unique_dim_ids = sorted_long_table_df.iloc[:, 4].unique()

    for i in range(len(unique_dim_ids)):
        my_channel=unique_dim_ids[i]
        sorted_long_table_df['channel']=sorted_long_table_df['channel'].replace({my_channel:i})
    unique_start_time = sorted_long_table_df.iloc[:, 2].unique()

    for i in range(len(unique_start_time)):
        my_time=unique_start_time[i]
        sorted_long_table_df['start time']=sorted_long_table_df['start time'].replace({my_time:i})

    
    sorted_long_table_df_stripped=sorted_long_table_df.drop(columns=['event','name','end time'])

    sorted_long_table_df_stripped.head()
    df_nested = from_long_to_nested(sorted_long_table_df_stripped)


    target='event'
    new_unique_start_time=sorted_long_table_df.iloc[:, 2].unique()
    labels=[]
    for e in new_unique_start_time:
        x=sorted_long_table_df.loc[sorted_long_table_df['start time']==e,[target]].iloc[0][0]
        labels.append(x)

    np_labels= np.asarray(labels, dtype=np.str)
    
    return df_nested, np_labels 

def splitTestTrain(X, y, percent_train):
    msk = np.random.rand(len(X)) < percent_train
    ytrain=y[msk]
    ytest=y[~msk]
    Xtrain=X[msk]
    Xtest=X[~msk]
    
    return Xtrain, ytrain, Xtest, ytest

def concatenateMethod(Classifier, x_train, y_train, x_test, y_test):
    steps = [
    ('concatenate', ColumnConcatenator()),
    ('classify', Classifier(n_estimators=10))]
    clf = Pipeline(steps)
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)

def concatenateMethodTake2(classifier, X, y, percent_train):
    print("classification...")
    concatenation_instance=ColumnConcatenator()
    X_concatenated=concatenation_instance.fit(X).transform(X)
    Xtrain, ytrain, Xtest, ytest= splitTestTrain(X_concatenated,y,percent_train)
    
    if(classifier=='TSF_CLF'):
        clf= TimeSeriesForestClassifier(n_estimators=10)
        clf.fit(Xtrain, ytrain)
        return clf.score(Xtest,ytest)
    elif (classifier=='KNN_CLF'):
        knn = KNeighborsTimeSeriesClassifier(metric='dtw')
        knn.fit(Xtrain, ytrain)
        return knn.score(Xtest, ytest)
    elif (classifier == 'PF_CLF'):    
        pf = ProximityForest(n_trees=10)
        pf.fit(Xtrain, ytrain)
        return pf.score(Xtest, ytest)
    else: 
        return 0
        


def main():
    script = sys.argv[0]
    json_file_name = sys.argv[1]
    print(json_file_name) 
    with open(json_file_name) as f:
        data = json.load(f)
    print(data)
    target=data['target']
    file_name=data['fileName']
    percent_train=data['percentTrain']
    should_profile=data['benchmark']
    X, y=reformatData(target,file_name)

    for job in data['jobs']:
        print("JOB:")
        print(job)
        start_time=time.time()
        acc= concatenateMethodTake2(job['classifier'], X[["dim_0", "dim_1"]], y, percent_train)
        print(acc)
        print("Total Time")
        end_time= time.time() -start_time
        print(end_time)



main()





