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
from datetime import date  
from datetime import datetime
#setting constants 
classifier={
    'TSF_CLF':0,
    'KNN_CLF':1,
    'PF_CLF':2,
}
TO_LOG= False 

#concatenates ....
#make this assumption cleare 
#write better columns 
#error messages
#define constants when refactoring(readability)

#extracting the data from the csv as a dataframe and reformatting it 

def reformatData(target, file_name):
    print("reformatting the data...")
    raw_df= pd.read_csv(file_name)
    
    #collapses the time cols into one single time column to match the rest of the columns 
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


    new_unique_start_time=sorted_long_table_df.iloc[:, 2].unique()
    labels=[]
    for e in new_unique_start_time:
        x=sorted_long_table_df.loc[sorted_long_table_df['start time']==e,[target]].iloc[0][0]
        labels.append(x)

    np_labels= np.asarray(labels, dtype=np.str)
    
    return df_nested, np_labels 
# #extracting the data from the csv as a dataframe and reformatting it for sktime compatibility 
# def reformatData(target, file_name):
    
#     print("reformatting the data...")
#     raw_df= pd.read_csv(file_name)
    
#     #collapses the time cols into one single time column to match the rest of the columns 
#     long_table_df= raw_df.melt(id_vars=["event", "name","start time", "end time","channel"], 
#             var_name="anindex", 
#             value_name="value")

#     sorted_long_table_df=long_table_df.sort_values(by=['event','name','start time','channel'], axis=0)

#     #start time qualifies as a unique start time
#     #maybe will switch this out later 
#     unique_dim_ids = sorted_long_table_df.iloc[:, 4].unique()

#     #replacing channel named to numeric values (need to do this doem the from_long_to_nested function)
#     for i in range(len(unique_dim_ids)):
#         my_channel=unique_dim_ids[i]
#         sorted_long_table_df['channel']=sorted_long_table_df['channel'].replace({my_channel:i})
#     unique_start_time = sorted_long_table_df.iloc[:, 2].unique()

#     #replacing start time column to numeric values (need to do this doem the from_long_to_nested function)
#     for i in range(len(unique_start_time)):
#         my_time=unique_start_time[i]
#         sorted_long_table_df['start time']=sorted_long_table_df['start time'].replace({my_time:i})

#     #excess columns are dropped for the frome_long_to_nested function
#     sorted_long_table_df_stripped=sorted_long_table_df.drop(columns=['event','name','end time'])

#     #sorted_long_table_df_stripped.head()
    
#     #table goes from long to nested 
#     df_nested = from_long_to_nested(sorted_long_table_df_stripped)

#     #create a list of labels 
#     new_unique_start_time=sorted_long_table_df.iloc[:, 2].unique()
#     labels=[]
#     for e in new_unique_start_time:
#         x=sorted_long_table_df.loc[sorted_long_table_df['start time']==e,[target]].iloc[0][0]
#         labels.append(x)

#     np_labels= np.asarray(labels, dtype=np.str)
    
#     return df_nested, np_labels 

def splitTestTrain(X, y, percent_train):
    msk = np.random.rand(len(X)) < percent_train
    ytrain=y[msk]
    ytest=y[~msk]
    Xtrain=X[msk]
    Xtest=X[~msk]
    
    return Xtrain, ytrain, Xtest, ytest

def concatenateMethod(classifier_num, X, y, percent_train):
    print("classification...")
    concatenation_instance=ColumnConcatenator()
    X_concatenated=concatenation_instance.fit(X).transform(X)
    Xtrain, ytrain, Xtest, ytest= splitTestTrain(X_concatenated,y,percent_train)
    print("Xtrain shape")
    print(Xtrain.shape)
    #print(Xtrain.head())
    if(classifier_num==classifier['TSF_CLF']):
        clf= TimeSeriesForestClassifier(n_estimators=10)
        clf.fit(Xtrain, ytrain)
        return clf.score(Xtest,ytest)
    elif (classifier_num==classifier['KNN_CLF']):
        knn = KNeighborsTimeSeriesClassifier(metric='dtw')
        knn.fit(Xtrain, ytrain)
        return knn.score(Xtest, ytest)
    elif (classifier_num == classifier['PF_CLF']):    
        pf = ProximityForest(n_trees=10)
        pf.fit(Xtrain, ytrain)
        return pf.score(Xtest, ytest)
    else: 
        return 0
        
#multivariate shapelet method         
def multivariateShapeletMethod(X, y, percent_train, time_contract=0.5):
    Xtrain, ytrain, Xtest, ytest= splitTestTrain(X,y,percent_train)
    clf = ShapeletTransformClassifier(time_contract_in_mins=time_contract)
    clf.fit(Xtrain, ytrain)
    return clf.score(Xtest, ytest)

def main():
    script = sys.argv[0]
    json_file_name = sys.argv[1]
    file_write_name=str(date.today())+":"+str(datetime.time(datetime.now()))+'.txt'
    file_write= open(file_write_name,"w+")
    #write to file here 
    file_write.write(json_file_name)
    print(json_file_name) 
    with open(json_file_name) as f:
        data = json.load(f)
    print(json.dumps(data, indent=4, sort_keys=True))    
  

    #setting values given the configuration files 
    target=data['targetCol']
    file_name=data['filePath']
    print(file_name)
    percent_train=data['percentTrain']
    TO_LOG= data['loggingEnabled']
    
    
    if(TO_LOG):
        file_write_name=str(date.today())+":::"+str(datetime.time(datetime.now()))+'.txt'
        file_write= open(file_write_name,"w+")
        file_write.write(json_file_name+'\n')
        file_write.write(file_name+'\n\n')

    
    
    X, y=reformatData(target,file_name)

    
    for job in data['jobs']:
        acc= 0
        print("JOB:")
        print(job)
        start_time=time.time()
        if(job['method']=='UNIVARIATE_TRANSFORMATION'):
            acc= concatenateMethod(classifier[job['classifier']], X, y, percent_train)
        elif(job['method']=='SHAPELET_TRANSFORM'):
            acc= multivariateShapeletMethod(X, y, 0.8)
        else: 
            raise ValueError(str(job['method']) +" method does not exist")
        print(acc)
        print("Total Time")
        end_time= time.time() - start_time
        print(end_time)
              
        if(TO_LOG):
            file_write.write('Job: '+str(job)+'\n') 
            file_write.write("Accuracy : "+str(acc)+'\n')
            file_write.write('Total Time : '+str(end_time)+'\n\n')


    if(TO_LOG):
        file_write.close()



main()
