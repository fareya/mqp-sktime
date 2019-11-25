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
TO_LOG= True

TSF_default_parameters={'base_estimator':None, 'n_estimators':10, 'criterion':'entropy', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'max_leaf_nodes':None, 'min_impurity_decrease':0.0, 'min_impurity_split':None, 'bootstrap':False, 'oob_score':False, 'n_jobs':None, 'random_state':None, 'verbose':0, 'warm_start':False, 'class_weight':None}

def list_to_dict(a): 
    it = iter(a) 
    res_dct = dict(zip(it, it)) 
    return res_dct 

#extracting the data from the csv as a dataframe and reformatting it for sktime compatibility 
def reformatData(target, file_name):
    
    print("reformatting the data...")
    raw_df= pd.read_csv(file_name)
    
    #collapses the time cols into one single time column to match the rest of the columns 
    long_table_df= raw_df.melt(id_vars=["event", "name","start time", "end time","channel"], 
            var_name="anindex", 
            value_name="value")

    sorted_long_table_df=long_table_df.sort_values(by=['event','name','start time','channel'], axis=0)

    #start time qualifies as a unique start time
    #maybe will switch this out later 
    unique_dim_ids = sorted_long_table_df.iloc[:, 4].unique()

    #replacing channel named to numeric values (need to do this doem the from_long_to_nested function)
    for i in range(len(unique_dim_ids)):
        my_channel=unique_dim_ids[i]
        sorted_long_table_df['channel']=sorted_long_table_df['channel'].replace({my_channel:i})
    unique_start_time = sorted_long_table_df.iloc[:, 2].unique()

    #replacing start time column to numeric values (need to do this doem the from_long_to_nested function)
    for i in range(len(unique_start_time)):
        my_time=unique_start_time[i]
        sorted_long_table_df['start time']=sorted_long_table_df['start time'].replace({my_time:i})

    #excess columns are dropped for the frome_long_to_nested function
    sorted_long_table_df_stripped=sorted_long_table_df.drop(columns=['event','name','end time'])

    #sorted_long_table_df_stripped.head()
    
    #table goes from long to nested 
    df_nested = from_long_to_nested(sorted_long_table_df_stripped)

    #create a list of labels 
    new_unique_start_time=sorted_long_table_df.iloc[:, 2].unique()
    labels=[]
    for e in new_unique_start_time:
        x=sorted_long_table_df.loc[sorted_long_table_df['start time']==e,[target]].iloc[0][0]
        labels.append(x)

    np_labels= np.asarray(labels, dtype=np.str)
    
    return df_nested, np_labels 

# splitting the test train based on the test train that is specified 
# i think that there might be issues with this function 
def splitTestTrain(X, y, percent_train):
    msk = np.random.rand(len(X)) < percent_train
    ytrain=y[msk]
    ytest=y[~msk]
    Xtrain=X[msk]
    Xtest=X[~msk]
    
    return Xtrain, ytrain, Xtest, ytest


#concatenate multivariate time series and classify using the specified classifier 
def concatenateMethod(classifier_num, X, y, percent_train):#,clf_parameters):
    print("classification...")
    concatenation_instance=ColumnConcatenator()
    X_concatenated=concatenation_instance.fit(X).transform(X)
    Xtrain, ytrain, Xtest, ytest= splitTestTrain(X_concatenated,y,percent_train)
    if(classifier_num==classifier['TSF_CLF']):
        #TSF_params=TSF_default_parameters
        #clf_params_dict=list_to_dict(clf_parameters)
        #print(TSF_params)
        #for e in clf_params_dict:
         #   TSF_params[e]=clf_params_dict[e]
        #print(TSF_params)
       # clf= TimeSeriesForestClassifier(base_estimator=TSF_params['base_estimator'], n_estimators=TSF_params['n_estimators'], criterion=TSF_params['criterion'], max_depth=TSF_params['max_depth'], min_samples_split=TSF_params['min_samples_split'], min_samples_leaf=TSF_params['min_samples_leaf'], min_weight_fraction_leaf=TSF_params['min_weight_fraction_leaf'], max_features=TSF_params['max_features'], max_leaf_nodes=TSF_params['max_leaf_nodes'], min_impurity_decrease=TSF_params['min_impurity_decrease'], min_impurity_split=TSF_params['min_impurity_split'], bootstrap=TSF_params['bootstrap'], oob_score=TSF_params['oob_score'], n_jobs=TSF_params['n_jobs'], random_state=TSF_params['random_state'], verbose=TSF_params['verbose'], warm_start=TSF_params['warm_start'], class_weight=TSF_params['class_weight'])
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
        raise ValueError("Specified classifier is not an option")


#multivariate shapelet method         
def multivariateShapeletMethod(X, y, percent_train, time_contract=0.5):
    Xtrain, ytrain, Xtest, ytest= splitTestTrain(X,y,percent_train)
    clf = ShapeletTransformClassifier(time_contract_in_mins=time_contract)
    clf.fit(Xtrain, ytrain)
    return clf.score(Xtest, ytest)


def main():
    script = sys.argv[0]
    json_file_name = sys.argv[1]

    #reading the inputted json file 
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
            acc= concatenateMethod(classifier[job['classifier']], X, y, percent_train)#,job['parameters'])
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





