#!/usr/bin/env python
# coding: utf-8

'''
configuration_checker.py
purpose:
1) To check if the configuration file is properly formatted
   * Is there anything that shouldn't be specified
   * Are there any values that shouldn't be in the data
2) Check if the data is formatted properly 
also an important note is to warn them about the size of the dataset 
    if the dataset is small, then specifying a large percent train is not suggested 
    if they are leading zeros in the last column,it may mess up stuff 
'''

import json
import sys
import pandas as pd 

required_keys=['filePath', 'loggingEnabled','targetCol', 'percentTrain', 'jobs']
u_classifiers=['TSF_CLF','KNN_CLF','PF_CLF']
headers=['name','event','channel','start time', 'end time']
def check_job(j):
    job_keys=j.keys()
    if ('method' not in job_keys):
        print("you need to specify the method")
    else:
        method= j['method']
        classifier=""
        if (method == 'UNIVARIATE_TRANSFORMATION'):
            if('classifier' not in job_keys):
                print("A classifier must be specified for univariate transform method")
            else:
                classifier=j['classifier'] 
            if(classifier not in u_classifiers):
                print("This classifier is not implemented")
        elif (method=="SHAPELET_TRANSFORM"):
            if ("classifier" in  job_keys):
                print("classifier does not need to be specified for shapelet transform method")
        elif (method == 'COLUMN_ENSEMBLE'): 
            info=j['ensembleInfo']
            for e in info:
                if("columnNum" not in e.keys()):
                    print("A column number must be specified for the column ensemble method")
                if('classifier' not in e.keys()):
                    print("A classifier must be specified for univariate transform method")
                else:
                    classifier=e['classifier'] 
                if(classifier not in u_classifiers):
                    print("This classifier is not implemented")
        else:
            print("Method "+method +" is not supported")
                
            
def main():
    #read in json file 
    script = sys.argv[0]
    json_file_name = sys.argv[1]
    with open(json_file_name) as f:
        data = json.load(f)


    #check the keys are right 
    print("checking configuration file: "+json_file_name+"...\n")
    input_keys=data.keys()
    for e in input_keys:
        if e not in required_keys:
            print (e +" is not a valid key within your inputed json file.")
            print("Valid keys include:")
            print(str(input_keys))
     #this is also where you can go ahead and also check the value of what you are sending in        
    
    #check the format/values of these jobs 
    if "jobs" in input_keys:
        all_jobs= data['jobs']
        for j in all_jobs:
            check_job(j)

    #check the value of within the json files 
    if "percentTrain" in input_keys:
        print("hey")

    if "percentTrain" in input_keys:
        print("hey")

    if "targetCol" in input_keys:
        print("hey")
 
    print("checking data file...")
    if "filePath" in input_keys:
        file_name=data['filePath']
        X=pd.read_csv(file_name)
        for e in list(X)[0:5]:
            if e not in headers:
                print(e + " is not a supported column")
                print("supported columns include "+ str(headers))

main()
