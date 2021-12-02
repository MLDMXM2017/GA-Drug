# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:57:44 2020

@author: 叶晓娜

读取医学数据v2：读取基因差值和基因均值
"""

import numpy as np
import pandas as pd

def get_data():
    address1 = "./data/HT29/drugdrug_extract.csv"
    address2 = "./data/HT29/drugfeature_sig_extract.csv"

    with open(address1) as f:
        drugdrug = np.loadtxt(f,str,delimiter = ",",skiprows = 1,usecols = (2,3,10,11))
        #print(drugdrug[0])
    with open(address2) as f:
        drugfeature = np.loadtxt(f,str,delimiter = ",")
        drugfeature = drugfeature[:,1:] #remove gene id
        #print(drugfeature[0].size)
    
    data=np.zeros([1,1957], dtype = float)
    # data = np.zeros([1, 1958], dtype=float)
    # data = np.zeros([1, 979], dtype=float)
    # data = np.zeros([1, 3913], dtype=float)
    #print(data.shape)
    
    for i in range (0,725):    
        drug1ID=drugdrug[i][0]
        drug2ID=drugdrug[i][1]
        # S=drugdrug[i][2]
        L=drugdrug[i][3]
        drug1data=drugfeature[1:,list(drugfeature[0]).index(drug1ID)].astype(np.float32)
        drug2data=drugfeature[1:,list(drugfeature[0]).index(drug2ID)].astype(np.float32)
        
        #case1 drug1+drug2
        if L=="NA":
            L=0.0
        if L=="antagonism":
            L=1.0
        if L=="synergy":
            L=2.0
        union=np.hstack((drug1data,drug2data,L)).reshape(1,1957)
        data=np.vstack((data,union))
    
        #case2 drug1-drug2 / (drug1+drug2)/2
        # temp1=drug2data-drug1data
        # temp2=(drug1data+drug2data)/2
        # if L=="NA":
        #     L=0.0
        # if L=="antagonism":
        #     L=1.0
        # if L=="synergy":
        #     L=2.0
        # union=np.hstack((temp1,temp2,L)).reshape(1,1957)
        # data=np.vstack((data,union))

        #case3 drug2-drug1
        # temp = drug2data-drug1data
        # if L=="NA":
        #     L=0.0
        # if L=="antagonism":
        #     L=1.0
        # if L=="synergy":
        #     L=2.0
        # union=np.hstack((abs(temp),L)).reshape(1,979)
        # data=np.vstack((data,union))

        #case4 (drug1+drug2)/2
        # temp = (drug1data+drug2data)/2
        # if L == "NA":
        #     L = 0.0
        # if L == "antagonism":
        #     L = 1.0
        # if L == "synergy":
        #     L = 2.0
        # union = np.hstack((temp, L)).reshape(1, 979)
        # data = np.vstack((data, union))

        #case5 drug1 drug2 drug1+drug2
        # temp1=drug2data-drug1data
        # temp2 = (drug1data + drug2data) / 2
        # if L=="NA":
        #     L=0.0
        # if L=="antagonism":
        #     L=1.0
        # if L=="synergy":
        #     L=2.0
        # union=np.hstack((drug1data,drug2data,temp1,temp2,L)).reshape(1,3913)
        # data=np.vstack((data,union))

        #case6 drug1 drug2 drug1-drug2 drug2-drug1
        # temp1=drug2data-drug1data
        # temp2 =drug1data-drug2data
        # if L=="NA":
        #     L=0.0
        # if L=="antagonism":
        #     L=1.0
        # if L=="synergy":
        #     L=2.0
        # union=np.hstack((drug1data,drug2data,temp1,temp2,L)).reshape(1,3913)
        # data=np.vstack((data,union))

        #case7 drug1 drug2 + drug2 drug1,复刻
        # if L=="NA":
        #     L=0.0
        # if L=="antagonism":
        #     L=1.0
        # if L=="synergy":
        #     L=2.0
        # union1 = np.hstack((drug1data,drug2data,L)).reshape(1,1957)
        # union2 = np.hstack((drug2data, drug1data, L)).reshape(1, 1957)
        # data=np.vstack((data,union1,union2))

        #case8 abs(drug1-drug2) (drug1+drug2)/2
        # temp1=abs(drug2data-drug1data)
        # temp2=(drug1data+drug2data)/2
        # if L=="NA":
        #     L=0.0
        # if L=="antagonism":
        #     L=1.0
        # if L=="synergy":
        #     L=2.0
        # union=np.hstack((temp1,temp2,L)).reshape(1,1957)
        # data=np.vstack((data,union))

        #case9 (drug1-drug2)*(drug1+drug2)/2
        # temp1=drug2data-drug1data
        # temp2=(drug1data+drug2data)/2
        # if L=="NA":
        #     L=0.0
        # if L=="antagonism":
        #     L=1.0
        # if L=="synergy":
        #     L=2.0
        # union=np.hstack((temp1*temp2,L)).reshape(1,979)
        # data=np.vstack((data,union))
    data=data[1:,:] 

    print(data.shape)
    #print(data[0])
    print("=================data loading completed================") 
    return data

def get_toVerify_data():
    file_path = "./data/0108-toVerify-HT29.csv"
    address2 = "./data/HT29/drugfeature_sig_extract.csv"

    data = pd.read_csv(file_path)
    feature_data = pd.read_csv(address2)

    toVerify_data = []
    for index, row in data.iterrows():
        toVerify_data.append(np.hstack((
            feature_data[str(row['drug_row'])].values,
            feature_data[str(row['drug_col'])].values
        )))

    toVerify_data = np.array(toVerify_data)
    return toVerify_data

def get_test_data():
    '用于做泛化测试的数据'
    address1 = "./20210627test/PC3/drugdrug_extract.csv"
    address2 = "./20210627test/PC3/drugfeature_sig_extract.csv"

    with open(address1) as f:
        drugdrug = np.loadtxt(f, str, delimiter=",", skiprows=1, usecols=(3, 4, 9, 10))

    with open(address2) as f:
        drugfeature = np.loadtxt(f, str, delimiter=",")
        drugfeature = drugfeature[:, 1:]  # remove gene id


    data1 = np.zeros([1, 1957], dtype=float)

    # MCF7 665
    for i in range(0, len(drugdrug)):
        drug1ID = drugdrug[i][0]
        drug2ID = drugdrug[i][1]
        # S=drugdrug[i][2]
        L = drugdrug[i][3]
        drug1data = drugfeature[1:, list(drugfeature[0]).index(drug1ID)].astype(np.float32)
        drug2data = drugfeature[1:, list(drugfeature[0]).index(drug2ID)].astype(np.float32)

        # case1 drug1+drug2
        if L == "NA":
            L = 0.0
        if L == "antagonism":
            L = 1.0
        if L == "synergy":
            L = 2.0
        union = np.hstack((drug1data, drug2data, L)).reshape(1, 1957)
        data1 = np.vstack((data1, union))

    data1 = data1[1:, :]

    # address1 = "./20210627test/PC3/drugdrug_extract.csv"
    # address2 = "./20210627test/PC3/drugfeature_sig_extract.csv"
    #
    # with open(address1) as f:
    #     drugdrug = np.loadtxt(f, str, delimiter=",", skiprows=1, usecols=(3, 4, 9, 10))
    #
    # with open(address2) as f:
    #     drugfeature = np.loadtxt(f, str, delimiter=",")
    #     drugfeature = drugfeature[:, 1:]  # remove gene id
    #
    # data2 = np.zeros([1, 1957], dtype=float)
    #
    # for i in range(0, 652):
    #     drug1ID = drugdrug[i][0]
    #     drug2ID = drugdrug[i][1]
    #     # S=drugdrug[i][2]
    #     L = drugdrug[i][3]
    #     drug1data = drugfeature[1:, list(drugfeature[0]).index(drug1ID)].astype(np.float32)
    #     drug2data = drugfeature[1:, list(drugfeature[0]).index(drug2ID)].astype(np.float32)
    #
    #     # case1 drug1+drug2
    #     if L == "NA":
    #         L = 0.0
    #     if L == "antagonism":
    #         L = 1.0
    #     if L == "synergy":
    #         L = 2.0
    #     union = np.hstack((drug1data, drug2data, L)).reshape(1, 1957)
    #     data2 = np.vstack((data2, union))
    #
    # data2 = data2[1:, :]
    #
    # data = np.vstack((data1, data2))

    data = data1

    print(data.shape)
    print("=================data loading completed================")
    return data