# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:06:37 2021

@author: zshaf
"""

import pandas as pd
import sqlite3
from scipy import stats
import numpy as np
import boto3
from botocore.exceptions import ClientError
import logging
import os
from matplotlib import pyplot as plt

def upload_file_S3(file_name,bucket,object_name=None):
    
    #Input Credentials to AWS
    session = boto3.Session(
        aws_access_key_id = '<Input_Access_Key_ID>',
        aws_secret_access_key = '<Input_Secret_Access_Key>')
    #Open Link to AWS S3
    s3 = session.client('s3')
    
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    #Uploading to s3 from file on disk
    #Could be changed to upload directly from code through a pickle dump
    #into a bytes object
    try:
        with open(file_name,'rb') as f:
            s3.upload_fileobj(f,bucket,object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def plot_test(Comb_DF,Comb_DF_Filtered):
    
    feat_0_unf = Comb_DF['feat_0']
    feat_0_fil = Comb_DF_Filtered['feat_0']
    
    plt.plot(feat_0_unf)
    plt.plot(feat_0_fil)
    plt.title('Feature 0 Unfiltered Vs Filtered')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.legend(['Feat_0_Unf','Feat_0_Fil'])
    plt.show()
    
    feat_1_unf = Comb_DF['feat_1']
    feat_1_fil = Comb_DF_Filtered['feat_1']
    
    plt.plot(feat_1_unf)
    plt.plot(feat_1_fil)
    plt.title('Feature 1 Unfiltered Vs Filtered')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.legend(['Feat_1_Unf','Feat_1_Fil'])
    plt.show()
    
    feat_2_unf = Comb_DF['feat_2']
    feat_2_fil = Comb_DF_Filtered['feat_2']
    
    plt.plot(feat_2_unf)
    plt.plot(feat_2_fil)
    plt.title('Feature 2 Unfiltered Vs Filtered')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.legend(['Feat_2_Unf','Feat_2_Fil'])
    plt.show()
    
    feat_3_unf = Comb_DF['feat_3']
    feat_3_fil = Comb_DF_Filtered['feat_3']
    
    plt.plot(feat_3_unf)
    plt.plot(feat_3_fil)
    plt.title('Feature 3 Unfiltered Vs Filtered')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.legend(['Feat_3_Unf','Feat_3_Fil'])
    plt.show()
    

def main():
    
    ##Open up SQL Database and transform into pandas databases
    #Currently database is setup to be in same directory as script but can
    #be changed with further functionalization to allow access to anywhere
    try:
        conn = sqlite3.connect("exampleco_db.db")    
    except Exception as e:
        print(e)

    #Now in order to read in pandas dataframe we need to know table name
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(f"Table Name : {cursor.fetchall()}")

    #Pull tables from database into dataframes
    static_data = pd.read_sql_query('SELECT * FROM static_data', conn)
    feat_0 = pd.read_sql_query('SELECT * FROM feat_0', conn)
    feat_0 = feat_0.rename(columns={'value':'feat_0'})

    feat_1 = pd.read_sql_query('SELECT * FROM feat_1', conn)
    feat_1 = feat_1.rename(columns={'value':'feat_1'})

    feat_2 = pd.read_sql_query('SELECT * FROM feat_2', conn)
    feat_2 = feat_2.rename(columns={'value':'feat_2'})

    feat_3 = pd.read_sql_query('SELECT * FROM feat_3', conn)
    feat_3 = feat_3.rename(columns={'value':'feat_3'})
    conn.close()

    #Combine dataframes into one
    Comb_DF = pd.concat([feat_0,feat_1['feat_1'],feat_2['feat_2'],feat_3['feat_3']],axis=1)

    #Remove statistical outliers
    #Marks with boolean all data points with a z-score greater than 3
    z_scores_feat_0 = stats.zscore(Comb_DF['feat_0'])
    abs_z_scores_feat_0 = np.abs(z_scores_feat_0)
    filtered_entries_feat_0 = (abs_z_scores_feat_0 < 3)

    z_scores_feat_1 = stats.zscore(Comb_DF['feat_1'])
    abs_z_scores_feat_1 = np.abs(z_scores_feat_1)
    filtered_entries_feat_1 = (abs_z_scores_feat_1 < 3)

    z_scores_feat_2 = stats.zscore(Comb_DF['feat_2'])
    abs_z_scores_feat_2 = np.abs(z_scores_feat_2)
    filtered_entries_feat_2 = (abs_z_scores_feat_2 < 3)

    z_scores_feat_3 = stats.zscore(Comb_DF['feat_3'])
    abs_z_scores_feat_3 = np.abs(z_scores_feat_3)
    filtered_entries_feat_3 = (abs_z_scores_feat_3 < 3)

    #Use boolean logic to combine arrays into single boolean array
    filtered_0_1 = np.logical_and(filtered_entries_feat_0,filtered_entries_feat_1)
    filtered_2_3 = np.logical_and(filtered_entries_feat_2,filtered_entries_feat_3)

    filtered_all = np.logical_and(filtered_0_1,filtered_2_3)

    #Combine original dataframe with boolean array for removal
    Comb_DF_Bools = pd.concat([Comb_DF,pd.Series(filtered_all)],axis=1)

    #Remove all values of False from dataframe
    Comb_DF_Filtered = Comb_DF_Bools.loc[Comb_DF_Bools[0]]

    #Remove boolean values from dataframe as is extraneous at this point
    Comb_DF_Drop_Bool = Comb_DF_Filtered.drop([0],axis=1)
    
    #Reset Comb_DF indices for adding in static data
    Comb_DF_Drop_Bool = Comb_DF_Drop_Bool.reset_index(drop=True)
    
    #Iterate Through DF to match machine numbers to add static data
    Static_DF = pd.DataFrame()
    
    #Bunch of if and elif statements to match the machine numbers from the Combined
    #DataFrame with the matching static
    #There is definitely a faster/easier way to do this but this was the first
    #method that popped into my mind. Can be funtionalized for better asthetics
    #and speed
    for index,row in Comb_DF_Drop_Bool.iterrows():
        if row.str.contains('machine_0',regex=False).any() == True:
            data = static_data.loc[0,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_1',regex=False).any() == True:
            data = static_data.loc[1,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)     
        elif row.str.contains('machine_2',regex=False).any() == True:
            data = static_data.loc[2,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_3',regex=False).any() == True:
            data = static_data.loc[3,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_4',regex=False).any() == True:
            data = static_data.loc[4,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_5',regex=False).any() == True:
            data = static_data.loc[5,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_6',regex=False).any() == True:
            data = static_data.loc[6,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_7',regex=False).any() == True:
            data = static_data.loc[7,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_8',regex=False).any() == True:
            data = static_data.loc[8,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_9',regex=False).any() == True:
            data = static_data.loc[9,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_10',regex=False).any() == True:
            data = static_data.loc[10,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_11',regex=False).any() == True:
            data = static_data.loc[11,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_12',regex=False).any() == True:
            data = static_data.loc[12,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_13',regex=False).any() == True:
            data = static_data.loc[13,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_14',regex=False).any() == True:
            data = static_data.loc[14,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_15',regex=False).any() == True:
            data = static_data.loc[15,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_16',regex=False).any() == True:
            data = static_data.loc[16,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_17',regex=False).any() == True:
            data = static_data.loc[17,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_18',regex=False).any() == True:
            data = static_data.loc[18,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
        elif row.str.contains('machine_19',regex=False).any() == True:
            data = static_data.loc[19,'install_date':'room'].transpose()
            Static_DF = Static_DF.append(data)
    #Reset Static_DF indices for proper concatenation
    Static_DF = Static_DF.reset_index(drop=True)     
    Comb_DF_Out = pd.concat([Comb_DF_Drop_Bool,Static_DF],axis=1)
    #Transform dataframe into numpy array
    Comb_Arr = Comb_DF_Out.to_numpy()
    
    #Save array to disk for upload
    np.save('ExampleCo_Data',Comb_Arr,allow_pickle=True)
    
    #Creates plot for visualization of data
    #plot_test(Comb_DF,Comb_DF_Out)
    
    #Function to upload data to S3. Is currently commented out as user info 
    #is not properly setup
    #upload_file_S3('ExampleCo_Data.npy','bucket_name')
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
