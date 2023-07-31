#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:34:39 2023

@author: ognyansimeonov
"""

import pandas as pd
import numpy as np

#------------------------------Banks-------------------------------------------
# Assuming you have already read the CSV into transfers_df
transfers_df = pd.read_csv('/Users/ognyansimeonov/Desktop/Extended_Project/Code/Transfer_Data.csv')

# Create a new DataFrame with unique customer IDs as rows and unique bank names as columns
customer_ids = transfers_df['Sender_customer_Id'].unique()
bank_names = transfers_df['Bank_of_Receiver'].unique()
adjacency_matrix_df = pd.DataFrame(0, index=customer_ids, columns=bank_names)

# Update the adjacency matrix with 1 where a customer has sent money to a specific bank
for index, row in transfers_df.iterrows():
    customer_id = row['Sender_customer_Id']
    bank_name = row['Bank_of_Receiver']
    adjacency_matrix_df.at[customer_id, bank_name] = 1

# Optionally, if you want to reset the index name and column name:
adjacency_matrix_df.index.name = 'Customer_ID'
adjacency_matrix_df.columns.name = 'Bank_Name'

# The adjacency_matrix_df now contains the desired adjacency matrix
adjacency_matrix_df.to_csv('/Users/ognyansimeonov/Desktop/Extended_Project/Code/bipartite_adjacency_matrix.csv')


#----------------------------Cities--------------------------------------------
# Assuming you have already read the CSV into transfers_df
customer_info = pd.read_csv('/Users/ognyansimeonov/Desktop/Extended_Project/Code/ekko_customers_info.csv')

# Create a new DataFrame with unique customer IDs as rows and unique bank names as columns
customer_ids = customer_info['Sender_customer_Id'].unique()
city_names = customer_info['address.townOrCity'].unique()
adjacency_matrix_df = pd.DataFrame(0, index=customer_ids, columns=city_names)

# Update the adjacency matrix with 1 where a customer has sent money to a specific bank
for index, row in customer_info.iterrows():
    customer_id = row['Sender_customer_Id']
    city_name = row['address.townOrCity']
    adjacency_matrix_df.at[customer_id, city_name] = 1

# Optionally, if you want to reset the index name and column name:
adjacency_matrix_df.index.name = 'Customer_ID'
adjacency_matrix_df.columns.name = 'City_Name'

# The adjacency_matrix_df now contains the desired adjacency matrix
adjacency_matrix_df.to_csv('/Users/ognyansimeonov/Desktop/Extended_Project/Code/bipartite_adjacency_matrix_cities.csv')

