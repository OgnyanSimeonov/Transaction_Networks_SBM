#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:38:45 2023

@author: ognyansimeonov
"""

import pandas as pd
import numpy as np
from sbmtm import sbmtm
import graph_tool.all as gt
import matplotlib.pyplot as plt

#import data
input_df_no_outcomes = pd.read_csv('/Users/ognyansimeonov/Desktop/Extended_Project/Code/bipartite_adjacency_matrix_cities.csv', index_col='Customer_ID')

#clean data
transaction_df=pd.DataFrame(np.where(input_df_no_outcomes.eq(1), input_df_no_outcomes.columns, input_df_no_outcomes), 
                  index=input_df_no_outcomes.index, 
                  columns=input_df_no_outcomes.columns)
features = [[t for t in transaction if t != 0] for transaction in transaction_df.values.tolist()]
id = [h.split()[0] for h in transaction_df.index.values.astype('str')]

#make model
model = sbmtm()
model.make_graph(features,documents=id,counts=False)
model.g

#fit model
model.fit(n_init=100)

#plot graph
model.state.draw(subsample_edges=1000,layout='bipartite',bip_aspect=1,
           hvertex_size=8, hedge_pen_width=1.9, hedge_color="#00deff", hvertex_fill_color="#00DEFF", output_size=(600, 600))

