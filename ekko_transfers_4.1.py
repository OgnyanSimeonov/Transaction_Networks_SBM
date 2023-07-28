#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:11:47 2023

@author: ognyansimeonov
"""

import pandas as pd
import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


transfers_df = pd.read_csv('/Users/ognyansimeonov/Desktop/Extended_Project/Code/Transfer_Data.csv')
customer_info = pd.read_csv('/Users/ognyansimeonov/Desktop/Extended_Project/Code/ekko_customers_info.csv')

transfers_df = transfers_df.head(4000)

# Create an empty graph
g = gt.Graph()

# Assuming you have a list of nodes and edges from your data
nodes = transfers_df['Sender_customer_Id'].unique().tolist()
nodes += transfers_df['Receiver_Sort_Code_Acc_Num'].unique().tolist()
edges = list(zip(transfers_df['Sender_customer_Id'], transfers_df['Receiver_Sort_Code_Acc_Num']))

# Create a vertex property map to store node labels
vlabel = g.new_vertex_property('string')
postcode = g.new_vertex_property('string')
bank = g.new_vertex_property('string')

# Add nodes to the graph
vertex_map = {}
for node in nodes:
    v = g.add_vertex()
    vlabel[v] = str(node)
    vertex_map[str(node)] = v

# Create an edge property map to store edge weights
weight = g.new_edge_property('double')

# Add edges to the graph with weight 1
for edge in edges:
    source, target = edge[0], edge[1]
    e = g.add_edge(vertex_map[str(source)], vertex_map[str(target)])
    weight[e] = 10

# Set the postcode property for sender vertices
for node in nodes:
    if str(node) in customer_info['Sender_customer_Id'].values:
        sender_postcode = customer_info.loc[customer_info['Sender_customer_Id'] == str(node), 'address.townOrCity'].iloc[0]
        v = vertex_map[str(node)]
        postcode[v] = sender_postcode

# Set the bank property for receiver vertices
for node in nodes:
    if str(node) in transfers_df['Receiver_Sort_Code_Acc_Num'].values:
        receiver_bank = transfers_df.loc[transfers_df['Receiver_Sort_Code_Acc_Num'] == str(node), 'Bank_of_Receiver'].iloc[0]
        v = vertex_map[str(node)]
        bank[v] = receiver_bank
    
# Create a combined property map for postcode and bank
postcode_bank = g.new_vertex_property('string')

# Iterate over the vertices and set the combined property values
for v in g.vertices():
    p = postcode[v]
    b = bank[v]
    if p is not None and b is not None:
        if p != '.':
            postcode_bank[v] = f"{b}\n{p}"
        else:
            postcode_bank[v] = b
    elif p is not None:
        postcode_bank[v] = p
    elif b is not None:
        postcode_bank[v] = b
    else:
        postcode_bank[v] = ''

# Create a subgraph containing only the largest component
g = gt.GraphView(g, vfilt=gt.label_largest_component(g, directed=False))

# Create a new edge set for same bank connections
same_bank_edges = set()

# Get the largest component subgraph
largest_component = gt.GraphView(g, vfilt=gt.label_largest_component(g, directed=False))

# Iterate over the vertices in the largest component that belong to Receiver_Sort_Code_Acc_Num
for v in largest_component.iter_vertices():
    if bank[v] != '':
        bank_attr = bank[v]  # Get the bank attribute of the vertex

        # Find other vertices in the largest component with the same bank attribute that belong to Receiver_Sort_Code_Acc_Num
        same_bank_vertices = [v2 for v2 in largest_component.iter_vertices() if v2 != v and bank[v2] == bank_attr and vlabel[v2] in transfers_df['Receiver_Sort_Code_Acc_Num'].values]

        # Create pairs of vertices with the same bank attribute
        node_pairs = [(v, v2) for v2 in same_bank_vertices]

        # Add the pairs to the set of same bank edges
        same_bank_edges.update(node_pairs)

# Convert the set of same bank edges to a list
same_bank_edges_list = list(same_bank_edges)

# Create a set to keep track of processed node pairs
processed_pairs = set()

# Add the same bank edges as undirected edges to the graph with weight 0.5
for edge in same_bank_edges_list:
    node_pair = frozenset(edge)  # Convert the edge to a frozenset for comparison
    if node_pair not in processed_pairs:
        e = g.add_edge(edge[0], edge[1])
        weight[e] = 1
        processed_pairs.add(node_pair)

# Create a new edge set for same postcode connections
same_postcode_edges = set()

# Get the largest component subgraph
largest_component = gt.GraphView(g, vfilt=gt.label_largest_component(g, directed=False))

# Iterate over the vertices in the largest component that belong to Sender_customer_Id
for v in largest_component.iter_vertices():
    if postcode[v] != '':
        postcode_attr = postcode[v]  # Get the postcode attribute of the vertex

        # Find other vertices in the largest component with the same postcode attribute that belong to Sender_customer_Id
        same_postcode_vertices = [v2 for v2 in largest_component.iter_vertices() if v2 != v and postcode[v2] == postcode_attr and vlabel[v2] in transfers_df['Sender_customer_Id'].values]

        # Create pairs of vertices with the same postcode attribute
        node_pairs = [(v, v2) for v2 in same_postcode_vertices]

        # Add the pairs to the set of same postcode edges
        same_postcode_edges.update(node_pairs)

# Convert the set of same postcode edges to a list
same_postcode_edges_list = list(same_postcode_edges)

# Create a set to keep track of processed node pairs
processed_pairs = set()

# Add the same postcode edges as undirected edges to the graph with weight 0.5
for edge in same_postcode_edges_list:
    node_pair = frozenset(edge)  # Convert the edge to a frozenset for comparison
    if node_pair not in processed_pairs:
        e = g.add_edge(edge[0], edge[1])
        weight[e] = 1
        processed_pairs.add(node_pair)

# Plot the largest component with the same bank and postcode connections
#gt.graph_draw(g, vertex_font_size=12, vertex_text=postcode_bank, edge_pen_width=1.5,
#              output_size=(10000, 10000), output="ekko_largest_component3.png")


state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(eweight = weight))  # Perform weighted clustering
state.draw(vertex_font_size=12, vertex_text=postcode_bank, edge_pen_width=1.5,
           output_size=(10000, 10000), vertex_color="black", output="ekko_states_nested_weighted=10.png")

"""
# Get the most common name of the postcode_bank property for each cluster
cluster_names = {}
for v in g.vertices():
    cluster_index = int(state.b[v])
    if cluster_index not in cluster_names:
        nodes_in_cluster = [vlabel[u] for u in g.vertices() if int(state.b[u]) == cluster_index]
        names_count = Counter(postcode_bank[u] for u in g.vertices() if int(state.b[u]) == cluster_index)
        most_common_name = names_count.most_common(1)[0][0]
        cluster_names[cluster_index] = most_common_name
print(cluster_names) 



# Iterate over clusters with city names
for cluster_index, cluster_name in cluster_names.items():
    if cluster_name != '':
        # Count the number of connections from the current city cluster to other clusters
        connection_counts = Counter()
        for e in g.edges():
            source_cluster = int(state.b[e.source()])
            target_cluster = int(state.b[e.target()])
            if source_cluster == cluster_index and target_cluster != cluster_index:
                connection_counts[target_cluster] += 1

        # Find the cluster with the most connections to the current city cluster
        most_connected_cluster = connection_counts.most_common(1)
        if most_connected_cluster:
            most_connected_cluster_index = most_connected_cluster[0][0]
            most_connected_cluster_count = most_connected_cluster[0][1]
            most_connected_cluster_name = cluster_names[most_connected_cluster_index]

            # Print the information about the cluster with the most connections to the current city cluster
            print(f"The cluster with the most connections to the '{cluster_name}' cluster is {most_connected_cluster_name}.")
            print(f"It has {most_connected_cluster_count} connections.")
            print()

"""