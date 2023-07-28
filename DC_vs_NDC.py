#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:50:33 2023

@author: ognyansimeonov
"""

import graph_tool.all as gt
import numpy as np

g = gt.collection.ns["polblogs"]
g = gt.GraphView(g, directed=False)
g = gt.GraphView(g, vfilt=gt.label_largest_component(g, directed=False))
#ug = gt.GraphView(g, directed=False)

#print(ug)

#vprop = ug.new_vertex_property("int")

#for v in ug.vertices():
#   vprop[v] = v.out_degree()

deg = g.degree_property_map("out")
deg.a = np.sqrt(deg.a) +4

#conn = gt.GraphView(ug, vfilt=deg.fa > 0)


#print(conn)

gt.graph_draw(g, vertex_size = deg, output="Pol_Blogs_Plot.png")


stateNDC = gt.minimize_blockmodel_dl(g, multilevel_mcmc_args=dict(B_max=2), state_args =dict(deg_corr = False))
stateNDC.draw(vertex_size = deg, vertex_color = "black", output="NDC_state.png")

stateDC = gt.minimize_nested_blockmodel_dl(g, multilevel_mcmc_args=dict(B_max=2), state_args =dict(deg_corr = True))
stateDC.draw(vertex_size = deg, vertex_color = "black", output="DC_state.png")