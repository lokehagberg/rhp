import numpy as np
from copy import deepcopy
from scipy import optimize


T=0
depreciation_list = [np.matrix([[0.5,0],[0,0.5]]),np.matrix([[0.5,0],[0,0.5]]),np.matrix([[0.5,0],[0,0.5]]),np.matrix([[0.5,0],[0,0.5]])]
planning_horizon = 2
supply_use_list = [np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]])]
full_target_output_list = [np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1])]
primary_resource_list = [np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1])]
export_constraint = False

#Constructing the aggregate constraint matrix (each DJ)
zero_matrix = np.zeros_like(supply_use_list[0])
horizontal_block_list = []
supply_use_part = deepcopy(supply_use_list[T])
for i in range(planning_horizon+1):
    if (i < planning_horizon): 
        zero_part = deepcopy(np.block([zero_matrix]*(planning_horizon-i)))        
        horizontal_block_list.append(np.block([supply_use_part, zero_part]))
        #depreciation
        supply_use_part = deepcopy(np.matmul(depreciation_list[T+i],supply_use_part))
        supply_use_part = deepcopy(np.block([supply_use_part,supply_use_list[T+i]])) 
horizontal_block_list.append(supply_use_part)
aggregate_constraint_matrix = np.vstack(horizontal_block_list)

#Constructing the aggregate constraint vector (each Dr)
vertical_block_list = []
modifier = deepcopy(np.zeros_like(full_target_output_list[0]))
for i in range(planning_horizon+1):
    vertical_block_list.append(full_target_output_list[T+i] + modifier)
    modifier = deepcopy(np.matmul(depreciation_list[T+i], np.sum(vertical_block_list, axis=0)))
aggregate_constraint_vector = np.vstack(vertical_block_list)

#Constructing the aggregate primary resource vector (each c)
aggregate_primary_resource_list = []
for i in range(planning_horizon+1):
    aggregate_primary_resource_list.append(primary_resource_list[T+i])
aggregate_primary_resource_vector = np.vstack(aggregate_primary_resource_list)

#Export constraint
if export_constraint: 
    pass

