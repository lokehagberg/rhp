import numpy as np
from copy import deepcopy
from scipy import optimize


N=0
imported_target_output_list = [np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1])]
supply_use_list = [np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[0.5,0],[0,0.5]])]
zero_matrix = np.zeros_like(supply_use_list[0])
use_imported_list = [np.matrix([[0.5,1],[0,0.5]]),np.matrix([[0.5,0],[0,0.5]]),np.matrix([[0.5,0],[0,0.5]]),np.matrix([[0.5,0],[0,0.5]])]
planning_horizon = 2
supply_use_list = [np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]])]
full_target_output_list = [np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1])]
import_prices_list = [np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]]),np.matrix([[1,0],[0,1]])]
export_prices_list = [np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1]),np.array([1,1]).reshape([-1,1])]
export_constraint = True

#Import prices list is a list of matrices (with only diagonal elements), TODO DIAGFLAT IT!
# export prices are a list of vectors

#Constructing the use imports constraint matrix (each p_imp^T T)
horizontal_block_list = []
for i in range(planning_horizon+1):
    price_augmented_use_imports = deepcopy(np.matmul(import_prices_list[N+i], 
                                                     np.transpose(use_imported_list[N+i])))
    horizontal_block_list.append(np.transpose(-price_augmented_use_imports))
use_imports_constraint_matrix = np.hstack(horizontal_block_list)

#Constructing the export constraint matrix (each p_exp^T)
horizontal_block_list = []
for i in range(planning_horizon+1):
    horizontal_block_list.append(np.transpose(export_prices_list[N+i]))
aggregate_export_prices = np.hstack(horizontal_block_list)
aggregate_export_price_matrix = np.diagflat(aggregate_export_prices)
zero_vector_list = [np.zeros_like(aggregate_export_prices)]*(planning_horizon+1)*(supply_use_list[0].shape[0])
aggregate_zero_matrix = np.vstack(zero_vector_list)
export_constraint_matrix = np.vstack([aggregate_zero_matrix, aggregate_export_price_matrix])

#Constructing the directly imported target output constraint vector (p_imp^T r_imp)
vertical_block_list = []
for i in range(planning_horizon+1):
    vertical_block_list.append(np.matmul(import_prices_list[N+i], imported_target_output_list[N+i]))
imported_target_output_constraint_vector = np.vstack(vertical_block_list)

