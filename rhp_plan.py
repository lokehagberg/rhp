import numpy as np
from copy import deepcopy
from scipy import optimize

#See the rhp_intro.pdf for an explanation of the comments

#Plan function; full designates including exports
def plan(time_steps, planning_horizon, primary_resource_list, supply_use_list, use_imported_list, 
         depreciation_list, full_domestic_target_output_list, imported_target_output_list, 
         export_constraint_boolean, export_target_list, export_prices_list, import_prices_list):
    
    result_list, lagrange_list, slack_list = [], [], []

    for T in range(time_steps):

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
        modifier = deepcopy(np.zeros_like(full_domestic_target_output_list[0]))
        for i in range(planning_horizon+1):
            vertical_block_list.append(full_domestic_target_output_list[T+i] + modifier)
            modifier = deepcopy(np.matmul(depreciation_list[T+i], 
                                          np.sum(vertical_block_list, axis=0)))
        aggregate_constraint_vector = np.vstack(vertical_block_list)

        #Constructing the aggregate primary resource vector (each c)
        aggregate_primary_resource_list = []
        for i in range(planning_horizon+1):
            aggregate_primary_resource_list.append(primary_resource_list[T+i])
        aggregate_primary_resource_vector = np.vstack(aggregate_primary_resource_list)

        #Export constraint
        if export_constraint_boolean: 
            pass

        # Plan
        result = optimize.linprog(c=aggregate_primary_resource_vector, 
                                  A_ub=-aggregate_constraint_matrix, 
                                  b_ub=-aggregate_constraint_vector,
                                  bounds=(0, None), method='highs-ipm')
        print(result.success)
        print(result.status)
        lagrange_ineq = -optimize.linprog(c=aggregate_primary_resource_vector, 
                                          A_ub=-aggregate_constraint_matrix, 
                                          b_ub=-aggregate_constraint_vector, 
                                          bounds=(0, None), 
                                          method='highs-ipm')['ineqlin']['marginals']

        result_list.append(result.x)
        lagrange_list.append(lagrange_ineq)
        slack_list.append(result.slack)

    return([result_list, lagrange_list, slack_list])