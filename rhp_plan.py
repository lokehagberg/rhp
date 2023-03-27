import numpy as np
from copy import deepcopy
from scipy import optimize

#See the rhp_intro.pdf for an explanation of the comments

#Plan function; full designates including exports
def plan(time_steps, planning_horizon, primary_resource_list, supply_use_list, use_imported_list, 
         depreciation_list, full_domestic_target_output_list, imported_target_output_list, 
         export_constraint_boolean, export_prices_list, import_prices_list):
    
    result_list, lagrange_list, slack_list = [], [], []

    for N in range(time_steps):

        #Constructing the aggregate constraint matrix (each DJ)
        zero_matrix = np.zeros_like(supply_use_list[0])
        horizontal_block_list = []
        supply_use_part = deepcopy(supply_use_list[N])
        for i in range(planning_horizon+1):
            if (i < planning_horizon): 
                zero_part = deepcopy(np.block([zero_matrix]*(planning_horizon-i)))        
                horizontal_block_list.append(np.block([supply_use_part, zero_part]))
                #depreciation
                supply_use_part = deepcopy(np.matmul(depreciation_list[N+i],supply_use_part))
                supply_use_part = deepcopy(np.block([supply_use_part,supply_use_list[N+i]])) 
        horizontal_block_list.append(supply_use_part)
        aggregate_constraint_matrix = np.vstack(horizontal_block_list)

        #Constructing the aggregate constraint vector (each Dr)
        vertical_block_list = []
        modifier = deepcopy(np.zeros_like(full_domestic_target_output_list[0]))
        for i in range(planning_horizon+1):
            vertical_block_list.append(full_domestic_target_output_list[N+i] + modifier)
            modifier = deepcopy(np.matmul(depreciation_list[N+i], 
                                          np.sum(vertical_block_list, axis=0)))
        aggregate_constraint_vector = np.vstack(vertical_block_list)

        #Constructing the aggregate primary resource vector (each c)
        aggregate_primary_resource_list = []
        for i in range(planning_horizon+1):
            aggregate_primary_resource_list.append(primary_resource_list[N+i])
        aggregate_primary_resource_vector = np.vstack(aggregate_primary_resource_list)

        #Export constraint
        if export_constraint_boolean: 
    
            #Import prices list is a list of vectors, but it is converted to matrices where the
            # diagonal takes on the vectors values and the other elements are 0
            import_prices_list = deepcopy([np.diagflat(import_prices_list[i]) for i in range(len(import_prices_list))])

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

            #Constructing the export augmentet constraint objects
            aggregate_constraint_matrix = deepcopy(np.vstack(aggregate_constraint_matrix, use_imports_constraint_matrix))
            aggregate_constraint_matrix = deepcopy(np.hstack(aggregate_constraint_matrix, export_constraint_matrix))
            aggregate_constraint_vector = deepcopy(np.vstack(aggregate_constraint_vector, imported_target_output_constraint_vector))
            aggregate_primary_resource_vector = deepcopy(np.vstack(aggregate_primary_resource_vector, np.ones_like(imported_target_output_constraint_vector)))

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

    return([result_list, lagrange_list, slack_list, aggregate_primary_resource_vector, aggregate_constraint_matrix, aggregate_constraint_vector])