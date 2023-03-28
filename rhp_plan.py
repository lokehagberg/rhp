import numpy as np
from copy import deepcopy
from scipy import optimize

#See the rhp_intro.pdf for an explanation of the comments

#Plan function; full designates including exports
def plan(time_steps, planning_horizon, primary_resource_list, supply_use_list, use_imported_list, 
         depreciation_list, full_domestic_target_output_list, imported_target_output_list, 
         export_constraint_boolean, export_prices_list, import_prices_list, upper_bound_on_activity,
         max_iterations, tolerance):
    
    result_list, lagrange_list, slack_list = [], [], []
    modifier_value = deepcopy(np.zeros_like(full_domestic_target_output_list[0]))

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
        modifier = deepcopy(-modifier_value)
        for i in range(planning_horizon+1):
            vertical_block_list.append(np.sum([full_domestic_target_output_list[N+i], modifier], axis=0))
            #depreciation and carry
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
                export_price_diagonal = deepcopy(np.transpose(export_prices_list[N+i]))
                horizontal_block_list.append(np.diagflat(export_price_diagonal))
            aggregate_export_price_matrix = np.hstack(horizontal_block_list)
            zero_like = np.zeros_like(aggregate_export_price_matrix)
            aggregate_zero_matrix = np.tile(zero_like, (planning_horizon+1,1))
            export_constraint_matrix = np.vstack([aggregate_zero_matrix, aggregate_export_price_matrix])

            #Constructing the directly imported target output constraint vector (p_imp^T r_imp)
            vertical_block_list = []
            for i in range(planning_horizon+1):
                vertical_block_list.append(np.matmul(import_prices_list[N+i], imported_target_output_list[N+i]))
            imported_target_output_constraint_vector = np.sum(vertical_block_list, axis=0)

            #Constructing the export augmentet constraint objects
            aggregate_constraint_matrix = deepcopy(np.vstack([aggregate_constraint_matrix, use_imports_constraint_matrix]))
            aggregate_constraint_matrix = deepcopy(np.hstack([aggregate_constraint_matrix, export_constraint_matrix]))
            aggregate_constraint_vector = deepcopy(np.vstack([aggregate_constraint_vector, imported_target_output_constraint_vector]))
            #TODO fix one vector shape issue
            ones_length = aggregate_primary_resource_vector.shape[0]
            one_vector = deepcopy(np.matrix(np.ones((ones_length, 1))))
            aggregate_primary_resource_vector = deepcopy(np.vstack([aggregate_primary_resource_vector, one_vector]))

        # Plan
        result = optimize.linprog(c=aggregate_primary_resource_vector, 
                                  A_ub=-aggregate_constraint_matrix, 
                                  b_ub=-aggregate_constraint_vector, options = {'maxiter': max_iterations, 'tol': tolerance},
                                  bounds=(0, upper_bound_on_activity), method='highs')
        print(result.success)
        print(result.status)
        lagrange_ineq = -optimize.linprog(c=aggregate_primary_resource_vector, 
                                          A_ub=-aggregate_constraint_matrix, 
                                          b_ub=-aggregate_constraint_vector, 
                                          bounds=(0, upper_bound_on_activity), options = {'maxiter': 1000, 'tol': 1e-8}, 
                                          method='highs')['ineqlin']['marginals']

        result_list.append(result.x)
        lagrange_list.append(lagrange_ineq)
        slack_list.append(result.slack)

        #Production carry into the next time step
        if export_constraint_boolean:
            recent_slack = deepcopy(np.array_split(slack_list[N], (planning_horizon+1)*2))
            modifier_value = deepcopy(np.matmul(depreciation_list[N], recent_slack[0]))
        else:
            recent_slack = deepcopy(np.array_split(slack_list[N], planning_horizon+1))
            modifier_value = deepcopy(np.matmul(depreciation_list[N], recent_slack[0]).reshape([-1,1]))

    return([result_list, lagrange_list, slack_list, aggregate_primary_resource_vector, aggregate_constraint_matrix, aggregate_constraint_vector])