import numpy as np
from copy import deepcopy
from scipy import optimize

#See the rhp_intro.pdf for an explanation of the comments

#Horizontal stack algorithm
def stack_horizontal(x, y, z):
    w = deepcopy(x[z])
    for i in range(y):
        w = deepcopy(np.hstack((w, x[z + i + 1])))
        
    return(w)

#Vertical stack algorithm
def stack_vertical(x, y, z):
    w = deepcopy(x[z])
    for i in range(y):
        w = deepcopy(np.vstack((w, x[z + i + 1])))
        
    return(w)

#Supply-use aggregating algorithm
def concatenator(depreciation_list, supply_use_list, planning_horizon, T):
    DJ_list = []
    zero_list = [] 
    for i in range(planning_horizon): 
        zero_list.append(np.matrix(np.zeros_like(np.asarray(supply_use_list[T+i]))))
        
    for i in range(planning_horizon):
        DJ_list.append(supply_use_list[T+i])
        row = deepcopy(stack_horizontal(DJ_list, i, 0))
        
        if planning_horizon - i - 1 > 0:
            zero_row = deepcopy(stack_horizontal(zero_list, planning_horizon - i - 2, 0))
            full_row = deepcopy(np.hstack((row, zero_row)))
            
        else:
            full_row = deepcopy(row)
            
        if i == 0: 
            DJ_aggregated = deepcopy(full_row)
            
        else:   
            for j in range(len(DJ_list)):
                DJ_list[j] = deepcopy(np.matmul(depreciation_list[T + 1 + i], DJ_list[j]))
            DJ_aggregated = deepcopy(np.vstack((DJ_aggregated, full_row)))
            
    return(DJ_aggregated)

#Plan function
def plan(time_steps, planning_horizon, primary_resource_list, supply_use_list, use_imported_list, depreciation_list, 
         domestic_target_output_list, imported_target_output_list, export_constraint, export_vector_list, export_prices_list, import_prices_list):
    
    result_list, lagrange_list, slack_list = [], [], []

    for T in range(time_steps):

        # Export constraints
        import_cost_matrix = deepcopy(use_imported_list[T])
        for i in range(use_imported_list[T].shape[0]):
            for j in range(use_imported_list[T].shape[1]):
                import_cost_matrix[i, j] = use_imported_list[T][i, j] * import_prices_list[T][i, 0]
                
        import_cost_list = deepcopy([import_cost_matrix])
        
        for i in range(planning_horizon - 1):
            cost_list = deepcopy(np.concatenate((import_cost_list[0], import_cost_matrix), axis=1))
            import_cost_list[0] = deepcopy(cost_list)
            
        augmented_import_cost_matrix = import_cost_list[0].sum(axis=0)
        export_value_list = []
        for i in range(time_steps):
            export_value_list.append(np.dot(export_prices_list[i].reshape([1,-1]), export_vector_list[i]))

        # Constructing DJ aggregated
        non_exp_DJ_aggregated = concatenator(depreciation_list, supply_use_list, planning_horizon, T)
        
        DJ_aggregated = np.vstack((concatenator(depreciation_list, supply_use_list, planning_horizon, T), 
                                   -augmented_import_cost_matrix))
        
        # Constructing Dr aggregated
        depreciated_target_output_list = [domestic_target_output_list[T]]
        for i in range(planning_horizon - 1):
            depreciated_target_output_list.append(
                domestic_target_output_list[T + i + 1] + 
                np.matmul(depreciation_list[T + i + 2], depreciated_target_output_list[i]))
        
        non_exp_Dr_aggregated = stack_vertical(depreciated_target_output_list, planning_horizon - 1, 0)
        
        Dr_aggregated = np.concatenate((stack_vertical(depreciated_target_output_list, planning_horizon - 1, 0), 
                                        np.dot(imported_target_output_list[T],import_prices_list[T])-export_value_list[T]))

        # Constructing c aggregated
        c_aggregated = stack_vertical(primary_resource_list, planning_horizon - 1, T)
        
        if export_constraint:
            DJ_matrix = deepcopy(DJ_aggregated)
            Dr_vector = deepcopy(Dr_aggregated)
        else:
            DJ_matrix = deepcopy(non_exp_DJ_aggregated)
            Dr_vector = deepcopy(non_exp_Dr_aggregated)

        # Plan
        result = optimize.linprog(c=c_aggregated, A_ub=-DJ_matrix, b_ub=-Dr_vector,
                                  bounds=(0, None), method='highs-ipm')
        print(result.success)
        print(result.status)
        lagrange_ineq = -optimize.linprog(c=c_aggregated, A_ub=-DJ_matrix, b_ub=-Dr_vector, bounds=(0, None),
                                          method='highs-ipm')['ineqlin']['marginals']

        result_list.append(result.x)
        lagrange_list.append(lagrange_ineq)
        slack_list.append(result.slack)

    return([result_list, lagrange_list, slack_list])