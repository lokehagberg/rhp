import numpy as np
from copy import deepcopy
from scipy import optimize

#See the rhp_intro.pdf for an explanation of the comments

def zdivide(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=y!=0)


def plan(time_steps, planning_horizon, primary_resource_list, augmented_supply_list, augmented_use_domestic_list, 
         augmented_use_imported_list, depreciation_matrix_list, augmented_target_output_list, augmented_export_vector_list, 
         export_prices_list, import_prices_list):
    
    result_list, lagrange_list,target_output_aggregated_list = [], [], []

    steps_horizon = time_steps + planning_horizon

    # Final production matrices DJ
    final_production_matrix_list = []
    for T in range(steps_horizon):
        final_production_matrix_list.append(np.matmul(depreciation_matrix_list[T + 1], (
                    augmented_supply_list[T] - (augmented_use_domestic_list[T] + augmented_use_imported_list[T]))))
    
    for T in range(time_steps):

        # Export constraints

        import_cost_matrix = deepcopy(augmented_use_imported_list[T])
        for i in range(augmented_use_imported_list[T].shape[0]):
            for j in range(augmented_use_imported_list[T].shape[1]):
                import_cost_matrix[i, j] = augmented_use_imported_list[T][i, j] * import_prices_list[T][i][0]

        import_cost_list = deepcopy([import_cost_matrix])
        for i in range(planning_horizon - 1):
            cost_list = deepcopy(np.concatenate((import_cost_list[0], import_cost_matrix), axis=1))
            import_cost_list[0] = deepcopy(cost_list)

        augmented_import_cost_matrix = import_cost_list[0].sum(axis=0)

        export_value_list = []
        for i in range(time_steps):
            exp_val = 0
            for j in range(len(export_prices_list[i])):
                exp_val += export_prices_list[i][j] * augmented_export_vector_list[i][j]
            export_value_list.append(exp_val)

        # Constructing a list of DJ

        def concatenator_3(arr, T, planning_horizon):
            zero_matrix = np.matrix(np.zeros_like(np.asarray(arr[0])))
            block_row_list = []

            for i in range(planning_horizon):
                bottom_triangle_list = deepcopy([arr[T]])
                top_triangle_list = deepcopy([zero_matrix])
                for j in range(i):
                    bottom = deepcopy(np.concatenate((bottom_triangle_list[0], arr[T + j]), axis=1))
                    bottom_triangle_list[0] = deepcopy(bottom)
                for j in range(planning_horizon - (i + 2)):
                    triangle = deepcopy(np.concatenate((top_triangle_list[0], zero_matrix), axis=1))
                    top_triangle_list[0] = deepcopy(triangle)
                if i != planning_horizon - 1:
                    block_row_list.append(np.concatenate((bottom_triangle_list[0], top_triangle_list[0]), axis=1))
                else:
                    block_row_list.append(bottom_triangle_list[0])

            if (len(block_row_list) > 1):
                for i in range(len(block_row_list) - 1):
                    result = deepcopy(np.concatenate((block_row_list[i], block_row_list[i + 1]), axis=0))
                    block_row_list[i + 1] = deepcopy(result)
                return (result)
            else:
                return (block_row_list[0])

        production_aggregated_primitive = concatenator_3(arr=final_production_matrix_list, T=T,
                                                         planning_horizon=planning_horizon)

        production_aggregated = np.concatenate((production_aggregated_primitive, -augmented_import_cost_matrix), axis=0)

        # Constructing a list of Dr

        v = deepcopy(np.matmul(depreciation_matrix_list[T + 1], augmented_target_output_list[T]))
        for i in range(planning_horizon - 1):
            w = deepcopy(np.concatenate((v, (
                np.asarray(np.matmul(depreciation_matrix_list[i + 2], augmented_target_output_list[i + 1]) + v[i])))))
            v = deepcopy(w)

        target_output_aggregated = np.concatenate((v, [-export_value_list[T]]))

        # Plan

        result = optimize.linprog(c=primary_resource_list[T], A_ub=-production_aggregated, b_ub=-target_output_aggregated,
                                  bounds=(0, None), method='highs-ipm')
        print(result.success)
        print(result.status)
        lagrange_ineq = - \
        optimize.linprog(primary_resource_list[T], A_ub=-production_aggregated, b_ub=-target_output_aggregated, bounds=(0, None),
                         method='highs-ipm')['ineqlin']['marginals']

        result_list.append(result.x)
        lagrange_list.append(lagrange_ineq)
        
        target_output_aggregated_list.append(target_output_aggregated[:-1])

    return([result_list, lagrange_list])