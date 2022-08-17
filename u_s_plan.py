import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from copy import deepcopy
from scipy import optimize


def zdivide(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=y!=0)


def plan(time_steps, planning_horizon, augmented_supply_list, augmented_use_domestic_list, augmented_use_imported_list,
         depreciation_matrix_list, augmented_target_output_list, augmented_export_vector_list, export_prices_list,
         import_prices_list, start_stock, sector_name, sector_with_all_outputs, worked_hours):
    result_list = []
    lagrange_list = []
    target_output_aggregated_list = []

    steps_horizon = time_steps + planning_horizon

    # final production matrices D(B - (A' + A''))
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

        # Constructing M

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

        # Constructing the one_vector
        aug_one_vector = np.array([[1] for i in range(production_aggregated.shape[1])])

        # Constructing v

        v = deepcopy(np.matmul(depreciation_matrix_list[T + 1], augmented_target_output_list[T]) - np.matmul(
            depreciation_matrix_list[T], start_stock))
        for i in range(planning_horizon - 1):
            w = deepcopy(np.concatenate((v, (
                np.asarray(np.matmul(depreciation_matrix_list[i + 2], augmented_target_output_list[i + 1]) + v[i])))))
            v = deepcopy(w)

        target_output_aggregated = np.concatenate((v, [-export_value_list[T]]))

        # plan

        result = optimize.linprog(aug_one_vector, A_ub=-production_aggregated, b_ub=-target_output_aggregated,
                                  bounds=(0, None), method='highs-ipm')
        print(result.success)
        print(result.status)
        lagrange_ineq = - \
        optimize.linprog(aug_one_vector, A_ub=-production_aggregated, b_ub=-target_output_aggregated, bounds=(0, None),
                         method='highs-ipm')['ineqlin']['marginals']

        result_list.append(result.x)
        lagrange_list.append(lagrange_ineq)

        x = deepcopy(np.array_split(result_list[T], planning_horizon))
        start_stock = deepcopy(np.matrix(np.matmul(final_production_matrix_list[T], x[0])).reshape([-1, 1]))

        target_output_aggregated_list.append(target_output_aggregated[:-1])

    # plan details
    overshoot = []
    for T in range(time_steps):

        x = np.array_split(result_list[T], planning_horizon)

        l = np.array_split(lagrange_list[T], planning_horizon)

        y = []
        for i in range(planning_horizon):
            y.append(np.transpose(np.squeeze(np.array(np.matmul(final_production_matrix_list[T + i], x[i])))))

        tout_planning_period = np.array_split(target_output_aggregated_list[T], planning_horizon)
        for i in range(planning_horizon):
            overshoot.append((zdivide(y[i], np.squeeze(np.asarray(tout_planning_period[i])))) - 1)

    # Displaying results of all plans in order

    plt.style.use('_mpl-gallery')
    fig = plt.figure(figsize=(30, 180))
    fig.suptitle('Results', fontsize=32)
    gs = gridspec.GridSpec(5 * time_steps * planning_horizon, 1)

    sector_with_all_outputs_and_EXP = deepcopy(sector_with_all_outputs)
    sector_with_all_outputs_and_EXP.append('EXP')

    labels = ['overshoot_target_output_quotient',
              'Worked hours (10K)',
              'Worked hours (10K) percentage',
              'Produced total period minus consumed total',
              'Lagrange multiplier']

    for i in range(time_steps):
        ax = fig.add_subplot(gs[i * 5 + 1, 0])
        ax.set_xlabel('Product', fontsize=14)
        ax.set_ylabel(labels[0], fontsize=14)
        ax.set_xticks(range(final_production_matrix_list[i].shape[0]), sector_with_all_outputs)
        for j in range(planning_horizon):
            ax.plot(range(final_production_matrix_list[j].shape[0]), overshoot[j])

        ax = fig.add_subplot(gs[i * 5 + 2, 0])
        ax.set_xlabel('Product', fontsize=14)
        ax.set_ylabel(labels[1], fontsize=14)
        ax.set_xticks(range(final_production_matrix_list[i].shape[1]), sector_name)
        for j in range(planning_horizon):
            x = np.array_split(result_list[i], planning_horizon)
            ax.plot(range(x[j].shape[0]), x[j])

        ax = fig.add_subplot(gs[i * 5 + 3, 0])
        ax.set_xlabel('Product', fontsize=14)
        ax.set_ylabel(labels[2], fontsize=14)
        ax.set_xticks(range(final_production_matrix_list[i].shape[1]), sector_name)
        for j in range(planning_horizon):
            x = np.array_split(result_list[i], planning_horizon)
            ax.plot(range(x[j].shape[0]), x[j] / worked_hours)

        ax = fig.add_subplot(gs[i * 5 + 4, 0])
        ax.set_xlabel('Product', fontsize=14)
        ax.set_ylabel(labels[3], fontsize=14)
        ax.set_xticks(range(len(sector_with_all_outputs)), sector_with_all_outputs)
        total_overshoot = 0.0
        for j in range(planning_horizon):
            total_overshoot += overshoot[j]
        ax.plot(range(final_production_matrix_list[i].shape[0]), total_overshoot)

        ax = fig.add_subplot(gs[i * 5 + 5, 0])
        ax.set_xlabel('Product', fontsize=14)
        ax.set_ylabel(labels[4], fontsize=14)
        ax.set_xticks(range(len(sector_with_all_outputs_and_EXP)), sector_with_all_outputs_and_EXP)
        for j in range(planning_horizon):
            l = np.array_split(lagrange_list[i], planning_horizon)
            ax.plot(range(l[j].shape[0]), l[j])

    plt.show()

    return(dict(start_stock=start_stock, result_list=result_list, lagrange_list=lagrange_list))