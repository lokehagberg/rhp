# Receding Horizon Planning for Economic Coordination

Python-implementation of receding horizon planning technique for macroeconomic coordination of production. Code is readily used together with real economic data in the form of supply and use tables.

## Background

In the era of climate change, viable societies will have to rapidly adapt to extreme weather events as well as coordinate economic activities in an efficient manner towards long-term goals. This will require a careful design and experimentation of *coordination protocols* and *optimization procedures* for large-scale economic systems.

The code in this repository implements a type of optimization procedure which we call *receding horizon planning* (RHP). It builds on ideas from optimization theory, economic analysis and control theory (cf. the works of Kantorovich, von Neumann, Bellman and Bertsekas). An introductory technical description is [provided here](rhp_intro.pdf).

A first example with Swedish national accounts data is given [here](example.ipynb).

## Python implementation

The goal is to coordinate $m$ units of production, so that their production levels utilize the minimal amount of resources required to meet material constraints over a future time horizon. This coordination process is revised at every time step. 

### Variables

| TeX paper | Code                         | Description                                           | Type & length                                  |
| --------- | ---------------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| $n$       | time_steps                   | simulated time steps                                  | non-negative integer & 1                       |
| $N$       | planning_horizon             | planning horizon                                      | non-negative integer & 1                       |
| $B$       | augmented_supply_list        | supply list in price unit per time unit               | list, each item in the list numpy.matrix & n*m |
| $A$       | augmented_use_domestic_list  | use domestic list in product price unit per time unit | list, each item in the list numpy.matrix & n*m |
| $T$       | augmented_use_imported_list  | use imported list in product price unit per time unit | list, each item in the list numpy.matrix & n*m |
| $D$       | depreciation_matrix_list     | depreciation matrix list                              | list, each item in the list numpy.matrix & n*n |
| $r$       | augmented_target_output_list | target output vector in product price unit            | list, each item in the list numpy.matrix & n*1 |
| $r_{exp}$ | augmented_export_vector_list | export vector in product price unit                   | list, each item in the list numpy.matrix & m*1 |
| $p_{exp}$| export_prices_list           | export prices list                                    | list, each item in the list numpy.matrix & m*1 |
| $p_{imp}$| import_prices_list           | import prices list                                    | list, each item in the list numpy.matrix & m*1 |
|           | sector_name                  | ingoing product names                                 | list, each item in the list string & m         |
|           | sector_with_all_outputs      | outgoing product names                                | list, each item in the list string & n         |
|           | worked_hours                 | actual worked hours for the period                    | numpy.array & m                                |



