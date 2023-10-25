import mlflow
import pandas as pd
import numpy as np
import os

# this script creates a table:
#            | FlowNetC | RAFT | GMA | SpyNet | PWC-Net | Defended FlowNet
#       -----|----------|------|-----|--------|---------|------------------
# Accuracy   | ...     | ...  | ... | ...    | ...     | ...
# Defended Accuracy LGS | ...     | ...  | ... | ...    | ...     | ...
# Defended Accuracy ILP | ...     | ...  | ... | ...    | ...     | ...
#   ----------------|----------|------|-----|--------|---------|------------------
# R^N_N | ...     | ...  | ... | ...    | ...     | ... # where R^N_N is Unmodified Attack, No Defense
# R^N_LGS| ...     | ...  | ... | ...    | ...     | ... # where R^N_LGS is Unmodified Attack, LGS Defense # robustness with lgs defense
# R^N_ILP| ...     | ...  | ... | ...    | ...     | ... # where R^N_ILP is Unmodified Attack, ILP Defense # robustness with ilp defense
# ----------------|----------|------|-----|--------|---------|------------------
# R^LGS_LGS | ...     | ...  | ... | ...    | ...     | ... # where R^LGS_LGS is Patch trained with LGS Defense, LGS Defense
# R^ILP_ILP | ...     | ...  | ... | ...    | ...     | ... # where R^ILP_ILP is Patch trained with ILP Defense, ILP Defense
#  ----------------|----------|------|-----|--------|---------|------------------
# R^LGS_N | ...     | ...  | ... | ...    | ...     | ... # where R^LGS_N is Patch trained with LGS Defense, No Defense
# R^ILP_N | ...     | ...  | ... | ...    | ...     | ... # where R^ILP_N is Patch trained with ILP Defense, No Defense
#  ----------------|----------|------|-----|--------|---------|------------------
# R^Manual_N | ...     | ...  | ... | ...    | ...     | ... # where R^Manual_N is evaluated with the manual patch, No Defense
# R^Manual_LGS | ...     | ...  | ... | ...    | ...     | ... # where R^Manual_LGS is evaluated with the manual patch, LGS Defense
# R^Manual_ILP | ...     | ...  | ... | ...    | ...     | ... # where R^Manual_ILP is evaluated with the manual patch, ILP Defense

# the best runs are saved in {folder}/<net>_<defense>_evals/<net>_<defense>_evals.xlsx
# there, the first row is the best run. 

nets = ['FlowNetC', 'FlowNetCRobust','PWCNet','SpyNet', 'RAFT', 'GMA', 'FlowFormer']
defenses = ['None', 'LGS',  'ILP']
folder = 'results'

mlrun_dir = 'mlruns'
mlflow.set_tracking_uri(mlrun_dir)

result_table = pd.DataFrame(columns=nets)

## functions
def get_from_results_table(net, defense, metric):
    runs = pd.read_csv(f'{folder}/{net}_{defense}_evals/{net}_{defense}_evals.csv')
    # extract the best run (first row)
    return runs.iloc[0][metric]
    
    

## Accuracies
# This is currently from the results table, this table averages over multiple runs.
for net in nets:
    metric = 'aee_avg_gt-def' # for none, the defended is just the unmodiefied image
    try:
        acc = get_from_results_table(net, 'none', metric)
        result_table.loc['Accuracy', net] = acc
    except FileNotFoundError:
        print(f'No results for {net} with no defense')
    try:
        def_acc_lgs = get_from_results_table(net, 'lgs', metric)
        result_table.loc['Defended Accuracy LGS', net] = def_acc_lgs
    except FileNotFoundError:
        print(f'No results for {net} with lgs defense')
    try:
        def_acc_ilp = get_from_results_table(net, 'ilp', metric)
        result_table.loc['Defended Accuracy ILP', net] = def_acc_ilp
    except FileNotFoundError:
        print(f'No results for {net} with ilp defense')
print(result_table)

## Robustness
# read this from mlflow runs. For every net, patch, defense combination there is '{net}_PatchAttack-with-defense_cd_u_{net}_{patch_defense}_{defense}_eval'
for net in nets:
    for defense in defenses:
        for patch_defense in defenses:
            run_name = f'{net}_PatchAttack-with-defense_cd_u_eval_{net}_{patch_defense.lower()}_{defense.lower()}_eval'
            # run_name should be contained in tags.mlflow.runName
            # actual run_name eg: '2023-02-09_13:16:05:610772_SpyNet_PatchAttack-with-defense_cd_u_eval_SpyNet_none_none_eval'
            runs = mlflow.search_runs(experiment_names=[run_name])
            # if there are no runs, continue
            if len(runs) == 0:
                continue
            # filter finished runs
            runs = runs[runs['status'] == 'FINISHED']
            # get the mean of the robustness
            robustness = runs['metrics.aee_avg_def-advdef'].mean()
            
            patch_defense = patch_defense if patch_defense != 'None' else 'Std'
            result_table.loc[f'$R^{patch_defense}_{defense}$', net] = robustness
print(result_table)

# Manual patch
for net in nets:
    for defense in defenses:
        run_name = f'{net}_PatchAttack-with-defense_cd_u_eval_Manual_{net}_none_{defense.lower()}_eval'
        # run_name should be contained in tags.mlflow.runName
        runs = mlflow.search_runs(experiment_names=[run_name])
        # if there are no runs, continue
        if len(runs) == 0:
            continue
        # filter finished runs
        runs = runs[runs['status'] == 'FINISHED']
        # get the mean of the robustness
        robustness = runs['metrics.aee_avg_def-advdef'].mean()
        result_table.loc[f'$R^Manual_{defense}$', net] = robustness

# save the table
result_table.to_csv(f'{folder}/result_table.csv')
# save to latex with 2 decimal places
result_table = result_table.applymap(lambda x: f'{x:.2f}')
result_table.to_latex(f'{folder}/result_table.tex')


# Now the same for aee_avg_undef-advdef
result_table = pd.DataFrame(columns=nets)
for net in nets:
    for defense in defenses:
        for patch_defense in defenses:
            run_name = f'{net}_PatchAttack-with-defense_cd_u_eval_{net}_{patch_defense.lower()}_{defense.lower()}_eval'
            # run_name should be contained in tags.mlflow.runName
            # actual run_name eg: '2023-02-09_13:16:05:610772_SpyNet_PatchAttack-with-defense_cd_u_eval_SpyNet_none_none_eval'
            runs = mlflow.search_runs(experiment_names=[run_name])
            # if there are no runs, continue
            if len(runs) == 0:
                continue
            # filter finished runs
            runs = runs[runs['status'] == 'FINISHED']
            # get the mean of the robustness
            robustness = runs['metrics.aee_avg_undef-advdef'].mean()
            
            patch_defense = patch_defense if patch_defense != 'None' else 'Std'
            result_table.loc[f'$R^{patch_defense}_{defense}$', net] = robustness
print(result_table)

# Manual patch
for net in nets:
    for defense in defenses:
        run_name = f'{net}_PatchAttack-with-defense_cd_u_eval_Manual_{net}_none_{defense.lower()}_eval'
        # run_name should be contained in tags.mlflow.runName
        runs = mlflow.search_runs(experiment_names=[run_name])
        # if there are no runs, continue
        if len(runs) == 0:
            continue
        # filter finished runs
        runs = runs[runs['status'] == 'FINISHED']
        # get the mean of the robustness
        robustness = runs['metrics.aee_avg_undef-advdef'].mean()
        result_table.loc[f'$R^Manual_{defense}$', net] = robustness

# save the table
result_table.to_csv(f'{folder}/result_table_undef.csv')
# save to latex with 2 decimal places
result_table = result_table.applymap(lambda x: f'{x:.2f}')
result_table.to_latex(f'{folder}/result_table_undef.tex')



# Now for comparing distances to ground truth between defended and undefended
result_table = pd.DataFrame(columns=nets) # each defense one row
for net in nets:    
    for defense in defenses:
        run_name = f'{net}_PatchAttack-with-defense_cd_u_eval_{net}_{defense.lower()}_{defense.lower()}_eval'
        runs = mlflow.search_runs(experiment_names=[run_name])
        # if there are no runs, continue
        if len(runs) == 0:
            continue
        # filter finished runs
        runs = runs[runs['status'] == 'FINISHED']
        # get the mean distance to ground truth
        aee_gt = runs['metrics.aee_avg_gt-advdef'].mean()
        # put in the table
        result_table.loc[defense, net] = aee_gt

        if defense == 'None':
            continue
        aee_gt = runs['metrics.aee_avg_gt-def'].mean()
        result_table.loc[f'Just Defended {defense}', net] = aee_gt
        
        # also do a baseline without attacking aka metrics.aee_avg_gt-unDef and metrics.aee_avg_gt-Def
        try:
            aee_gt = runs['metrics.aee_avg_gt-undef'].mean()
            result_table.loc['Unmodified', net] = aee_gt
        except KeyError:
            pass
    
            
print(result_table)

# Manual patch
for net in nets:
    for defense in defenses:
        run_name = f'{net}_PatchAttack-with-defense_cd_u_eval_Manual_{net}_none_{defense.lower()}_eval'
        # run_name should be contained in tags.mlflow.runName
        runs = mlflow.search_runs(experiment_names=[run_name])
        # if there are no runs, continue
        if len(runs) == 0:
            continue
        # filter finished runs
        runs = runs[runs['status'] == 'FINISHED']
        # get the mean distance to ground truth
        aee_gt = runs['metrics.aee_avg_gt-advdef'].mean()
        # put in the table
        result_table.loc[f'Manual {defense}', net] = aee_gt

# save the table
result_table.to_csv(f'{folder}/result_table_aee_gt.csv')

result_table = result_table.applymap(lambda x: f'{x:.2f}')
result_table.to_latex(f'{folder}/result_table_aee_gt.tex')


# get results for manual patch
result_table = pd.DataFrame(columns=nets) # each defense one row
for net in nets:
    for defense in defenses:
        # like: RAFT_PatchAttack-with-defense_cd_u_eval_Manual_RAFT_none_none_eval
        run_name = f'{net}_PatchAttack-with-defense_cd_u_eval_Manual_{net}_{defense.lower()}_{defense.lower()}_eval'
        runs = mlflow.search_runs(experiment_names=[run_name])
        # if there are no runs, continue
        if len(runs) == 0:
            continue
        # filter finished runs
        runs = runs[runs['status'] == 'FINISHED']
        # get robustness
        robustness = runs['metrics.aee_avg_def-advdef'].mean()
        result_table.loc[defense, net] = robustness
        
print(result_table)

# save the table
result_table.to_csv(f'{folder}/result_table_manual.csv')
result_table = result_table.applymap(lambda x: f'{x:.2f}')
result_table.to_latex(f'{folder}/result_table_manual.tex')