import mlflow 
import numpy as np
import pandas as pd
import os

# this script copies all patches to a folder, and creates a csv and latex table sorted by a metric


# enter experiment id for each network and defense
exp_id_dict = {
    'FlowNetC_none': ['1'],
    'FlowNetC_lgs': ['2'],
    'FlowNetC_ilp': ['3'],
    'FlowNetCRobust_none': ['4'],
    'FlowNetCRobust_lgs': ['5'],
    'FlowNetCRobust_ilp': ['6'],
    'PWCNet_none': ['7'],
    'PWCNet_lgs': ['8'],
    'PWCNet_ilp': ['9'],
    'SpyNet_none': ['10'],
    'SpyNet_lgs': ['11'],
    'SpyNet_ilp': ['12'],
    'RAFT_none': ['13'],
    'RAFT_lgs': ['14'],
    'RAFT_ilp': ['15'],
    'GMA_none': ['16'],
    'GMA_lgs': ['17'],
    'GMA_ilp': ['18'],
    'FlowFormer_none': ['19'],
    'FlowFormer_lgs': ['20'],
    'FlowFormer_ilp': ['21'],
}

def create_table(net, defense, metric='aee_avg_def-advdef', descend=True, 
                 parameters=['optimizer','lr','alpha', 'change_of_variables', 'attack_target', 'loss'], 
                 metrics_of_interest=['aee_avg_def-advdef', 'aee_avg_gt-undef', 'aee_avg_gt-def'],
                 n_runs=1000, folder_path=f'{os.environ["PATCHATTACK_HOME"]}/results/'):
    """
        net: name of network
        defense: name of defense
        metric: metric to sort by
        descend: True: lower is better, False: higher is better
        parameters: parameters to differentiate by
        metrics_of_interest: metrics of interest. These are the metrics that are printed in the table
        n_runs: upper bound on number of runs to get
        folder_path: path to folder to save results in
    """
    
    exp_ids = exp_id_dict[f'{net}_{defense}']
    path_name=f'{net}_{defense}_evals' # name of folder to save results in
    folder_path = os.path.join(folder_path, path_name)

    os.makedirs(f'{folder_path}/patches', exist_ok=True)
    os.makedirs(f'{folder_path}/best-patches', exist_ok=True)


    print(f"Experiment(s): {exp_ids}")
    print(f"Metric: {metric}")
    print(f"Parameters: {parameters}")

    # get all runs
    runs = mlflow.search_runs(experiment_ids=exp_ids)
    # filter out runs that are not successful
    runs = runs[runs['status'] == 'FINISHED']
    # filter out runs where metrics.loss is not nan
    runs = runs[np.isnan(runs['metrics.loss']) == False]

    if len(runs) == 0:
        print('No runs found')
        exit(0)
    else:
        print(f"Got all {len(runs)} runs")

    # sort by given metric
    runs = runs.sort_values(by=f'metrics.{metric}', ascending=not descend)

    # empty pandas dataframe from runs
    df = pd.DataFrame(columns=parameters+metrics_of_interest+['n_runs'])

    marked = set() # set of runs that have already been added to the dataframe (because we train 4 patches per parameter combination)

    # print parameters of best runs and their metric (in latex format)
    # find multiple runs with same parameters and extract mean of metric. Put mean in dataframe
    for j, (i, run) in enumerate(runs.iterrows()):
        # copy the best patch from a config to a different folder
        print(f"Run {j}: run_id={run['run_id']}, experiment_id={run['experiment_id']}, {metric}={run[f'metrics.{metric}']:.4f}", end='')
        print(f", parameters={', '.join([str(run[f'params.{param}']) for param in parameters])}")
        
        artifact_folder = mlflow.artifacts.download_artifacts(run.artifact_uri) # load artifact folder
        patch_name = os.path.join(artifact_folder,f'{int(run["params.n"])-1:05d}_Patch.png') # add the last patch to the path
        print(f"Best patch: {patch_name}")
        # copy patch to folder
        os.system(f"cp {patch_name} {folder_path}/patches/{j:04}_{'_'.join([run[f'params.{param}'] for param in parameters])}.png")
        
        if i in marked:
            continue
        
        # find mean of metric for all runs with same parameters
        mean = [0]*len(metrics_of_interest)
        counter = 0
        for k, otherrun in runs.iterrows():
            if all([run[f'params.{param}'] == otherrun[f'params.{param}'] for param in parameters if param != 'seed']):
                counter += 1
                marked.add(k)
                for k, metric_of_interest in enumerate(metrics_of_interest):
                    mean[k] += otherrun[f'metrics.{metric_of_interest}']
        mean = np.array(mean)/counter

        # add mean values to dataframe 
        new_row = {param: run[f'params.{param}'] for param in parameters}
        new_row.update({metric_of_interest: mean[k] for k, metric_of_interest in enumerate(metrics_of_interest)})
        new_row.update({'n_runs': counter})
        df = pd.concat([df, pd.DataFrame(new_row, index=[j])], ignore_index=True)
        
        # print mean
        # for k, metric_of_interest in enumerate(metrics_of_interest):
        #     print(f"{mean[k]:.4}", end=' & ' if k != len(metrics_of_interest)-1 else ' \\\\')
        # print()
        # stop after n_runs
        if j == n_runs-1:
            break
    print()

    # sort dataframe by given metric
    df.sort_values(by=metric, ascending=not descend, inplace=True)
    
    # save all patches from the best parameter combination
    for k, (i, run) in enumerate(runs.iterrows()):
        if all([run[f'params.{param}'] == df.iloc[0][param] for param in parameters if param != 'seed']):
            artifact_folder = mlflow.artifacts.download_artifacts(run.artifact_uri)
            patch_name = os.path.join(artifact_folder,f'{int(run["params.n"])-1:05d}_Patch.png')
            print(f"Best patch: {patch_name}")
            os.system(f"cp {patch_name} {folder_path}/best-patches/{k:04}_{'_'.join([run[f'params.{param}'] for param in parameters])}.png")

    # optimizer: opt, lr: lr, alpha: alpha, change_of_variables: cov, attack_target: target, loss: loss
    df.rename(columns={'optimizer': 'opt', 'lr': 'lr', 'alpha': 'alpha', 'change_of_variables': 'cov', 'attack_target': 'target', 'loss': 'loss'}, inplace=True)

    df.to_csv(f'{folder_path}/{path_name}.csv')
    print(f"Saved to {folder_path}/{path_name}.csv")

    # replace aee_avg_def-advdef with Defended Robustness and so on
    df.rename(columns={'aee_avg_def-advdef': 'Defended Robustness', 'aee_avg_undef-advdef': 'Undefended Robustness', 'aee_avg_adv-advdef': 'Effect of defense', 'aee_avg_gt-undef':'Accuracy', 'aee_avg_gt-def':'Defended Accuracy'}, inplace=True)
    # add an up or down arrow to the metric column header
    metric = {'aee_avg_def-advdef': 'Defended Robustness', 'aee_avg_undef-advdef': 'Undefended Robustness', 'aee_avg_adv-advdef': 'Effect of defense', 'aee_avg_gt-undef': 'Accuracy', 'aee_avg_gt-def': 'Defended Accuracy'}[metric]
    df.rename(columns={metric: rf'{metric} ↓' if descend else rf'{metric} ↑'}, inplace=True)


    # print df in latex format
    df.to_latex(buf=f'{folder_path}/{path_name}.tex', index=False, header=True, float_format='%.2f', multirow=True)
    print(f"Saved to {folder_path}/{path_name}.tex")


if __name__ == "__main__":
    
    #net = 'RAFT' #! Change: net to get best runs for
    #defense = 'lgs' #! Change: defense to get best runs for
    
    for net in ['FlowNetC','FlowNetCRobust','PWCNet','SpyNet','RAFT','GMA','FlowFormer']:
        for defense in ['none','lgs','ilp']:
            create_table(net, defense)
