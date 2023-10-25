
import mlflow
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
params = {
    # increase font size
    'font.size'   : 16,
    'font.weight' : 'normal',
    'font.family' : 'Times New Roman',
    'text.usetex' : True,    
    # increase marker size
    'lines.markersize' : 12,
    # 'lines.markeredgewidth' : 1,
    
    }
plt.rcParams.update(params)

# Run this script with the following commands:
# python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_k_o_lgs_eval --variables k,o --defense lgs
# python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_k_o_ilp_eval --variables k,o --defense ilp
# python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_t_s_lgs_eval --variables t,s --defense lgs
# python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_t_s_ilp_eval --variables t,s --defense ilp
# python training_scripts/eval_defense_params.py --mlruns_dir mlruns --experiment_name FlowNetC_PatchAttack-with-defense_cd_u_defense_parameter_r_ilp_eval --variables r --defense ilp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlruns_dir", type=str, default="mlruns")
    parser.add_argument("--experiment_name", type=str, default='')
    parser.add_argument("--experiment_ids", nargs='*', type=str, default=[])
    parser.add_argument("--variables", type=str, default='r') # k,o
    parser.add_argument("--defense", type=str, default="lgs", help="Additional filter for defense name. lgs, ilp, or None")
    args = parser.parse_args()
    savefolder = 'training_scripts/defense_params'
    os.makedirs(savefolder, exist_ok=True)
    if len(args.variables)==3:
        x_name,y_name = args.variables.split(',')

        mlruns_dir = args.mlruns_dir
        mlflow.set_tracking_uri(mlruns_dir)
        experiment_name = args.experiment_name

        # get experiment id
        experiment_id = args.experiment_ids or mlflow.get_experiment_by_name(experiment_name).experiment_id
        print('Analyzing experiment', experiment_id)

        # get all runs
        runs = mlflow.search_runs(experiment_ids=experiment_id)
        # filter out runs that are not successful
        runs = runs[runs['status'] == 'FINISHED']
        if len(runs) == 0:
            print('No runs found')
            exit(0)
        # filter out runs that are not the right defense. (using params.defense)
        runs = runs[runs['params.defense'] == args.defense]
        
        
        # get all defense parameters
        Xs = list(runs[f'params.{x_name}'].unique())
        Xs.sort(key=lambda x: float(x) if x != 'None' else None)
        Xs = Xs[:25]
        Ys = list(runs[f'params.{y_name}'].unique())
        Ys.sort(key=lambda x: float(x) if x != 'None' else None)
        Ys = Ys[::-1]

        # defended robustness
        defended_robustness = np.zeros((len(Ys),len(Xs)))
        for i,y in enumerate(Ys):
            for j,x in enumerate(Xs):
                run = runs[(runs[f'params.{x_name}'] == x) & (runs[f'params.{y_name}'] == y)]
                if not run.empty:
                    defended_robustness[i,j] = run['metrics.aee_avg_def-advdef'].values[0]

        ax = plt.gca()
        plot = ax.imshow(defended_robustness, interpolation='bilinear')
        # plt.title('Defended robustness $AEE(\\tilde{F}, \\check{F} )$ - Lower is better defense')
        # rename 's' to '$s_\text{ilp}$' if defense is ilp and to $b_\text{lgs}$ if defense is lgs
        if args.defense == 'ilp' and y_name == 's':
            print('renaming s to s_ilp')
            y_disp_name = '$s_\mathrm{ILP}$'
            x_disp_name = x_name
        elif args.defense == 'lgs' and y_name == 's':
            print('renaming s to b_lgs')
            y_disp_name = '$b_\mathrm{LGS}$'
            x_disp_name = x_name
        # switch k and o to uppercase
        elif x_name == 'k' and y_name == 'o':
            print('renaming k to K and o to O')
            x_disp_name = '$K$'
            y_disp_name = '$O$'
        else:
            print('no renaming')
            x_disp_name = f'${x_name}$'
            y_disp_name = f'${y_name}$'
        plt.xlabel(x_disp_name)
        plt.ylabel(y_disp_name)
        # plt.xticks(np.arange(len(Xs)), Xs)
        # plt.yticks(np.arange(len(Ys)), Ys)
        # only show every other tick
        plt.xticks(np.arange(len(Xs))[::2], Xs[::2])
        plt.yticks(np.arange(len(Ys))[::2], Ys[::2])
        # plt.colorbar() only as large as the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
        # smoothed level curve
        levels = np.arange(0, max(defended_robustness.flatten()), 5)
        CS = plt.contour(defended_robustness, levels, colors='k', linewidths=0.5)
        # plt.show(block=False)
        plt.savefig(f'./{savefolder}/defended_robustness-{args.defense}-{args.variables}.pdf')

        # 3d plot
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.figure().add_subplot(projection='3d')
        X, Y = np.meshgrid([float(x) for x in Xs], [float(y) for y in Ys])
        ax.plot_surface(X, Y, defended_robustness, cmap='viridis', edgecolor='none')
        ax.set_title('Defended robustness $AEE(\\tilde{F}, \\check{F} )$ - Lower is better defense')
        # rename 's' to '$s_\text{ilp}$' if defense is ilp and to $b_\text{lgs}$ if defense is lgs
        if args.defense == 'ilp' and y_name == 's':
            print('renaming s to s_ilp')
            y_disp_name = '$s_\mathrm{ILP}$'
            x_disp_name = x_name
        elif args.defense == 'lgs' and y_name == 's':
            print('renaming s to b_lgs')
            y_disp_name = '$b_\mathrm{LGS}$'
            x_disp_name = x_name
        # switch k and o to uppercase
        elif x_name == 'k' and y_name == 'o':
            print('renaming k to K and o to O')
            x_disp_name = '$K$'
            y_disp_name = '$O$'
        else:
            print('no renaming')
            x_disp_name = f'${x_name}$'
            y_disp_name = f'${y_name}$'
        ax.set_xlabel(x_disp_name)
        ax.set_ylabel(y_disp_name)
        ax.set_zlabel('AEE')
        plt.savefig(f'./{savefolder}/defended_robustness_3d-{args.defense}-{args.variables}.pdf')

        # now aee_avg_undef-def
        accuracy = np.zeros((len(Ys),len(Xs)))
        for i,y in enumerate(Ys):
            for j,x in enumerate(Xs):
                run = runs[(runs[f'params.{x_name}'] == x) & (runs[f'params.{y_name}'] == y)]
                if not run.empty:
                    accuracy[i,j] = run['metrics.aee_avg_undef-def'].values[0]

        plt.figure()    
        ax = plt.gca()
        plot = ax.imshow(accuracy, interpolation='bilinear')
        # plt.title('Accuray $AEE(F, \\tilde{F} )$ - Lower is better (nondisruptive defense)')
        plt.xlabel(x_disp_name)
        plt.ylabel(y_disp_name)
        # plt.xticks(np.arange(len(Xs)), Xs)
        # plt.yticks(np.arange(len(Ys)), Ys)
        # only show every other tick
        plt.xticks(np.arange(len(Xs))[::2], Xs[::2])
        plt.yticks(np.arange(len(Ys))[::2], Ys[::2])
        # level curve
        levels = np.arange(0, max(accuracy.flatten()), 5)
        CS = plt.contour(accuracy, levels, colors='k', linewidths=0.5)
        # plt.colorbar() only as large as the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
        # plt.show()
        plt.savefig(f'./{savefolder}/accuracy-{args.defense}-{args.variables}.pdf')
        print('done')
    
    if len(args.variables)==1:
        x_name = args.variables

        mlruns_dir = args.mlruns_dir
        mlflow.set_tracking_uri(mlruns_dir)
        experiment_name = args.experiment_name

        # get experiment id
        experiment_id = args.experiment_id or mlflow.get_experiment_by_name(experiment_name).experiment_id

        # get all runs
        runs = mlflow.search_runs(experiment_ids=experiment_id)
        # filter out runs that are not successful
        runs = runs[runs['status'] == 'FINISHED']
        if len(runs) == 0:
            print('No runs found')
            exit(0)
        
        
        # get all defense parameters
        Xs = list(runs[f'params.{x_name}'].unique())
        Xs.sort(key=lambda x: float(x) if x != 'None' else 0)

        defended_robustness = np.zeros((len(Xs)))
        for j,x in enumerate(Xs):
            run = runs[(runs[f'params.{x_name}'] == x)]
            if not run.empty:
                defended_robustness[j] = run['metrics.aee_avg_def-advdef'].values[0]

        plt.figure()
        plt.plot(Xs, defended_robustness)
        # plt.title('Defended robustness $AEE(\\tilde{F}, \\check{F} )$ - Lower is better defense')
        plt.xlabel(x_name)
        plt.ylabel('AEE')
        # plt.show(block=False)
        plt.savefig(f'./{savefolder}/defended_robustness-{args.defense}-{args.variables}.pdf')

        # now aee_avg_undef-def
        accuracy = np.zeros((len(Xs)))
        for j,x in enumerate(Xs):
            run = runs[(runs[f'params.{x_name}'] == x)]
            if not run.empty:
                accuracy[j] = run['metrics.aee_avg_undef-def'].values[0]
        
        plt.figure()
        plt.plot(Xs, accuracy)
        # plt.title('Accuray $AEE(F, \\tilde{F} )$ - Lower is better (nondisruptive defense)')
        plt.xlabel(x_name)
        plt.ylabel('AEE')
        # plt.show()
        plt.savefig(f'./{savefolder}/accuracy-{args.defense}-{args.variables}.pdf')