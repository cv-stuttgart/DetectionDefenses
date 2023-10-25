# %% Script to extract patches ordered by parameters and neural networks

import matplotlib.pyplot as plt
import glob
import numpy as np
import os
# go to PATCHATTACK_HOME environment variable
os.chdir(os.environ['PATCHATTACK_HOME'])

nets = ['FlowNetC',  'FlowNetCRobust', 'PWCNet' ,'SpyNet', 'RAFT', 'GMA','FlowFormer']
defenses = ['None', 'LGS',  'ILP']
folder = 'results'

parameters={'optimizer':['sgd','ifgsm'],'lr':[.001,.01,.1,1.0,10.0,100.0,1000.0],'alpha':[1e-08], 'change_of_variables':[True,False], 'attack_target':['neg_flow'], 'loss':['acs']}

all_patches = {}
rankings = {}
for net in nets:
    for opt in parameters['optimizer']:
        for lr in parameters['lr']:
            for alpha in parameters['alpha']:
                for cov in parameters['change_of_variables']:
                    for attack_target in parameters['attack_target']:
                        for loss in parameters['loss']:
                            patches = []
                            for defense in defenses:
                                # find patch with param combination
                                file = f'{folder}/{net}_{defense.lower()}_evals/patches/*_{opt}_{lr}_{alpha}_{cov}_{attack_target}_{loss}_{defense.lower()}.png'
                                files = glob.glob(file)
                                if len(files)==0: continue
                                
                                files.sort() # first file is best performing
                                path = files[0]
                                # number is the * before optimizer
                                number = int(os.path.basename(path).split('_')[0])
                                rankings[f'{net}_{opt}_{lr}_{alpha}_{cov}_{attack_target}_{loss}_{defense.lower()}'] = number
                                all_patches[f'{net}_{opt}_{lr}_{alpha}_{cov}_{attack_target}_{loss}_{defense.lower()}'] = plt.imread(path)
                                # patches.append(plt.imread(path))
                            # if len(patches)==0: continue
                            # # plot patches by putting them side by side first
                            # patches = np.concatenate(patches, axis=1)
                            # plt.imshow(patches)
                            # plt.title(f'{net}_{opt}_{lr}_{alpha}_{cov}_{attack_target}_{loss}')
                            # plt.show(block=False)
                            # plt.pause(0.1)
                            # plt.waitforbuttonpress()
                            

# plot patches for all nets and some parameter combinations
attack_target = 'neg_flow'
loss = 'acs'
alpha = 1e-08

combinations = [
        # {'optimizer':'sgd','lr':.001,'change_of_variables':True},
        # {'optimizer':'sgd','lr':.001,'change_of_variables':False},
        # {'optimizer':'sgd','lr':.01,'change_of_variables':True},
        # {'optimizer':'sgd','lr':.01,'change_of_variables':False},
        # {'optimizer':'sgd','lr':.1,'change_of_variables':True},
        # {'optimizer':'sgd','lr':.1,'change_of_variables':False},
        # {'optimizer':'sgd','lr':1.0,'change_of_variables':True},
        # {'optimizer':'sgd','lr':1.0,'change_of_variables':False},
        {'optimizer':'sgd','lr':10.0,'change_of_variables':True},
        {'optimizer':'sgd','lr':10.0,'change_of_variables':False},
        {'optimizer':'sgd','lr':100.0,'change_of_variables':True},
        {'optimizer':'sgd','lr':100.0,'change_of_variables':False},
        # {'optimizer':'sgd','lr':1000.0,'change_of_variables':True},
        # {'optimizer':'sgd','lr':1000.0,'change_of_variables':False},
        # {'optimizer':'ifgsm','lr':.001,'change_of_variables':True},
        # {'optimizer':'ifgsm','lr':.001,'change_of_variables':False},
        {'optimizer':'ifgsm','lr':.01,'change_of_variables':True},
        {'optimizer':'ifgsm','lr':.01,'change_of_variables':False},
        {'optimizer':'ifgsm','lr':.1,'change_of_variables':True},
        {'optimizer':'ifgsm','lr':.1,'change_of_variables':False},
        {'optimizer':'ifgsm','lr':1.0,'change_of_variables':True},
        {'optimizer':'ifgsm','lr':1.0,'change_of_variables':False},
        # {'optimizer':'ifgsm','lr':10.0,'change_of_variables':True},
        # {'optimizer':'ifgsm','lr':10.0,'change_of_variables':False},
        # {'optimizer':'ifgsm','lr':100.0,'change_of_variables':True},
        # {'optimizer':'ifgsm','lr':100.0,'change_of_variables':False},
        # {'optimizer':'ifgsm','lr':1000.0,'change_of_variables':True},
        # {'optimizer':'ifgsm','lr':1000.0,'change_of_variables':False},
]
#%%
fig, axs = plt.subplots(len(combinations)*3+1,len(nets)+1) # +1 for overview what is what
[axi.set_axis_off() for axi in axs.ravel()]

# write in first row the names of the nets
for i, net in enumerate(nets):
    axs[0,i+1].set_title(net)
    axs[0,i+1].set_axis_off()
# write in first column the names of the parameter combinations 
# optimizer, lr, change of variables and write it vertically
for j, comb in enumerate(combinations):
    axs[1+j*3,0].text(0.5, 0.65, f'{comb["optimizer"].upper()}, {defenses[0]}, ', rotation=0, va='center', ha='center')
    axs[1+j*3,0].text(0.5, 0.40, f'lr:{comb["lr"]}, cov:{comb["change_of_variables"]}', rotation=0, va='center', ha='center')
    axs[1+j*3,0].set_axis_off()
    axs[2+j*3,0].text(0.5, 0.65, f'{comb["optimizer"].upper()}, {defenses[1]}, ', rotation=0, va='center', ha='center')
    axs[2+j*3,0].text(0.5, 0.40, f'lr:{comb["lr"]}, cov:{comb["change_of_variables"]}', rotation=0, va='center', ha='center')
    axs[2+j*3,0].set_axis_off()
    axs[3+j*3,0].text(0.5, 0.65, f'{comb["optimizer"].upper()}, {defenses[2]}, ', rotation=0, va='center', ha='center')
    axs[3+j*3,0].text(0.5, 0.40, f'lr:{comb["lr"]}, cov:{comb["change_of_variables"]}', rotation=0, va='center', ha='center')
    axs[3+j*3,0].set_axis_off()
for i, net in enumerate(nets):
    for j, comb in enumerate(combinations):
        for k, defense in enumerate(defenses):
            key = f'{net}_{comb["optimizer"]}_{comb["lr"]}_{alpha}_{comb["change_of_variables"]}_{attack_target}_{loss}_{defense.lower()}'
            try:
                axs[j*3+k+1,i+1].imshow(all_patches[key])
                axs[j*3+k+1,i+1].set_title(rankings[f'{net}_{comb["optimizer"]}_{comb["lr"]}_{alpha}_{comb["change_of_variables"]}_{attack_target}_{loss}_{defense.lower()}']+1)
            except KeyError:
                continue
            # axs[1+j*3+k,1+i].set_title(f'{net}_{comb["optimizer"]}_{comb["lr"]}_{alpha}_{comb["change_of_variables"]}_{attack_target}_{loss}_{defense.lower()}')
            axs[1+j*3+k,1+i].set_aspect('equal')
            
# increase the size of the figure
fig.set_size_inches(10, 70)
# save the figure with
plt.savefig(f'{folder}/patch-overview.pdf', bbox_inches='tight')

# %% Now export all patches to a folder and create latex table containing \includegraphics to this folder
# create folder
os.makedirs(f'{folder}/patches-overview', exist_ok=True)
# export all patches
for key, patch in all_patches.items():
    plt.imsave(f'{folder}/patches-overview/{key}.png', patch)
# create latex table

for defense in defenses:
    with open(f'{folder}/patches-overview/patches-overview_{defense}.tex', 'w') as f:
        f.write('\\begin{table*}\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular}{ c c c c c c c c c}\n')
        f.write('\\toprule\n')
        f.write(' & FlowNetC & FlowNetCRobust & RAFT & GMA & FlowFormer & SpyNet & PWCNet \\\\ \n')
        f.write('\\midrule\n')
        # write combinations in first column            
        for j, comb in enumerate(combinations):
            for i, net in enumerate([0]+nets):
                key = f'{net}_{comb["optimizer"]}_{comb["lr"]}_{alpha}_{comb["change_of_variables"]}_{attack_target}_{loss}_{defense.lower()}'
                if i == 0:
                    f.write(f'{comb["optimizer"].upper()},')
                    cov = "\\ding{51}" if comb["change_of_variables"] else "\\ding{55}"
                    f.write(f'lr:{comb["lr"]}, cov:{cov} & ')
                else:
                    try:
                        plt.imread(f'{folder}/patches-overview/{key}.png')
                        print('writing', key)
                        f.write(f'\\includegraphics[width=0.12\\textwidth]{{graphics/patches-overview/{key}.png}} & ')
                    except FileNotFoundError:
                        print('not found', key)
                        f.write(f' & ')
            f.write('\\\\ \n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\label{tab:patches-overview_'+defense+'}\n')
        f.write('\\caption{Patches of the adversarial examples for different networks and parameter combinations.}\n')
        f.write('\\end{table*}\n')

# %% create latex table containing the rankings
# in {folder}/{net}_{defense}_evals/{net}_{defense}_evals.csv} is the ranking of each parameter combination for each net
import pandas as pd
# looks like this:
# ,opt,lr,alpha,cov,target,loss,aee_avg_def-advdef,aee_avg_gt-undef,aee_avg_gt-def,n_runs
# 0,ifgsm,1.0,1e-08,False,neg_flow,acs_ilp,1.8157554945442824,0.6178302276320755,1.3006538267899304,4
# 1,sgd,10.0,1e-08,False,neg_flow,acs_ilp,1.7492947073932734,0.6178302276320755,1.3006538267899304,4
# 2,sgd,100.0,1e-08,False,neg_flow,acs_ilp,1.6873957523852587,0.6178302276320755,1.3006538267899304,4
# 3,ifgsm,0.1,1e-08,False,neg_flow,acs_ilp,1.388444800568395,0.6178302276320755,1.3006538267899304,4
# 4,ifgsm,0.01,1e-08,False,neg_flow,acs_ilp,1.148050673170248,0.6178302276320755,1.3006538267899304,4

for defense in defenses:
    with open(f'{folder}/patches-overview/patches-overview_{defense}_ranking.tex', 'w') as f:
        f.write('\\begin{table*}\n')
        f.write('\\centering\n')
        f.write('\\resizebox{\\textwidth}{!}{\n')
        f.write('\\begin{tabular}{ c c c c c c c c c}\n')
        f.write('\\toprule\n')
        f.write(' & FlowNetC & FlowNetCRobust & RAFT & GMA & FlowFormer & SpyNet & PWCNet \\\\ \n')
        f.write('\\midrule\n')
        # write combinations in first column            
        for j, comb in enumerate(combinations):
            for i, net in enumerate([0]+nets):
                key = f'{net}_{comb["optimizer"]}_{comb["lr"]}_{alpha}_{comb["change_of_variables"]}_{attack_target}_{loss}_{defense.lower()}'
                if i == 0:
                    f.write(f'{comb["optimizer"].upper()},')
                    cov = "\\ding{51}" if comb["change_of_variables"] else "\\ding{55}"
                    f.write(f'lr:{comb["lr"]}, cov:{cov} & ')
                else:
                    try:
                        # get aee_avg_def-advdef of this combination
                        df = pd.read_csv(f'{folder}/{net}_{defense.lower()}_evals/{net}_{defense.lower()}_evals.csv')
                        runs = df[df['opt']==comb["optimizer"]][df['lr']==comb["lr"]][df['cov']==comb["change_of_variables"]]
                        if len(runs) == 0:
                            raise FileNotFoundError
                        ranking = runs['aee_avg_def-advdef'].tolist()[0]
                        print('writing', key, ranking)
                        f.write(f'{ranking:.2f} & ')
                    except FileNotFoundError:
                        print('not found', key)
                        f.write(f' & ')
            f.write('\\\\ \n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('}\n')
        f.write(f'\\caption{{Defended Robustness for {defense} of the adversarial examples for different networks and parameter combinations.}}\n')
        f.write('\\label{tab:patches-overview_'+defense+'_ranking}\n')
        f.write('\\end{table*}\n')