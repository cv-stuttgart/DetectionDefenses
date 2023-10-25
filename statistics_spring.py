import numpy as np
from tqdm import tqdm
from glob import glob
import torch
import os
from torch.utils.data import DataLoader, Subset

import matplotlib
import matplotlib.pyplot as plt

from helper_functions import datasets
from helper_functions.config_specs import Paths
from helper_functions.ownutilities import prepare_dataloader
from helper_functions import ownutilities

outputfolder = "./spring_eval"
read_data = False
save_img_statistics = False
plot_img_statistics = True
process_data_spring = False
process_data_hd1k = False
test_data = False
test_data_sintel = False

def greedy_hist_sampler(hist_list, hist_all, bins_all, seed=1, num_scenes=5, fov=5, scene_pick=None):

    hist_list = np.array(hist_list)
    hist_list_orig = np.copy(hist_list)

    hist_all_normalized = hist_all / np.sum(hist_all)
    hist_greedy = np.zeros_like(hist_all)
    hist_greedyp1 = np.zeros_like(hist_all)
    
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(hist_list)

    ctr = 0
    error_norm_prev = np.inf
    val_hists = []
    trn_hists = []
    # for idx in len(hist_list):
    for hst_pth in hist_list:
        # print(hst_pth)
        if (ctr < num_scenes and scene_pick is None) or (scene_pick is not None and hst_pth.split("/")[-1].split("_")[0] in scene_pick):
            hst = np.load(hst_pth)
            hist_greedyp1 = hist_greedy + hst
            hist_greedyp1_normalized = hist_greedyp1 / np.sum(hist_greedyp1)

            error = np.sum(np.abs(hist_all_normalized-hist_greedyp1_normalized))
            # print(error)
            if error < error_norm_prev or (scene_pick is not None and hst_pth.split("/")[-1].split("_")[0] in scene_pick):
                # print("Added hst")
                val_hists += [hst_pth]
                ctr += 1
                error_norm_prev = error
                hist_greedy = np.copy(hist_greedyp1)
            else:
                trn_hists += [hst_pth]
        else:
            trn_hists += [hst_pth]

    print(f"Final matching error = {error_norm_prev}")

    return val_hists, trn_hists

if read_data:
    # ### Spring
    # scenes = sorted(os.listdir(os.path.join(Paths.config("spring"),Paths.splits("spring_train"))))
    # for scene in scenes:
    #     print(scene)
    #     dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True, scenes=[scene], fwd_only=True, camera=["left"])
    #     # dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True, scenes=None, fwd_only=True, camera=["left"])
    #     data_loader = DataLoader(dataset, batch_size=30, shuffle=False)

    #     hist_scene = torch.zeros(2000)
    #     bins_scene = None
    #     flow_vecs = 0
    #     for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
    #         flow_lengths = torch.sum(flow**2, dim=1).sqrt()
    #         print(flow_lengths.size())
    #         print(flow_lengths.flatten().size())
    #         flow_vecs += flow_lengths.flatten().size()[0]
    #         hist, bins = torch.histogram(flow_lengths.flatten(), torch.linspace(0,2000,2001))
    #         hist_scene += hist
    #         bins_scene = bins
    #         print(torch.sum(hist))
    #         print(torch.sum(hist_scene))
    #         print(flow_vecs)

    #     hist_scene_np = hist_scene.detach().numpy()
    #     bins_scene_np = bins_scene.detach().numpy()

    #     outpath = os.path.join(outputfolder, Paths.splits("spring_train"))
    #     os.makedirs(outpath, exist_ok=True)

    #     np.save(os.path.join(outpath, f"{scene}_hist_fwd_left.npy"), hist_scene_np)
    #     np.save(os.path.join(outpath, f"{scene}_bins_fwd_left.npy"), bins_scene_np)
    #     np.save(os.path.join(outpath, f"{scene}_nums_fwd_left.npy"), flow_vecs)


    #### HD1K

    scenes = ['000000','000001','000002','000003','000004','000005','000006','000007','000008','000009','000010','000011','000012','000013','000014','000015','000016','000017','000018','000019','000020','000021','000022','000023','000024','000025','000026','000027','000028','000029','000030','000031','000032','000033','000034','000035']
    for scene in scenes:
        print(scene)
        dataset = datasets.HD1K(root=Paths.config("hd1k"), has_gt=True, scenes=[scene])
        data_loader = DataLoader(dataset, batch_size=30, shuffle=False)

        hist_scene = torch.zeros(2000)
        bins_scene = None
        flow_vecs = 0
        for batch, (image1, image2, flow, valid) in enumerate(tqdm(data_loader)):
            flow_lengths = torch.sum(flow**2, dim=1).sqrt()
            valid_flatbool = valid.flatten()>0.5
            flow_vecs += flow_lengths.flatten()[valid_flatbool].size()[0]
            hist, bins = torch.histogram(flow_lengths.flatten()[valid_flatbool], torch.linspace(0,2000,2001))
            hist_scene += hist
            bins_scene = bins

        hist_scene_np = hist_scene.detach().numpy()
        bins_scene_np = bins_scene.detach().numpy()

        outpath = os.path.join(outputfolder, "hd1k_train")
        os.makedirs(outpath, exist_ok=True)

        np.save(os.path.join(outpath, f"{scene}_hist.npy"), hist_scene_np)
        np.save(os.path.join(outpath, f"{scene}_bins.npy"), bins_scene_np)
        np.save(os.path.join(outpath, f"{scene}_nums.npy"), flow_vecs)


if save_img_statistics:

    def jointGradMag(img):
        N,C,H,W = img.size()
        forward_h = torch.tensor([[[[0.,0.,0.],[0.,-1.,1.],[0.,0.,0.]]]])
        forward_v = forward_h.permute(0,1,3,2).contiguous()
        kv,kh = [forward_v,forward_h]
        kv,kh = kv.to(img.device),kh.to(img.device)
        
        # Perform Convolution
        img_pad = torch.nn.functional.pad(img,(1,1,1,1),mode="reflect")
        dy = torch.nn.functional.conv2d(img_pad.view(-1,1,H+2,W+2).contiguous(),kv,padding="valid").view(N,C,H,W).contiguous()
        dx = torch.nn.functional.conv2d(img_pad.view(-1,1,H+2,W+2).contiguous(),kh,padding="valid").view(N,C,H,W).contiguous()
        
        # Joint Magnitude
        G_joint_mag=torch.sqrt(torch.sum(dx**2,1,keepdim=True)+\
                               torch.sum(dy**2,1,keepdim=True)+1e-6)
        
        return G_joint_mag

    def joint2ndGradMag(img):
        N,C,H,W = img.size()
        forward_h = torch.tensor([[[[0.,0.,0.],[0.,-1.,1.],[0.,0.,0.]]]])
        forward_v = forward_h.permute(0,1,3,2).contiguous()
        kv,kh = [forward_v,forward_h]
        kv,kh = kv.to(img.device),kh.to(img.device)
        
        # Perform Convolution
        img_pad = torch.nn.functional.pad(img,(1,1,1,1),mode="reflect")
        dy = torch.nn.functional.conv2d(img_pad.view(-1,1,H+2,W+2).contiguous(),kv,padding="valid").view(N,C,H,W).contiguous()
        dx = torch.nn.functional.conv2d(img_pad.view(-1,1,H+2,W+2).contiguous(),kh,padding="valid").view(N,C,H,W).contiguous()
        dy_pad = torch.nn.functional.pad(dy,(1,1,1,1),mode="reflect")
        dx_pad = torch.nn.functional.pad(dx,(1,1,1,1),mode="reflect")
        dydy=torch.nn.functional.conv2d(dy_pad.view(-1,1,H+2,W+2).contiguous(),kv,padding="valid").view(N,C,H,W).contiguous()
        dxdx=torch.nn.functional.conv2d(dx_pad.view(-1,1,H+2,W+2).contiguous(),kh,padding="valid").view(N,C,H,W).contiguous()

        # Joint Magnitude
        G_joint_mag=torch.sqrt(torch.sum(dxdx**2,1,keepdim=True)+\
                               torch.sum(dydy**2,1,keepdim=True)+1e-6)
        
        return G_joint_mag


    def save_gradmags_plot_hist(dataset, ds_name, outputfolder):
        data_loader = DataLoader(dataset, batch_size=30, shuffle=False)

        hist_gradmag = torch.zeros(1000)
        bins_gradmag = None
        hist_grad2ndmag = torch.zeros(1000)
        bins_grad2ndmag = None
        px_gradmag = 0
        px_grad2ndmag = 0
        for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
            grad_mag = jointGradMag(image1)
            grad_2ndmag = joint2ndGradMag(image1)

            hist_gm, bins_gm = torch.histogram(grad_mag.flatten(), torch.linspace(0,1000,1001))
            hist_gradmag += hist_gm
            bins_gradmag = bins_gm
            px_gradmag += grad_mag.flatten().size()[0]
            hist_gm2, bins_gm2 = torch.histogram(grad_2ndmag.flatten(), torch.linspace(0,1000,1001))
            hist_grad2ndmag += hist_gm2
            bins_grad2ndmag = bins_gm2
            px_grad2ndmag += grad_2ndmag.flatten().size()[0]

        hist_gradmag_np = hist_gradmag.detach().numpy()
        bins_gradmag_np = bins_gradmag.detach().numpy()
        hist_grad2ndmag_np = hist_grad2ndmag.detach().numpy()
        bins_grad2ndmag_np = bins_grad2ndmag.detach().numpy()

        outpath = os.path.join(outputfolder, "image_statistics")
        os.makedirs(outpath, exist_ok=True)

        np.save(os.path.join(outpath, f"{ds_name}_hist_gradmag.npy"), hist_gradmag_np)
        np.save(os.path.join(outpath, f"{ds_name}_bins_gradmag.npy"), bins_gradmag_np)
        np.save(os.path.join(outpath, f"{ds_name}_nums_gradmag.npy"), px_gradmag)
        np.save(os.path.join(outpath, f"{ds_name}_hist_grad2ndmag.npy"), hist_grad2ndmag_np)
        np.save(os.path.join(outpath, f"{ds_name}_bins_grad2ndmag.npy"), bins_grad2ndmag_np)
        np.save(os.path.join(outpath, f"{ds_name}_nums_grad2ndmag.npy"), px_grad2ndmag)

        hist_gradmag_np_scaled = hist_gradmag_np / px_gradmag
        hist_grad2ndmag_np_scaled = hist_grad2ndmag_np / px_grad2ndmag

        fig, ax = plt.subplots(3, 2, figsize=(14, 12))
        ax[0,0].stairs(hist_gradmag_np, bins_gradmag_np, label='gradmag')
        ax[0,0].stairs(hist_grad2ndmag_np, bins_grad2ndmag_np, label='grad2ndmag')
        ax[0,0].legend()
        ax[0,1].stairs(hist_gradmag_np_scaled, bins_gradmag_np, label='gradmag')
        ax[0,1].stairs(hist_grad2ndmag_np_scaled, bins_grad2ndmag_np, label='grad2ndmag')
        ax[1,0].stairs(np.log(hist_gradmag_np), bins_gradmag_np, label='gradmag')
        ax[1,0].stairs(np.log(hist_grad2ndmag_np), bins_grad2ndmag_np, label='grad2ndmag')
        ax[1,1].stairs(np.log(hist_gradmag_np_scaled), bins_gradmag_np, label='gradmag')
        ax[1,1].stairs(np.log(hist_grad2ndmag_np_scaled), bins_grad2ndmag_np, label='grad2ndmag')
        ax[2,0].stairs(np.log(hist_gradmag_np)/np.sum([i if i != -np.inf else 0 for i in np.log(hist_gradmag_np)]), bins_gradmag_np, label='gradmag')
        ax[2,0].stairs(np.log(hist_grad2ndmag_np)/np.sum([i if i != -np.inf else 0 for i in np.log(hist_grad2ndmag_np)]), bins_grad2ndmag_np, label='grad2ndmag')
        plt.savefig(f'histogram_{ds_name}.png')

    # ######## KITTI ##########################################################################################
    # dataset = datasets.KITTI(split=Paths.splits("kitti_train"), aug_params=None, root=Paths.config("kitti15"), has_gt=True)
    # save_gradmags_plot_hist(dataset, "KITTI", outputfolder)


    # ######## Sintel ##########################################################################################
    # dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), dstype="final", has_gt=True,
    #         scenes=["ambush_2", "bamboo_2", "cave_2", "market_2", "shaman_2", "temple_2"], every_nth_img=3)
    # save_gradmags_plot_hist(dataset, "SintelYang", outputfolder)


    # ######## Sintel clean ##########################################################################################
    # dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), dstype="clean", has_gt=True,
    #         scenes=["ambush_2", "bamboo_2", "cave_2", "market_2", "shaman_2", "temple_2"], every_nth_img=3)
    # save_gradmags_plot_hist(dataset, "SintelYangC", outputfolder)


    # ######## Spring ##########################################################################################
    # dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True, 
    #         scenes=["0002","0010","0018","0026","0032","0045"], 
    #         fwd_only=True, camera=["left"], half_dimensions=True)
    # save_gradmags_plot_hist(dataset, "SpringScheurer", outputfolder)


    ######## Spring split but not half ##########################################################################################
    # dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True, 
    #         scenes=["0002","0010","0018","0026","0032","0045"], 
    #         fwd_only=True, camera=["left"], half_dimensions=False)
    # save_gradmags_plot_hist(dataset, "SpringScheurerfull", outputfolder)


    # # ######## Driving ##########################################################################################
    # dataset = datasets.Driving(root=Paths.config("driving"), has_gt=True,
    #         # dstype=['clean', 'final'], focallength=["15mm", "30mm"], drivingcamview=["forward", "backward"], 
    #         # direction=["forward", "backward"], speed=["fast", "slow"], camera=["left","right"])
    #         dstype=['clean'], focallength=["15mm"], drivingcamview=["forward"], 
    #         direction=["forward"], speed=["fast"], camera=["left"])
    # save_gradmags_plot_hist(dataset, "DrivingC15fast", outputfolder)

    # dataset = datasets.Driving(root=Paths.config("driving"), has_gt=True,
    #         # dstype=['clean', 'final'], focallength=["15mm", "30mm"], drivingcamview=["forward", "backward"], 
    #         # direction=["forward", "backward"], speed=["fast", "slow"], camera=["left","right"])
    #         dstype=['final'], focallength=["15mm"], drivingcamview=["forward"], 
    #         direction=["forward"], speed=["fast"], camera=["left"])
    # save_gradmags_plot_hist(dataset, "DrivingF15fast", outputfolder)

    ######## DH1K ##########################################################################################
    # dataset = datasets.HD1K(root=Paths.config("hd1k"), has_gt=True, every_nth_img=5)
    # save_gradmags_plot_hist(dataset, "HD1K5", outputfolder)

    # dataset = datasets.HD1K(root=Paths.config("hd1k"), has_gt=True, every_nth_img=5, half_dimensions=True)
    # save_gradmags_plot_hist(dataset, "HD1K5h", outputfolder)

    scenes_val = ["000009", "000013", "000018", "000019", "000032"]
    dataset = datasets.HD1K(root=Paths.config("hd1k"), has_gt=True, scenes=scenes_val)
    save_gradmags_plot_hist(dataset, "HD1KS", outputfolder)

    scenes_val = ["000009", "000013", "000018", "000019", "000032"]
    dataset = datasets.HD1K(root=Paths.config("hd1k"), has_gt=True, scenes=scenes_val, half_dimensions=True)
    save_gradmags_plot_hist(dataset, "HD1KSh", outputfolder)
    

if plot_img_statistics:
    outpath = os.path.join(outputfolder, "image_statistics")
    cutoff_gradmag = 500
    fig_gradmag, ax_gradmag = plt.subplots(3, 2, figsize=(14, 12))
    cutoff_grad2ndmag = 100
    fig_grad2ndmag, ax_grad2ndmag = plt.subplots(3, 2, figsize=(14, 12))


    def add_hist_to_plot(ds_name, ds_label, ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag, smooth=True):
        hist_gradmag = np.load(os.path.join(outpath, f"{ds_name}_hist_gradmag.npy"))
        bins_gradmag = np.load(os.path.join(outpath, f"{ds_name}_bins_gradmag.npy"))
        hist_grad2ndmag = np.load(os.path.join(outpath, f"{ds_name}_hist_grad2ndmag.npy"))
        bins_grad2ndmag = np.load(os.path.join(outpath, f"{ds_name}_bins_grad2ndmag.npy"))

        if smooth:
            smoothing = 5
            hist_gradmag = np.sum(hist_gradmag.reshape((int(hist_gradmag.shape[0]/smoothing), smoothing)), axis=1)
            hist_grad2ndmag = np.sum(hist_grad2ndmag.reshape((int(hist_grad2ndmag.shape[0]/smoothing), smoothing)), axis=1)
            cutoff_gradmag = int(cutoff_gradmag/smoothing)
            cutoff_grad2ndmag = int(cutoff_grad2ndmag/smoothing)
            bins_gradmag = bins_gradmag[::smoothing]
            bins_grad2ndmag = bins_grad2ndmag[::smoothing]

        hist_gradmag_scaled = hist_gradmag / np.sum(hist_gradmag)
        hist_grad2ndmag_scaled = hist_grad2ndmag / np.sum(hist_grad2ndmag)

        ax_gradmag[0,0].stairs(hist_gradmag[:-cutoff_gradmag], bins_gradmag[:-cutoff_gradmag], label=ds_label)
        ax_gradmag[0,1].stairs(hist_gradmag_scaled[:-cutoff_gradmag], bins_gradmag[:-cutoff_gradmag], label=ds_label)
        ax_gradmag[1,0].stairs(np.log(hist_gradmag[:-cutoff_gradmag]), bins_gradmag[:-cutoff_gradmag], label=ds_label)
        ax_gradmag[1,1].stairs(np.log(hist_gradmag_scaled[:-cutoff_gradmag]), bins_gradmag[:-cutoff_gradmag], label=ds_label)
        ax_gradmag[2,0].stairs(np.log(hist_gradmag[:-cutoff_gradmag])/np.sum([i if i != -np.inf else 0 for i in np.log(hist_gradmag[:-cutoff_gradmag])]), bins_gradmag[:-cutoff_gradmag], label=ds_label)

        ax_grad2ndmag[0,0].stairs(hist_grad2ndmag[:-cutoff_grad2ndmag], bins_grad2ndmag[:-cutoff_grad2ndmag], label=ds_label)
        ax_grad2ndmag[0,1].stairs(hist_grad2ndmag_scaled[:-cutoff_grad2ndmag], bins_grad2ndmag[:-cutoff_grad2ndmag], label=ds_label)
        ax_grad2ndmag[1,0].stairs(np.log(hist_grad2ndmag[:-cutoff_grad2ndmag]), bins_grad2ndmag[:-cutoff_grad2ndmag], label=ds_label)
        ax_grad2ndmag[1,1].stairs(np.log(hist_grad2ndmag_scaled[:-cutoff_grad2ndmag]), bins_grad2ndmag[:-cutoff_grad2ndmag], label=ds_label)
        ax_grad2ndmag[2,0].stairs(np.log(hist_grad2ndmag[:-cutoff_grad2ndmag])/np.sum([i if i != -np.inf else 0 for i in np.log(hist_grad2ndmag[:-cutoff_grad2ndmag])]), bins_grad2ndmag[:-cutoff_grad2ndmag], label=ds_label)

        return ax_gradmag, ax_grad2ndmag

    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("KITTI", "kitti", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("SintelYang", "sinYa", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("SpringScheurer", "spShr", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("SpringScheurerfull", "spShf", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("DrivingC15fast", "dC15f", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("DrivingF15fast", "dF15f", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("SintelYangC", "sinYC", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    # ax_gradmag, ax_grad2ndmag = add_hist_to_plot("HD1K5", "HD1K5", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("HD1K5h", "HD1K5h", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("HD1KS", "HD1KS", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)
    ax_gradmag, ax_grad2ndmag = add_hist_to_plot("HD1KSh", "HD1KSh", ax_gradmag, ax_grad2ndmag, cutoff_gradmag, cutoff_grad2ndmag)

    ax_gradmag[0,0].legend()
    ax_grad2ndmag[0,0].legend()

    fig_gradmag.savefig('histogram_allds_gradmag.png')
    fig_grad2ndmag.savefig('histogram_allds_grad2ndmag.png')


if process_data_spring:
    hist_list = sorted(glob(os.path.join(outputfolder, Paths.splits("spring_train"), '*_hist_fwd_left.npy')))
    # print(hist_list)
    bins = np.linspace(0,2000,2001)

    hist_all = np.zeros(2000)
    hist_val = np.zeros(2000)
    hist_trn = np.zeros(2000)

    hist_all_scaled = np.zeros(2000)
    hist_val_scaled = np.zeros(2000)
    hist_trn_scaled = np.zeros(2000)

    for hst_pth in hist_list:
        hst = np.load(hst_pth)
        hist_all += hst


    number_list = ["0001", "0002", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0020", "0021", "0022", "0023", "0024", "0025", "0026", "0027", "0030", "0032", "0033", "0036", "0037", "0038", "0039", "0041", "0043", "0044", "0045", "0047"]
    # number_list = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 30, 32, 33, 36, 37, 38, 39, 41, 43, 44, 45, 47]
    count_list = [247, 130, 97, 115, 190, 147, 176, 121, 121, 94, 265, 83, 157, 262, 281, 47, 110, 307, 298, 18, 82, 63, 126, 28, 12, 125, 72, 65, 107, 78, 144, 76, 95, 90, 70, 197, 267]

    def sum_numbers(val_list, number_list, count_list):
        numbers = sorted([i.split("/")[-1].split("_")[0] for i in val_list])
        sm = 0
        for num in numbers:
            # print(num)
            idx = np.where(np.array(number_list)==num)[0][0]
            # print(idx)
            sm += count_list[idx]
        return sm
    print(f"all numbers {sum_numbers(number_list,number_list,count_list)}")

    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=0)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=1)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=2)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=3)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=4)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=5)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=6)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=7)
    print(sum_numbers(val_list, number_list, count_list))
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=8)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=9)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=10)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=11)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=12)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=13)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=14)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=16)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=17)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=18)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=19)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=20)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=21)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=22)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=23)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=24)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=25)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=26)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=27)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=28)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=29)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=30)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=31)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=32)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=33)
    print(sum_numbers(val_list, number_list, count_list))
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=34)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=35)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=36)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=38)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=39)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=40)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=41)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=42)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=43)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=45)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=46)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=47)
    print(sum_numbers(val_list, number_list, count_list))
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=48)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=49)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=37)
    print(sum_numbers(val_list, number_list, count_list))
    print(val_list)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=44)
    print(sum_numbers(val_list, number_list, count_list))
    print(val_list)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=1, num_scenes=5, fov=5, scene_pick=["0002","0010","0018","0026","0032","0045"])
    print(sum_numbers(val_list, number_list, count_list))
    print(val_list)

    for hst_pth in val_list:
        hst = np.load(hst_pth)
        hist_val += hst

    for hst_pth in train_list:
        hst = np.load(hst_pth)
        hist_trn += hst

    print(hist_all)
    sum_all = np.sum(hist_all)
    hist_all_scaled = hist_all / sum_all

    sum_val = np.sum(hist_val)
    hist_val_scaled = hist_val / sum_val

    sum_trn = np.sum(hist_trn)
    hist_trn_scaled = hist_trn / sum_trn


    cutoff = 1
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    ax[0,0].stairs(hist_all[:-cutoff], bins[:-cutoff], label='all')
    ax[0,0].stairs(hist_val[:-cutoff], bins[:-cutoff], label='val')
    ax[0,0].stairs(hist_trn[:-cutoff], bins[:-cutoff], label='trn')
    ax[0,1].stairs(hist_all_scaled[:-cutoff], bins[:-cutoff], label='all')
    ax[0,1].stairs(hist_val_scaled[:-cutoff], bins[:-cutoff], label='val')
    ax[0,1].stairs(hist_trn_scaled[:-cutoff], bins[:-cutoff], label='trn')
    ax[1,0].stairs(np.log(hist_all)[:-cutoff], bins[:-cutoff], label='all')
    ax[1,0].stairs(np.log(hist_val)[:-cutoff], bins[:-cutoff], label='val')
    ax[1,0].stairs(np.log(hist_trn)[:-cutoff], bins[:-cutoff], label='trn')
    ax[1,1].stairs(np.log(hist_all_scaled)[:-cutoff], bins[:-cutoff], label='all')
    ax[1,1].stairs(np.log(hist_val_scaled)[:-cutoff], bins[:-cutoff], label='val')
    ax[1,1].stairs(np.log(hist_trn_scaled)[:-cutoff], bins[:-cutoff], label='trn')
    ax[2,0].stairs(np.log(hist_all)[:-cutoff]/np.sum([i if i != -np.inf else 0 for i in np.log(hist_all)]), bins[:-cutoff], label='all')
    ax[2,0].stairs(np.log(hist_val)[:-cutoff]/np.sum([i if i != -np.inf else 0 for i in np.log(hist_val)]), bins[:-cutoff], label='val')
    ax[2,0].stairs(np.log(hist_trn)[:-cutoff]/np.sum([i if i != -np.inf else 0 for i in np.log(hist_trn)]), bins[:-cutoff], label='trn')
    plt.savefig('histogram.png')

    print(np.sum([i if i != -np.inf else 0 for i in np.log(hist_all)]))
    print(np.sum([i if i != -np.inf else 0 for i in np.log(hist_val)]))
    print(np.sum([i if i != -np.inf else 0 for i in np.log(hist_trn)]))
    print("done")

if process_data_hd1k:
    hist_list = sorted(glob(os.path.join(outputfolder, "hd1k_train", '*_hist.npy')))
    # print(hist_list)
    bins = np.linspace(0,2000,2001)

    hist_all = np.zeros(2000)
    hist_val = np.zeros(2000)
    hist_trn = np.zeros(2000)

    hist_all_scaled = np.zeros(2000)
    hist_val_scaled = np.zeros(2000)
    hist_trn_scaled = np.zeros(2000)

    for hst_pth in hist_list:
        hst = np.load(hst_pth)
        hist_all += hst


    number_list = ['000000','000001','000002','000003','000004','000005','000006','000007','000008','000009','000010','000011','000012','000013','000014','000015','000016','000017','000018','000019','000020','000021','000022','000023','000024','000025','000026','000027','000028','000029','000030','000031','000032','000033','000034','000035']
    count_list = [44, 9, 8, 8, 13, 8, 12, 66, 38, 16, 26, 68, 69, 25, 8, 13, 68, 34, 25, 14, 44, 9, 8, 40, 10, 27, 54, 15, 41, 57, 70, 13, 14, 17, 15, 41]

    def sum_numbers(val_list, number_list, count_list):
        numbers = sorted([i.split("/")[-1].split("_")[0] for i in val_list])
        sm = 0
        for num in numbers:
            # print(num)
            idx = np.where(np.array(number_list)==num)[0][0]
            # print(idx)
            sm += count_list[idx]
        return sm
    print(f"all numbers {sum_numbers(number_list,number_list,count_list)}")

    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=0)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=1)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=2)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=3)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=4)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=5)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=6)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=7)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=8)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=9)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=10)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=11)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=12)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=13)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=14)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=16)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=17)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=18)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=19)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=20)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=22)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=23)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=24)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=25)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=26)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=27)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=28)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=29)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=30)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=31)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=32)
    print(sum_numbers(val_list, number_list, count_list))
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=33)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=34)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=35)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=36)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=37)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=38)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=39)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=40)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=41)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=42)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=43)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=44)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=45)
    print(sum_numbers(val_list, number_list, count_list))
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=46)
    print(sum_numbers(val_list, number_list, count_list))
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=47)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=48)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=49)
    val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=21)
    print(sum_numbers(val_list, number_list, count_list))
    print(val_list)
    # print(sum_numbers(val_list, number_list, count_list))
    # print(val_list)
    # val_list, train_list = greedy_hist_sampler(hist_list, hist_all, bins, seed=1, num_scenes=5, fov=5, scene_pick=["0002","0010","0018","0026","0032","0045"])
    # print(sum_numbers(val_list, number_list, count_list))
    # print(val_list)

    for hst_pth in val_list:
        hst = np.load(hst_pth)
        hist_val += hst

    for hst_pth in train_list:
        hst = np.load(hst_pth)
        hist_trn += hst

    print(hist_all)
    sum_all = np.sum(hist_all)
    hist_all_scaled = hist_all / sum_all

    sum_val = np.sum(hist_val)
    hist_val_scaled = hist_val / sum_val

    sum_trn = np.sum(hist_trn)
    hist_trn_scaled = hist_trn / sum_trn


    cutoff = 1
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    ax[0,0].stairs(hist_all[:-cutoff], bins[:-cutoff], label='all')
    ax[0,0].stairs(hist_val[:-cutoff], bins[:-cutoff], label='val')
    ax[0,0].stairs(hist_trn[:-cutoff], bins[:-cutoff], label='trn')
    ax[0,1].stairs(hist_all_scaled[:-cutoff], bins[:-cutoff], label='all')
    ax[0,1].stairs(hist_val_scaled[:-cutoff], bins[:-cutoff], label='val')
    ax[0,1].stairs(hist_trn_scaled[:-cutoff], bins[:-cutoff], label='trn')
    ax[1,0].stairs(np.log(hist_all)[:-cutoff], bins[:-cutoff], label='all')
    ax[1,0].stairs(np.log(hist_val)[:-cutoff], bins[:-cutoff], label='val')
    ax[1,0].stairs(np.log(hist_trn)[:-cutoff], bins[:-cutoff], label='trn')
    ax[1,1].stairs(np.log(hist_all_scaled)[:-cutoff], bins[:-cutoff], label='all')
    ax[1,1].stairs(np.log(hist_val_scaled)[:-cutoff], bins[:-cutoff], label='val')
    ax[1,1].stairs(np.log(hist_trn_scaled)[:-cutoff], bins[:-cutoff], label='trn')
    ax[2,0].stairs(np.log(hist_all)[:-cutoff]/np.sum([i if i != -np.inf else 0 for i in np.log(hist_all)]), bins[:-cutoff], label='all')
    ax[2,0].stairs(np.log(hist_val)[:-cutoff]/np.sum([i if i != -np.inf else 0 for i in np.log(hist_val)]), bins[:-cutoff], label='val')
    ax[2,0].stairs(np.log(hist_trn)[:-cutoff]/np.sum([i if i != -np.inf else 0 for i in np.log(hist_trn)]), bins[:-cutoff], label='trn')
    plt.savefig('histogram.png')

    print(np.sum([i if i != -np.inf else 0 for i in np.log(hist_all)]))
    print(np.sum([i if i != -np.inf else 0 for i in np.log(hist_val)]))
    print(np.sum([i if i != -np.inf else 0 for i in np.log(hist_trn)]))
    print("done")

if test_data:
    data_loader, has_gt = prepare_dataloader(mode='training', dataset='Spring', batch_size=1)
    print(f"Testing Spring-training, has_gt={has_gt}")
    for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
        print(image1.size())
        print(flow.size())

    data_loader, has_gt = prepare_dataloader(mode='evaluation', dataset='Spring', batch_size=1)
    print(f"Testing Spring-evaluation, has_gt={has_gt}")
    for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
        print(image1.size())
        print(flow.size())

    data_loader, has_gt = prepare_dataloader(mode='training', dataset='SpringSplitScheurer', batch_size=1)
    print(f"Testing SpringSplitScheurer-training, has_gt={has_gt}")
    for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
        print(image1.size())
        print(flow.size())

    data_loader, has_gt = prepare_dataloader(mode='evaluation', dataset='SpringSplitScheurer', batch_size=1)
    print(f"Testing SpringSplitScheurer-evaluation, has_gt={has_gt}")
    for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
        print(image1.size())
        print(flow.size())
        print(torch.max(image1))
        print(torch.max(image2))
        print(torch.min(image1))
        ownutilities.quickvisualization_tensor(image1[0], "testimg.png", min=0., max=255.)
        ownutilities.quickvis_flow(flow[0], "testflow.png", auto_scale=True, max_scale=-1)


if test_data_sintel:

    # data_loader, has_gt = prepare_dataloader(mode='training', dataset='SintelSplitZhao', batch_size=1)
    # print(f"Testing SintelZhao-training, has_gt={has_gt}")
    # for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
    #     print(image1.size())
    #     print(flow.size())

    # data_loader, has_gt = prepare_dataloader(mode='evaluation', dataset='SintelSplitZhao', batch_size=1)
    # print(f"Testing SintelZhao-evaluation, has_gt={has_gt}")
    # for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
    #     print(image1.size())
    #     print(flow.size())


    data_loader, has_gt = prepare_dataloader(mode='training', dataset='SintelSplitYang', batch_size=1)
    print(f"Testing SintelYang-training, has_gt={has_gt}")
    for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
        print(image1.size())
        print(flow.size())

    data_loader, has_gt = prepare_dataloader(mode='evaluation', dataset='SintelSplitYang', batch_size=1)
    print(f"Testing SintelYang-evaluation, has_gt={has_gt}")
    for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):
        print(image1.size())
        print(flow.size())