import torch, time, math, os

import trimesh
import numpy as np
import point_cloud_utils as pcu
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from scipy.interpolate import interp1d


SMALL_SIZE = 13
MEDIUM_SIZE = 13
BIGGER_SIZE = 16
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlematplotlib.rc('font', **font)
my_cmap = plt.colormaps.get_cmap('tab20')

def compute_curve(intervals, our_ae, smooth=True):
    cdf_ratios = []
    for xi in intervals:
        data_lt_xi_mask = our_ae < xi 
        cdf_ratios.append(np.sum(data_lt_xi_mask) / data_lt_xi_mask.shape[0])
    cdf_ratios = np.asarray(cdf_ratios)
    if smooth:
        interpolation = interp1d(intervals, cdf_ratios, kind='quadratic')

        ecdf_x = np.arange(0, intervals.max(), 0.0001)
        ecdf_y = interpolation(ecdf_x)
    else:
        ecdf_x = intervals
        ecdf_y = cdf_ratios

    return ecdf_x, ecdf_y


file = 'SE3_normalize_iter4_init8.txt'
dict_ours = np.load("lib_shape_prior/log/server/3rscan_bs64/error_dict_category.npz")
dict_baseline1  = np.load("baselines/GeoTransformer/error_dict.npz")
dict_baseline2 = np.load("baselines/RPMNet/error_dict.npz")
dict_baseline3 = np.load("baselines/Free_Reg/error_dict.npz")

full_labels = ['chair', 'sofa', 'pillow', 'table', 'trash_bin', 'others', 'bench']
labels = ['chair', 'sofa', 'pillow', 'table', 'trash_bin', 'others']


rre_list = [dict_ours['rre_lst'][dict_ours['shape_lst']==label] for label in labels]
rre_list[-2] = np.concatenate([rre_list[-2], rre_list[-1]]) # merge bench to others
rre_thres = 10
intervals = np.arange(0, rre_thres+1, rre_thres/3)


ecdf1_x, ecdf1_y = compute_curve(intervals, rre_list[0])
ecdf2_x, ecdf2_y = compute_curve(intervals, rre_list[1])
ecdf3_x, ecdf3_y = compute_curve(intervals, rre_list[2])
ecdf4_x, ecdf4_y = compute_curve(intervals, rre_list[3])
ecdf5_x, ecdf5_y = compute_curve(intervals, rre_list[4])
# ecdf6_x, ecdf6_y = compute_curve(intervals, rre_list[5])


# fig = plt.figure(figsize=(6, 9))
fig, axs = plt.subplots(1,2,figsize=(12,4))
ax_11 = plt.subplot(1,2,1)
ax_11.plot(ecdf1_x, ecdf1_y*100, color= my_cmap(1), linewidth=3)
ax_11.plot(ecdf2_x, ecdf2_y*100, color= my_cmap(3), linewidth=3)
ax_11.plot(ecdf3_x, ecdf3_y*100, color= my_cmap(5), linewidth=3)
ax_11.plot(ecdf4_x, ecdf4_y*100, color= my_cmap(7), linewidth=3)
ax_11.plot(ecdf5_x, ecdf5_y*100, color= my_cmap(9), linewidth=3)
# ax_11.plot(ecdf6_x, ecdf6_y*100, color= my_cmap(11), linewidth=3)


ax_11.set_ylabel('ECDF [%]')
ax_11.set_xlabel('Rotation Error [$^\circ$]')
ax_11.grid(color='grey', linewidth=0.4)
ax_11.spines["top"].set_visible(False)
ax_11.spines["right"].set_visible(False)
ax_11.spines["left"].set_visible(False)
ax_11.spines["bottom"].set_visible(False)
ax_11.tick_params(axis=u'both', which=u'both',length=0)
ax_11.set_axisbelow(True)



te_list = [dict_ours['tsfm_lst'][dict_ours['shape_lst']==label] for label in labels]
te_list[-2] = np.concatenate([te_list[-2], te_list[-1]])
ax_12 = plt.subplot(1,2,2)
te_thres = 0.1
intervals = np.arange(0, te_thres+0.01, te_thres/3)
ecdf1_x, ecdf1_y = compute_curve(intervals, te_list[0])
ecdf2_x, ecdf2_y = compute_curve(intervals, te_list[1])
ecdf3_x, ecdf3_y = compute_curve(intervals, te_list[2])
ecdf4_x, ecdf4_y = compute_curve(intervals, te_list[3])
ecdf5_x, ecdf5_y = compute_curve(intervals, te_list[4])
# ecdf6_x, ecdf6_y = compute_curve(intervals, te_list[5])

ax_12.plot(ecdf1_x, ecdf1_y*100, color= my_cmap(1), linewidth=3)
ax_12.plot(ecdf2_x, ecdf2_y*100, color= my_cmap(3), linewidth=3)
ax_12.plot(ecdf3_x, ecdf3_y*100, color= my_cmap(5), linewidth=3)
ax_12.plot(ecdf4_x, ecdf4_y*100, color= my_cmap(7), linewidth=3)
ax_12.plot(ecdf5_x, ecdf5_y*100, color= my_cmap(9), linewidth=3)
# ax_12.plot(ecdf6_x, ecdf6_y*100, color= my_cmap(11), linewidth=3)

ax_12.set_ylabel('ECDF [%]')
ax_12.set_xlabel('Transformation RMSE [m]')
ax_12.grid(color='grey', linewidth=0.4)
ax_12.spines["top"].set_visible(False)
ax_12.spines["right"].set_visible(False)
ax_12.spines["left"].set_visible(False)
ax_12.spines["bottom"].set_visible(False)
ax_12.tick_params(axis=u'both', which=u'both',length=0)
ax_12.set_axisbelow(True)
# ax_12.legend(['Ours', 'GeoTransformer', 'RPMNet'], fancybox=True, shadow=False, frameon=True)
# plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0.)
fig.legend([ax_11, ax_12], labels=labels, loc="upper center", ncol=len(labels), frameon=False) 
plt.show()