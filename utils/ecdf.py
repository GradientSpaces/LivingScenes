import torch, time, math, os

import trimesh
import numpy as np
import point_cloud_utils as pcu
from pathlib import Path

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

def compute_curve(intervals, our_ae):
    cdf_ratios = []
    for xi in intervals:
        data_lt_xi_mask = our_ae < xi 
        cdf_ratios.append(np.sum(data_lt_xi_mask) / data_lt_xi_mask.shape[0])
    cdf_ratios = np.asarray(cdf_ratios)
    interpolation = interp1d(intervals, cdf_ratios, kind='quadratic')

    ecdf_x = np.arange(0, intervals.max(), 0.0001)
    ecdf_y = interpolation(ecdf_x)
    
    return ecdf_x, ecdf_y


file = 'SE3_normalize_iter4_init8.txt'
dict_ours = np.load("lib_shape_prior/log/server/3rscan_bs64/summary/relocalization_dict_mask3d.npz")
dict_baseline1  = np.load("baselines/GeoTransformer/summary_dict_mask3d.npz")
dict_baseline2 = np.load("baselines/RPMNet/summary_dict_mask3d.npz")
dict_baseline3 = np.load("baselines/Free_Reg/relocalization_dict_mask3d.npz")

labels = ['Ours', 'GeoTransformer', 'RPMNet', 'FreeReg']

result_lst = [dict_ours, dict_baseline1, dict_baseline2, dict_baseline3]

for label, result in zip(labels, result_lst):
    print(f"{label}: RR: {100 * (result['rre_lst']<10).mean():.2f} ")
    print(f"{label}: MedRE: {np.median(result['rre_lst'][result['rre_lst']<10]):.2f} (deg)")
    print(f"{label}: TE: {100* np.median(result['tsfm_lst']):.2f} (cm)")
    print(f"{label}: CD: {np.median(result['cd_lst']):.5f} ")



rre_thres = 15
intervals = np.arange(0, rre_thres+1, rre_thres/3)


ecdf1_x, ecdf1_y = compute_curve(intervals, dict_ours['rre_lst'])
ecdf2_x, ecdf2_y = compute_curve(intervals, dict_baseline1['rre_lst'])
ecdf3_x, ecdf3_y = compute_curve(intervals, dict_baseline2['rre_lst'])
ecdf4_x, ecdf4_y = compute_curve(intervals, dict_baseline3['rre_lst'])
# ecdf3 = compute_curve(intervals, results[200:300])
# ecdf4 = compute_curve(intervals, results[300:])

# fig = plt.figure(figsize=(6, 9))
fig, axs = plt.subplots(1,2,figsize=(12,4))
ax_11 = plt.subplot(1,2,1)
ax_11.plot(ecdf1_x, ecdf1_y*100, color= my_cmap(1), linewidth=3)
ax_11.plot(ecdf2_x, ecdf2_y*100, color= my_cmap(3), linewidth=3)
ax_11.plot(ecdf3_x, ecdf3_y*100, color= my_cmap(5), linewidth=3)
ax_11.plot(ecdf4_x, ecdf4_y*100, color= my_cmap(7), linewidth=3)


ax_11.set_ylabel('ECDF [%]')
ax_11.set_xlabel('Rotation Error [$^\circ$]')
ax_11.grid(color='grey', linewidth=0.4)
ax_11.spines["top"].set_visible(False)
ax_11.spines["right"].set_visible(False)
ax_11.spines["left"].set_visible(False)
ax_11.spines["bottom"].set_visible(False)
ax_11.tick_params(axis=u'both', which=u'both',length=0)
ax_11.set_axisbelow(True)
# ax_11.legend(['Ours', 'GeoTransformer', "RPMNet"], fancybox=True, shadow=False, frameon=True)

# results = np.loadtxt(file)[:,0]

ax_12 = plt.subplot(1,2,2)
rte_thres = 0.1
intervals = np.arange(0, rte_thres+0.01, rte_thres/3)
ecdf1_x, ecdf1_y = compute_curve(intervals, dict_ours['tsfm_lst'])
ecdf2_x, ecdf2_y= compute_curve(intervals, dict_baseline1['tsfm_lst'])
ecdf3_x, ecdf3_y= compute_curve(intervals, dict_baseline2['tsfm_lst'])
ecdf4_x, ecdf4_y= compute_curve(intervals, dict_baseline3['tsfm_lst'])
ecdf4_y[ecdf4_y<0] = 0
ecdf3_y[ecdf3_y<0] = 0

ax_12.plot(ecdf1_x*100, ecdf1_y*100, color= my_cmap(1), linewidth=3)
ax_12.plot(ecdf2_x*100, ecdf2_y*100, color= my_cmap(3), linewidth=3)
ax_12.plot(ecdf3_x*100, ecdf3_y*100, color= my_cmap(5), linewidth=3)
ax_12.plot(ecdf4_x*100, ecdf4_y*100, color= my_cmap(7), linewidth=3)

ax_12.set_ylabel('ECDF [%]')
ax_12.set_xlabel('RMSE [cm]')
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