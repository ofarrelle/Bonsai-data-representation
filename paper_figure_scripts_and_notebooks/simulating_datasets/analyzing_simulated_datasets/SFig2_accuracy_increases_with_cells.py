import os, sys
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from scipy.spatial import distance
import numpy as np
import csv
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.spatial.distance import squareform
import seaborn as sns
import logging

FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
logging.basicConfig(format=FORMAT, datefmt='%H:%M:%S',
                    level=log_level)

plt.set_loglevel(level='warning')
logging.getLogger("umap").disabled = True

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai.bonsai_helpers import str2bool, find_latest_tree_folder
from knn_recall_helpers import get_pdists_on_tree, Dataset, compare_pdists_to_truth_per_cell, \
    compare_pdists_to_truth_per_cell_adjusted, compare_nearest_neighbours_to_truth, do_pca, fit_umap, fit_phate

parser = ArgumentParser(
    description='Runs Bonsai on several simulated datasets.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_folder', type=str, default='data/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where simulated trees can be found.")
parser.add_argument('--results_folder', type=str, default='results/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where reconstructed trees can be found.")
parser.add_argument('--num_dims', type=str, default="100",
                    help="Number of dimensions in which we sample the cells.")
parser.add_argument('--n_sampled_clsts', type=str, default="100",
                    help="Number of clusters in star-tree.")
parser.add_argument('--n_cells_per_clst', type=str, default="1,2,5,10,20",
                    help="Number of cells per cluster.")
parser.add_argument('--random_times', type=str2bool, default=True,
                    help="Determine if branch lengths of true tree should be sampled uniform in a logspace between"
                         "0.1 and 10, instead of keeping them all at 1.")
parser.add_argument('--sample_umi_counts', type=str2bool, default=False,
                    help="Determines if we want to ensure the tqs add up to 1. Doesn't have to when we don't sample "
                         "counts but rather give the true data to Bonsai.")
parser.add_argument('--add_noise', type=str2bool, default=True,
                    help="Determines if we want to ensure the tqs add up to 1. Doesn't have to when we don't sample "
                         "counts but rather give the true data to Bonsai.")
parser.add_argument('--seed', type=int, default=1231,
                    help="Sets the random seed.")
parser.add_argument('--noise_var', type=float, default=2.5,
                    help="Determines the variance of the Gaussian noise from which we sample cells around cluster"
                         "Typical variance from cluster-centers to mean is 1.0.")
parser.add_argument('--recalculate', type=str2bool, default=False,
                    help="Determines whether we recalculate PCAs and UMAPs.")

args = parser.parse_args()
print(args)

"""--------------------Layout settings----------------------"""
SMALL_SIZE = 9
MEDIUM_SIZE = 11
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

"""Constants"""
RECALCULATE = args.recalculate
DO_OTHER_TOOLS = True
PCA_COMPS = [2, 10]

seed = args.seed

num_dims_list = [int(num_dim) for num_dim in args.num_dims.split(',')]
n_cells_per_clst_list = [int(ncpc) for ncpc in args.n_cells_per_clst.split(',')]
n_clsts = int(args.n_sampled_clsts)
noise_var = args.noise_var

ADD_NOISE = args.add_noise
noise_var = args.noise_var
CELL_DEPENDENT = False

if ADD_NOISE:
    if not CELL_DEPENDENT:
        add_noise = '_add_noise_{}'.format(int(noise_var))
    else:
        add_noise = '_add_noise_{}_celldependent'.format(int(noise_var))
else:
    add_noise = ''

fig, axs = plt.subplots(nrows=1, ncols=len(n_cells_per_clst_list), figsize=(10, 4), sharey=True)
if len(n_cells_per_clst_list) == 1:
    axs = [axs]
dist_objcts = []
# bonsai_dist_objcts = []
# umap_dist_objcts = []
# pca_dist_objcts = []

num_dims = num_dims_list[0]

for ind_dim, n_cells_per_clst in enumerate(n_cells_per_clst_list):
    n_cells = n_clsts * n_cells_per_clst

    datadir = "simulate_equidistant_{}_clsts_{}_cells_{}_dims".format(n_clsts, n_cells, num_dims)
    if args.random_times:
        datadir += '_random_times'
    if not args.sample_umi_counts:
        datadir += '_no_umi_counts'
    if args.add_noise:
        datadir += '_add_noise_{}'.format(int(noise_var))
    datadir += '_seed_{}'.format(seed)
    dataset = os.path.join(args.input_folder, datadir)
    data_path = os.path.abspath(os.path.join(args.input_folder, datadir))
    results_path = os.path.abspath(os.path.join(args.results_folder, datadir))

    subset_cells = np.arange(0, n_cells, n_cells_per_clst, dtype=int)
    args.input_simulated_dataset = data_path
    args.bonsai_results = os.path.join(results_path, find_latest_tree_folder(results_folder=results_path))
    args.output_folder = os.path.join('useful_scripts_not_bonsai/simulating_datasets/analyzing_simulated_datasets/results', dataset)
    # args.input_simulated_dataset = 'data/simulated_datasets/simulated_equidistant_clean/simulate_equidistant_{}_clsts_{}_cells_{}_dims_random_times_no_umi_counts_add_noise_2_seed_1231'.format(
    #     n_clsts, n_cells, num_dims)
    # args.bonsai_results = 'results/simulated_datasets/simulated_equidistant_clean/simulate_equidistant_{}_clsts_{}_cells_{}_dims_random_times_no_umi_counts_add_noise_2_seed_1231/final_bonsai_zscore1.0'.format(
    #     n_clsts, n_cells, num_dims)
    # args.results_folder = 'useful_scripts_not_bonsai/simulating_datasets/analyzing_simulated_datasets/results/simulated_datasets/simulated_equidistant_clean/simulate_equidistant_{}_clsts_{}_cells_{}_dims_random_times_no_umi_counts_add_noise_2_seed_1231'.format(
    #     n_clsts, n_cells, num_dims)
    Path(os.path.join(args.output_folder, 'intermediate_files')).mkdir(parents=True, exist_ok=True)

    # args.input_simulated_dataset = 'data/simulated_datasets/simulated_equidistant/simulate_equidistant_{}_clsts_{}_cells_{}_dims_random_times_no_umi_counts_add_noise_seed_1231'.format(
    #     n_clsts, n_cells, num_dims)
    # args.bonsai_results = 'results/simulated_datasets/simulated_equidistant/simulate_equidistant_{}_clsts_{}_cells_{}_dims_random_times_no_umi_counts_add_noise_seed_1231/final_bonsai_zscore1.0'.format(
    #     n_clsts, n_cells, num_dims)
    # args.results_folder = 'useful_scripts_not_bonsai/simulating_datasets/analyzing_simulated_datasets/results/simulated_datasets/simulated_equidistant/simulate_equidistant_{}_clsts_{}_cells_{}_dims_random_times_no_umi_counts_add_noise_seed_1231'.format(
    #     n_clsts, n_cells, num_dims)
    # Path(os.path.join(args.results_folder, 'intermediate_files')).mkdir(parents=True, exist_ok=True)


    # Get cell-ids
    if (not RECALCULATE) and os.path.exists(os.path.join(args.output_folder, 'intermediate_files', 'cellID.txt')):
        cell_ids = []
        with open(os.path.join(args.output_folder, 'intermediate_files', 'cellID.txt'), 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                cell_ids.append(row[0])
    else:
        try:
            umi_counts_df = pd.read_csv(os.path.join(args.input_simulated_dataset, 'Gene_table.txt'), header=0,
                                        index_col=0,
                                        sep='\t')
            cell_ids = list(umi_counts_df.columns)
        except FileNotFoundError:
            cell_ids = []
            with open(os.path.join(args.input_simulated_dataset, 'cellID.txt'), 'r') as file:
                reader = csv.reader(file, delimiter="\t")
                for row in reader:
                    cell_ids.append(row[0])

    # Calculate pairwise distances for ground truth
    if RECALCULATE or (
            not os.path.exists(os.path.join(args.output_folder, 'intermediate_files', 'delta_true_pdists.npy'))):
        delta_gc_true = pd.read_csv(os.path.join(args.input_simulated_dataset, 'delta_true.txt'), header=None,
                                    index_col=None,
                                    sep='\t')
        delta_gc_true = delta_gc_true.iloc[:, subset_cells]
        true_dists = distance.pdist(delta_gc_true.T, metric='sqeuclidean') / num_dims
        np.save(os.path.join(args.output_folder, 'intermediate_files', 'delta_true_pdists.npy'), true_dists,
                allow_pickle=False)

    # Calculate pairwise distances for Bonsai.
    if RECALCULATE or (
            not os.path.exists(os.path.join(args.output_folder, 'intermediate_files', 'bonsai_pdists.npy'))):
        bonsai_dists = get_pdists_on_tree(os.path.join(args.bonsai_results, 'tree.nwk'), cell_ids)
        bonsai_dists = squareform(squareform(bonsai_dists)[subset_cells, :][:, subset_cells])
        np.save(os.path.join(args.output_folder, 'intermediate_files', 'bonsai_pdists.npy'), bonsai_dists,
                allow_pickle=False)

    if DO_OTHER_TOOLS:
        # Perform PCA-projection.
        all_pca_files = [os.path.basename(filepath) for filepath in
                         Path(os.path.join(args.output_folder, 'intermediate_files')).glob('pca_*.npy')]
        all_pca_files = [filepath for filepath in all_pca_files if 'pdists' not in filepath]
        if (not RECALCULATE) and len(all_pca_files):
            pca_projected = {}
            for pca_file in all_pca_files:
                n_comps = int(pca_file.split('pca_')[1].split('.npy')[0])
                pca_projected[n_comps] = np.load(os.path.join(args.output_folder, 'intermediate_files', pca_file),
                                                 allow_pickle=False)
        else:
            delta_gc = pd.read_csv(os.path.join(args.input_simulated_dataset, 'delta.txt'), header=None,
                                   index_col=None,
                                   sep='\t')
            delta_gc = delta_gc.iloc[:, subset_cells]
            pca_projected = do_pca(delta_gc, n_comps_list=PCA_COMPS)
            for n_comps, pca_proj in pca_projected.items():
                np.save(os.path.join(args.output_folder, 'intermediate_files', 'pca_{}.npy'.format(n_comps)), pca_proj,
                        allow_pickle=False)

        # Perform UMAP.
        all_umap_files = [os.path.basename(filepath) for filepath in
                          Path(os.path.join(args.output_folder, 'intermediate_files')).glob('umap_*.npy')]
        all_umap_files = [filepath for filepath in all_umap_files if 'pdists' not in filepath]
        if (not RECALCULATE) and len(all_umap_files):
            umap_projected = {}
            for umap_file in all_umap_files:
                n_comps = int(umap_file.split('umap_')[1].split('.npy')[0])
                umap_projected[n_comps] = np.load(os.path.join(args.output_folder, 'intermediate_files', umap_file),
                                                  allow_pickle=False)
        else:
            umap_projected = {}
            for n_comps, pca_proj in pca_projected.items():
                if n_comps == 2:
                    continue
                umap_projected[n_comps] = fit_umap(pca_proj, random_state=None, n_neighbors=15, min_dist=0.1,
                                                   n_components=2,
                                                   metric='euclidean',
                                                   make_plot=False, title='')
                np.save(os.path.join(args.output_folder, 'intermediate_files', 'umap_{}.npy'.format(n_comps)),
                        umap_projected[n_comps],
                        allow_pickle=False)

        # Perform PHATE.
        all_phate_files = [os.path.basename(filepath) for filepath in
                          Path(os.path.join(args.output_folder, 'intermediate_files')).glob('phate*.npy')]
        all_phate_files = [filepath for filepath in all_phate_files if 'pdists' not in filepath]
        if (not RECALCULATE) and len(all_phate_files):
            phate_projected = {}
            for phate_file in all_phate_files:
                # n_comps = int(phate_file.split('phate_')[1].split('.npy')[0])
                phate_projected['all'] = np.load(os.path.join(args.output_folder, 'intermediate_files', phate_file),
                                                  allow_pickle=False)
        else:
            phate_projected = {}
            phate_projected['all'] = fit_phate(delta_gc)
            np.save(os.path.join(args.output_folder, 'intermediate_files', 'phate.npy'),
                    phate_projected['all'],
                    allow_pickle=False)

        # Calculate pairwise distances for 2D-PCA, UMAP
        all_pca_dist_files = [os.path.basename(filepath) for filepath in
                              Path(os.path.join(args.output_folder, 'intermediate_files')).glob('pca_*_pdists.npy')]
        if RECALCULATE or (not len(all_pca_dist_files)):
            for n_comps, pca_proj in pca_projected.items():
                if n_comps != 2:
                    continue
                pca_dists = distance.pdist(pca_proj.T, metric='sqeuclidean') / 2
                np.save(os.path.join(args.output_folder, 'intermediate_files', 'pca_{}_pdists.npy'.format(n_comps)),
                        pca_dists,
                        allow_pickle=False)

        all_umap_dist_files = [os.path.basename(filepath) for filepath in
                               Path(os.path.join(args.output_folder, 'intermediate_files')).glob('umap_*_pdists.npy')]
        if RECALCULATE or (not len(all_umap_dist_files)):
            for n_comps, umap_proj in umap_projected.items():
                umap_dists = distance.pdist(umap_proj.T, metric='sqeuclidean') / 2
                np.save(os.path.join(args.output_folder, 'intermediate_files', 'umap_{}_pdists.npy'.format(n_comps)),
                        umap_dists, allow_pickle=False)

        all_phate_dist_files = [os.path.basename(filepath) for filepath in
                               Path(os.path.join(args.output_folder, 'intermediate_files')).glob('phate_pdists.npy')]
        if RECALCULATE or (not len(all_phate_dist_files)):
            for n_comps, phate_proj in phate_projected.items():
                phate_dists = distance.pdist(phate_proj.T, metric='sqeuclidean') / 2
                np.save(os.path.join(args.output_folder, 'intermediate_files', 'phate_pdists.npy'),
                        phate_dists, allow_pickle=False)

    if DO_OTHER_TOOLS:
        alldistfiles = list(Path(os.path.join(args.output_folder, 'intermediate_files')).glob('*_pdists.npy'))
    else:
        alldistfiles = list(Path(os.path.join(args.output_folder, 'intermediate_files')).glob('*bonsai*_pdists.npy'))
        alldistfiles += list(Path(os.path.join(args.output_folder, 'intermediate_files')).glob('*true*_pdists.npy'))

    alldistfiles = natsorted(alldistfiles)

    datasets = []
    for distfile in alldistfiles:
        data_type = os.path.basename(distfile).split("_pdists")[0]
        data_id = data_type + ' {}_dims'.format(num_dims)
        datasets.append(
            Dataset(pdist_file=distfile, data_type=data_type, data_id=data_id, color_types=['sanity', 'bonsai', 'pca', 'umap', 'phate']))
        datasets[-1].n_cells_per_clst = n_cells_per_clst
        if data_type[:10] != 'delta_true':
            dist_objcts.append(datasets[-1])
        # if data_type == 'bonsai':
        #     bonsai_dist_objcts.append(datasets[-1])
        # elif data_type[:4] == 'umap':
        #     umap_dist_objcts.append(datasets[-1])
        # elif data_type[:3] == 'pca':
        #     pca_dist_objcts.append(datasets[-1])

    axs_col = axs[ind_dim]
    # avg_rel_diff_list = compare_pdists_to_truth(dataset_subset, make_fig=True, axs=axs_col, axs2=axs2, axs3=axs3,
    #                                             YLABEL=ind_dim == 0,
    #                                             first_title='{}-cells per cluster'.format(n_cells_per_clst))
    avg_rel_diff_list, R_vals = compare_pdists_to_truth_per_cell(datasets, make_fig=True, axs=axs_col,
                                                                          set_lims=False,
                                                                          return_Rvals=True, flip_axes=False,
                                                                          YLABEL=ind_dim == 0, share_y=True,
                                                                          first_title='{}-cells per cluster'.format(
                                                                              n_cells_per_clst), density=True, bins=50,
                                                                          XLABEL=ind_dim == 0, loglog_corr=False)

    # n_neighbours_list = compare_nearest_neighbours_to_truth_adjusted(datasets, make_fig=False, max_neighbours=90,
    #                                                                  only_powers_of_2=False)

    # avg_rel_diff_list = compare_pdists_to_truth_per_cell(dataset_subset, make_fig=True, axs=axs_col,
    #                                             YLABEL=ind_dim == 0, set_lims=False,
    #                                             first_title='{}-cells per cluster'.format(n_cells_per_clst))
    # avg_rel_diffs.append(avg_rel_diff_list[0])

# """Create boxplots for relative errors"""
# boxprops = dict(linewidth=0.05)
# flierprops = dict(markersize=2, markeredgewidth=0.5)
# medianprops = dict(color='black', linewidth=1)
# fig_bp, axs_bp = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
#
# avg_rel_errors = []
# dataset_names = []
# for ind_dataset, bonsai_dist_objct in enumerate(bonsai_dist_objcts):
#     dataset_names.append("{}-cells per cluster".format(ns_cells_per_clst[ind_dataset]))
#     avg_rel_errors.append(bonsai_dist_objct.avg_rel_errors)
# ax = axs_bp
# bplot = ax.boxplot(avg_rel_errors, whis=(5, 95), labels=dataset_names, patch_artist=True, flierprops=flierprops,
#                    medianprops=medianprops, boxprops=boxprops)
# # fill with colors
# for ind_patch, patch in enumerate(bplot['boxes']):
#     patch.set_facecolor(color=bonsai_dist_objct.data_type_color)
# ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
# ax.set_ylim(-0.05, 1.05)
# ax.set_ylabel("Mean relative errors between\ntrue and inferred distances\nof each cell to all others")
# plt.tight_layout()

pearsonRSqs = []
logRatios = []
# pearsonRSqs_bnsi = []
# pearsonRSqs_umap = []
# pearsonRSqs_pca = []
dataset_names = []
plot_colors = []
ncpcs = []
for ind_dataset, dist_objct in enumerate(dist_objcts):
    data_type = dist_objct.data_type
    if data_type[:4] == 'umap':
        data_type = 'UMAP'
    elif data_type[:3] == 'pca':
        data_type = 'PCA'
    elif data_type[:5] == 'phate':
        data_type = 'PHATE'
    # dataset_names.append("{}\n{}-cell-clusters".format(data_type, dist_objct.n_cells_per_clst))
    dataset_names.append("{}".format(data_type))
    ncpcs.append(dist_objct.n_cells_per_clst)
    pearsonRSqs.append(dist_objct.pearsonRs ** 2)
    logRatios.append(-np.log10(1-dist_objct.pearsonRs ** 2))
    plot_colors.append(dist_objct.data_type_color)

combined = list(zip(dataset_names, pearsonRSqs, logRatios, plot_colors, ncpcs))
combined_sorted = sorted(combined, key=lambda x: x[0].lower())
dataset_names_sorted, pearsonRSqs_sorted, logRatios_sorted, plot_colors_sorted, ncpcs_sorted = zip(*combined_sorted)
dataset_names_sorted = [(ds_name.capitalize() if ds_name == 'bonsai' else ds_name) for ds_name in dataset_names_sorted]
n_boxes = len(n_cells_per_clst_list)

"""Create boxplots for Pearson R-values."""
RSQ = True

boxprops = dict(linewidth=0.05)
flierprops = dict(markersize=2, markeredgewidth=0.5)
medianprops = dict(color='black', linewidth=1)
fig_bp, axs_bp = plt.subplots(figsize=(13.5, 7.5), nrows=1, ncols=1)
ax = axs_bp
if RSQ:
    bplot = ax.boxplot(pearsonRSqs_sorted, whis=(5, 95), labels=ncpcs_sorted, patch_artist=True, flierprops=flierprops,
                       medianprops=medianprops, boxprops=boxprops)
else:
    bplot = ax.boxplot(logRatios_sorted, whis=(5, 95), labels=ncpcs_sorted, patch_artist=True, flierprops=flierprops,
                       medianprops=medianprops, boxprops=boxprops)
# fill with colors
for ind_patch, patch in enumerate(bplot['boxes']):
    patch.set_facecolor(color=plot_colors_sorted[ind_patch])
    if ind_patch % n_boxes == int(n_boxes/2):
        ax.text(ind_patch + 1, 0.95, dataset_names_sorted[ind_patch], ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontsize=14)

# ax.set_xticks(xticks, xtick_labels, rotation=45, ha='right')
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_xlabel("Number of cells in cluster")

if RSQ:
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r'Pearson $R^2$-values between' + "\ntrue and inferred distances\nof each cell to all others")
else:
    ax.set_ylabel(r'-log10(1-Pearson $R^2$) for $R^2$ between' + "\ntrue and inferred distances\nof each cell to all others")
plt.tight_layout()

# base_folder = '/Users/Daan/Documents/postdoc/bonsai-development/data/simulated_datasets/simulated_equidistant_clean'
base_folder = os.path.join('useful_scripts_not_bonsai/simulating_datasets/analyzing_simulated_datasets/results', args.input_folder)

plt.savefig(os.path.join(base_folder, "SI_tree_better_at_more_cells_new.svg"))
plt.savefig(os.path.join(base_folder, "SI_tree_better_at_more_cells_new.png"), dpi=300)

print("Stored the png-figure at {}".format(os.path.join(base_folder, "SI_tree_better_at_more_cells_new.png")))

"""Also make a violin-plot"""
# data = {dataset_names[ind]: pearsonR for ind, pearsonR in enumerate(pearsonRSqs)}
# df = pd.DataFrame(data)
# df_melted = df.melt(var_name='n_cells', value_name='PearsonR')
# fig_violin, ax_violin = plt.subplots(figsize=(12, 6))
# sns.violinplot(x='n_cells', y='PearsonR', data=df_melted, hue='n_cells', palette=plot_colors)
# # sns.stripplot(x='n_cells', y='PearsonR', data=df_melted, color='k', alpha=0.6, jitter=True)
# plt.xlabel("")
# plt.ylabel("Pearson R^2-values between\ntrue and inferred distances\nof each cell to all others")
# plt.ylim(0, 1.2)

"""And a strip-plot"""
# fig_strip, ax_strip = plt.subplots(figsize=(12, 6))
# sns.stripplot(x='n_cells', y='PearsonR', data=df_melted, hue='n_cells', palette=plot_colors, alpha=0.6, jitter=True)
# plt.xlabel("")
# plt.ylabel("Pearson R^2-values between\ntrue and inferred distances\nof each cell to all others")
# plt.ylim(0, 1.0)

"""Add a plot for getting kNN-comparison"""
# marker = '*-' if len(n_neighbours_list) < 40 else '-'
# fig_knn, ax_knn = plt.subplots(figsize=(6, 6))
# for ind_dataset, bonsai_dist_objct in enumerate(bonsai_dist_objcts):
#     label = "{}-cells per cluster".format(ns_cells_per_clst[ind_dataset])
#     plot_color = bonsai_dist_objct.data_type_color
#     correct_nns = bonsai_dist_objct.correct_fractions_of_neighbours
#     ax_knn.plot(n_neighbours_list, correct_nns, marker, label=label)
#
# box = ax_knn.get_position()
# ax_knn.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax_knn.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax_knn.set_xlabel('Number of nearest neighbours')
# ax_knn.set_xscale('log')
# ax_knn.set_ylabel('Fraction of correct nearest neighbours')

plt.tight_layout()
plt.show()
