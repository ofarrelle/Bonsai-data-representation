import os, sys
import pandas as pd
import numpy as np
import umap.umap_ as umap
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from bonsai_scout.my_tree_layout import Layout_Tree
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
import logging
import phate
from dtne import DTNE
from pathlib import Path
import subprocess
import glob

FORMAT = '%(asctime)s %(name)s %(funcName)s %(message)s'
log_level = logging.WARNING
logging.basicConfig(format=FORMAT, datefmt='%H:%M:%S',
                    level=log_level)
plt.set_loglevel(level='warning')
logging.getLogger('umap').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


DATA_FOLDER = os.getcwd()
RAW_INPUT_FOLDER = 'Sim_Baron_N_random_walk_from_random_cell'
raw_folder = os.path.join(DATA_FOLDER, RAW_INPUT_FOLDER)
# custom_colors = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']
dataset_inds = list(np.arange(0, 100))
# custom_colors = plt.get_cmap('tab10', 10).colors
custom_colors = plt.get_cmap('Set2', 9)
family_colors = ['Greens', 'Blues', 'Oranges', 'Purples', 'Wistia', 'pink', 'bone', 'winter', 'summer', 'autumn']
custom_colors_families = [plt.get_cmap(cmap, 5) for cmap in family_colors]
family_counter_dict = {ind: list(np.arange(2, 5)) for ind in range(100)}


def do_logp1(counts):
    avgTotCount = np.sum(counts) / counts.shape[1]
    logp1counts = np.log(counts / (np.sum(counts, axis=0) / avgTotCount) + 1)
    return logp1counts


def do_pca(data, n_comps_list=[50]):
    """

    :param data: should be a numpy array with features (genes) as rows, observations (cells) as columns
    :param n_comps: Should be a list of integers indicating for what numbers of components, PCA should be done
    :return:
    """
    data_T = data.T
    n, m = data_T.shape
    pca_centers = data_T.mean(axis=0)
    data_cd = data_T - pca_centers

    U, S, Vh = np.linalg.svd(data_cd, full_matrices=False)
    transformed_data = np.matmul(U, np.diag(S))

    pca_projected = {}
    for n_comps in n_comps_list:
        proj_data = transformed_data[:, :n_comps].T
        pca_projected[n_comps] = proj_data
    return pca_projected


def fit_phate(data):
    """

    :param data: should be a numpy array with features (genes) as rows, observations (cells) as columns
    :return:
    """
    data_T = data.T
    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(data_T)

    return data_phate[:, :2].T


def fit_DTNE(data):
    """

    :param data: should be a numpy array with features (genes) as rows, observations (cells) as columns
    :return:
    """
    data_T = data.T
    dtne_operator = DTNE()
    Y = dtne_operator.fit_transform(data_T)
    return Y.T


def fit_umap(data, random_state=42, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',
             make_plot=False, title=''):
    """
    :param data: data: should be a numpy array with features (genes) as rows, observations (cells) as columns

    UMAP Parameters
    :param random_state:
    :param n_neighbors:
    :param min_dist:
    :param n_components:
    :param metric:

    Whether to make a umap-plot and what the title should be
    :param make_plot:
    :param title:
    :return:
    """
    fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                    random_state=random_state)
    u = fit.fit_transform(data.T)

    if make_plot:
        fig, ax = plt.subplots()
        ax.scatter(u[:, 0], u[:, 1])
        plt.title(title, fontsize=18)
    return u.T


def fit_poincare_map(data, path_to_data=None, path_to_poincaremap=None):
    """

    :param data: should be a numpy array with features (genes) as rows, observations (cells) as columns
    :return:
    """
    if path_to_poincaremap is None:
        exit("You should give a directory-path to the Poincare-map GitHub folder that is cloned locally.")

    # Store the data somewhere
    if path_to_data is None:
        path_to_data = 'tmp'
    Path(path_to_data).mkdir(parents=True, exist_ok=True)

    data_df = pd.DataFrame(data.T, columns=['Gene_{}'.format(ind) for ind in range(data.shape[0])])
    # data_df['labels'] = ['same'] * data.shape[1]
    data_df.to_csv(os.path.join(path_to_data, 'input_data.csv'), index=False)

    # Set output path
    output_path = path_to_data

    # Call the Poincare map program with subprocess
    command1 = [os.path.join(path_to_poincaremap, 'main.py'), '--dset', 'input_data', '--path', path_to_data + '/', '--dest', path_to_data + '/', '--cuda', '0', '--labels', '0']
    output1 = subprocess.run([sys.executable] + command1, stdout=subprocess.PIPE, text=True)
    print(output1.stdout)
    print(output1.stderr)

    # Read in the resulting coordinates
    # Find filename
    result_file = glob.glob(os.path.join(path_to_data, 'input_data_PM*.csv'))[0]
    embedding_coords = pd.read_csv(result_file, header=None, index_col=None).values

    # Return the coordinates in the right format
    return embedding_coords.T


def plot_poincare_map(embedding, colors, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    circle = plt.Circle((0, 0), radius=1, fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, 'x', c=(0, 0, 0), ms=5)

    ax.scatter(embedding[0, :], embedding[1, :], s=5, c=colors)
    ax.plot(0, 0, 'x', c=(1, 1, 1), ms=5)
    ax.axis('off')
    ax.set_aspect('equal')  # instead of plt.axis('equal')


def get_pdists_on_tree(nwk_file, cell_ids):
    # Read in newick file
    tree = Layout_Tree()

    with open(nwk_file, "r") as f:
        nwk_str = f.readline()

    tree.from_newick(nwk_str=nwk_str)

    # Renumber vert_inds on tree such that they are in line with a depth-first search
    vertIndToNode, tree.nNodes = tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
    tree.vert_ind_to_node = vertIndToNode
    tree.root.storeParent()

    # Get pairwise distances
    pdists = get_pairwise_dist_on_tree(tree, cell_ids)
    return pdists


def get_pairwise_dist_on_tree(tree, node_ids_of_interest):
    node_id_to_vert_ind = {node.nodeId: vert_ind for vert_ind, node in tree.vert_ind_to_node.items()}

    # Get indices between which distances are calculated
    indices = [node_id_to_vert_ind[node_id] for node_id in node_ids_of_interest]
    # edge_dict has keys source, target, dist, source_ind, target_ind
    edge_dict = tree.get_edge_dict(nodeIdToVertInd=node_id_to_vert_ind)

    cols = np.array(edge_dict['source_ind'])
    rows = np.array(edge_dict['target_ind'])
    weights = np.array(edge_dict['dist'])

    colsComplete = np.concatenate((cols, rows))
    rowsComplete = np.concatenate((rows, cols))
    weightsComplete = np.concatenate((weights, weights))
    nVerts = np.max(colsComplete) + 1
    distance_csr = csr_matrix((weightsComplete, (rowsComplete, colsComplete)), shape=(nVerts, nVerts))

    distances = squareform(shortest_path(distance_csr, method='auto', directed=False, return_predecessors=False,
                                         unweighted=False, overwrite=False, indices=indices)[:, indices], checks=False)
    return distances


class Dataset:
    umi_table = None
    distances = None
    deltas = None
    ltqs = None
    gene_variances = None
    neighbours = {}
    n_genes = None
    n_cells = None
    tree_parents = None
    nearest_neighbour_mat = None
    data_type = None
    data_type_color = None
    sorted_genes = None
    sorted_inds = None
    correct_fractions_of_neighbours = None
    dataset_ind = None
    data_id = None
    true_dataset_ranks = None
    pearsonRs = None

    def __init__(self, pdist_file=None, data_type='delta_true', data_families=None, data_id=None, color_types=None,
                 distances=None):
        if data_families is None:
            global dataset_inds
            dataset_inds = list(np.arange(0, 100))
        self.data_type = data_type
        if data_id is None:
            self.data_id = data_type
        else:
            self.data_id = data_id
        if data_type == 'delta_true':
            self.dataset_ind = -1
        else:
            self.dataset_ind = dataset_inds.pop(0)
            if data_families is not None:
                data_family = data_type.split('_')[0]
                try:
                    fam_ind = data_families.index(data_family)
                    fam_counter = family_counter_dict[fam_ind].pop(0)
                    self.data_type_color = custom_colors_families[fam_ind](fam_counter)
                except ValueError:
                    self.data_type_color = custom_colors(self.dataset_ind)
            elif color_types is not None:
                color_type = data_type.split('_')[0]
                try:
                    color_ind = color_types.index(color_type)
                    self.data_type_color = custom_colors(color_ind)
                except ValueError:
                    self.data_type_color = custom_colors(self.dataset_ind)
            else:
                self.data_type_color = custom_colors(self.dataset_ind)
        # Distances are stored in one column:
        # [D(cell1,cell2), D(cell1,cell3), ..., D(cell1,cellN), D(cell2,cell3),...,D(cellN-1,cellN)]^T
        if pdist_file is None:
            self.distances = distances
        else:
            self.distances = np.load(pdist_file, allow_pickle=False)
        print("Finished loading dataset of type " + self.data_type)

    def create_distance_mat_from_vect(self):
        if self.distances is None:
            print("To create the distance matrix we need distances stored as a vector in self.distances, like:")
            print("[D(cell1,cell2), D(cell1,cell3), ..., D(cell1,cellN), D(cell2,cell3),...,D(cellN-1,cellN)]^T")

        distance_mat = squareform(self.distances, force='tomatrix')
        return distance_mat

    def create_nearest_neighbour_mat(self):
        """
        Creates matrix with in column i the indices of all cells ordered by their distance from cell i.
        First row must thus be i in column i, since cells are always closest to themselves.
        """
        distance_mat = self.create_distance_mat_from_vect()
        self.nearest_neighbour_mat = np.argsort(distance_mat, axis=0)
        # test = np.take_along_axis(distance_mat, self.nearest_neighbour_mat, axis=0)

    def histogram_gene_variances(self):
        fig, ax = plt.subplots()
        # the histogram of the data
        n, bins, patches = ax.hist(self.gene_variances, 50, density=1, facecolor=self.data_type_color, alpha=0.75)

        ax.set_xlabel('Gene variance')
        ax.set_ylabel('Frequency')
        # plt.axis([40, 160, 0, 0.03])
        ax.grid(True)

    def get_variability_sorted_gene_list(self):
        self.sorted_inds = np.argsort(1 - self.gene_variances)
        self.sorted_genes = np.array(self.gene_names)[self.sorted_inds]


def compare_gene_sortings(data_true, data_base, data_alt, genes_to_show=2, fig=None, axs=None):
    # We first order the genes decreasing in the following function: (1/rank[base])*(rank[base]-rank[alt])
    # The first of these lists captures which gene is much more variable according to base than to alt
    difference_gene_ranking_base_vs_alt = np.zeros(data_base.n_genes)
    # And vice versa
    difference_gene_ranking_alt_vs_base = np.zeros(data_base.n_genes)
    for ind, gene in enumerate(data_base.sorted_genes):
        rank_alt = np.where(data_alt.sorted_genes == gene)[0]
        if len(rank_alt) != 1:
            print("Gene " + gene + " is not present in second data")
            exit()
        else:
            rank_base = ind + 1
            rank_alt = rank_alt[0] + 1
        difference_gene_ranking_base_vs_alt[ind] = (1 / rank_base) * (rank_alt - rank_base)

    most_more_var_genes_in_base = data_base.sorted_genes[np.argsort(1 - difference_gene_ranking_base_vs_alt)]

    if fig is None:
        fig, axs = plt.subplots(genes_to_show, 3)
    for ind in range(genes_to_show):
        gene = most_more_var_genes_in_base[ind]
        rank_base = np.where(data_base.sorted_genes == gene)[0]
        rank_alt = np.where(data_alt.sorted_genes == gene)[0]
        gene_ind_true = [ind for ind, gene_i in enumerate(data_true.gene_names) if gene_i == gene][0]
        gene_ind_base = [ind for ind, gene_i in enumerate(data_base.gene_names) if gene_i == gene][0]
        gene_ind_alt = [ind for ind, gene_i in enumerate(data_alt.gene_names) if gene_i == gene][0]
        X_true = data_true.umi_table[gene_ind_true, :]
        X_base = data_base.deltas[gene_ind_base, :]
        X_alt = data_alt.deltas[gene_ind_alt, :]
        axs[ind, 0].hist(X_true, 50, density=1, facecolor=data_true.data_type_color, alpha=0.75)
        axs[ind, 1].hist(X_base, 50, density=1, facecolor=data_base.data_type_color, alpha=0.75)
        axs[ind, 2].hist(X_alt, 50, density=1, facecolor=data_alt.data_type_color, alpha=0.75)

        if ind == 0:
            axs[ind, 0].set_title("Raw counts")
            axs[ind, 1].set_title(data_base.data_type)
            axs[ind, 2].set_title(data_alt.data_type)

        for hist_ind in range(3):
            axs[ind, hist_ind].set_xlabel('Transcription strength')
            if hist_ind == 0:
                axs[ind, hist_ind].set_ylabel(gene + '\n Rank ' + str(rank_base[0]) + ' vs ' + str(rank_alt[0]))

    fig.suptitle(
        "Genes with much higher variability \n according to " + data_base.data_type + " than to " + data_alt.data_type)


from scipy.stats import rankdata
from scipy.spatial.distance import squareform


def distance_to_rank_matrix(distance_matrix):
    n = distance_matrix.shape[0]
    # rank_matrix = np.zeros_like(distance_matrix, dtype=int)

    # for i in range(n):
    # Rank distances in row i; method='min' gives tied values the same lowest rank
    # `rankdata` ranks in ascending order, so smaller distance = higher rank
    # rank_matrix[i] = rankdata(distance_matrix[i], method='min')
    rank_matrix = rankdata(distance_matrix, method='min', axis=0)
    return rank_matrix


def compare_nearest_neighbours_to_truth(datasets, make_fig=True, max_neighbours=100, ax=None,
                                        only_powers_of_2=False,
                                        title=''):
    true_dataset_ind = [ind for ind, dataset in enumerate(datasets) if dataset.data_type == 'delta_true']
    if len(true_dataset_ind) != 1:
        print("Exactly one of the provided dataset should contain the true nearest neighbours")
        exit()
    else:
        true_dataset_ind = true_dataset_ind[0]
        true_dataset = datasets[true_dataset_ind]
        # if true_dataset.nearest_neighbour_mat is None:
        # true_dataset.create_nearest_neighbour_mat()
        # nn_mat_true = true_dataset.nearest_neighbour_mat[1:, :]
        if true_dataset.true_dataset_ranks is None:
            true_dataset.true_dataset_ranks = rankdata(squareform(true_dataset.distances), method='min', axis=0)

    n_cells = true_dataset.true_dataset_ranks.shape[0]
    cols_idx = np.arange(n_cells)[None, :]
    if only_powers_of_2:
        n_steps = int(np.log2(max_neighbours))
        n_neighbours_list = [2 ** ind - 1 for ind in range(1, n_steps + 1)]
    else:
        n_steps = max_neighbours
        n_neighbours_list = np.arange(1, max_neighbours + 1)

    correct_fractions = np.zeros((n_steps, len(datasets)))
    for ind_dataset, dataset in enumerate(datasets):
        # print(dataset.data_type)
        if ind_dataset == true_dataset_ind:
            continue
        print("Comparing nearest neighbours of dataset: '" + dataset.data_type + "'.")
        if dataset.correct_fractions_of_neighbours is None:
            if dataset.nearest_neighbour_mat is None:
                dataset.create_nearest_neighbour_mat()
            nn_mat_data = dataset.nearest_neighbour_mat[1:, :]
            for ind_step, n_neighbours in enumerate(n_neighbours_list):
                n_correct = 0
                # true_neighbours = nn_mat_true[:n_neighbours, :]
                data_neighbours = nn_mat_data[:n_neighbours, :]
                # true_ranks = true_dataset_ranks[data_neighbours]
                true_ranks = true_dataset.true_dataset_ranks[data_neighbours, cols_idx]
                # print(true_ranks[:5,:])
                # A neighbor is correct, if the rank is smaller than n_neighbours + 1
                n_correct = np.sum(true_ranks <= n_neighbours + 1)

                # for cell in range(n_neighbours):
                # n_correct += np.sum((data_neighbours - true_neighbours[cell, :]) == 0)
                correct_fractions[ind_step, ind_dataset] = n_correct / (n_neighbours * n_cells)
            dataset.correct_fractions_of_neighbours = correct_fractions[:, ind_dataset]

    if make_fig:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        # x = np.arange(1, max_neighbours)
        marker = '*-' if n_steps < 40 else '-'
        for dataset in datasets:
            if not dataset.data_type == 'delta_true':
                fam = dataset.data_type.split('_')[0]
                if fam != 'bonsai':
                    ax.plot(n_neighbours_list, dataset.correct_fractions_of_neighbours, marker,
                            c=dataset.data_type_color,
                            label=dataset.data_type, zorder=0)
                else:
                    ax.plot(n_neighbours_list, dataset.correct_fractions_of_neighbours, marker,
                            c=dataset.data_type_color, linewidth=3, label=dataset.data_type, zorder=1)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Number of nearest neighbours')
        ax.set_xscale('log')
        ax.set_ylabel('Fraction of correct nearest neighbours')
        ax.set_title(title)
    return n_neighbours_list


def compare_pdists_to_truth(datasets, make_fig=True, axs=None, axs2=None, axs3=None, title='', YLABEL=True, XLABEL=True,
                            first_title=None, loglog_corr=True,
                            set_lims=True, corr_measure='Rsq', return_Rvals=False, flip_axes=False):
    if make_fig and (axs is None):
        n_datasets = len(datasets) - 1  # Minus 1 to subtract for true dataset
        n_rows = int(np.ceil(np.sqrt(n_datasets)))
        n_cols = int(np.ceil(n_datasets / n_rows))
        fig, axs = plt.subplots(n_rows=n_rows, n_cols=n_cols, figsize=(8, 6))

    true_dataset_ind = [ind for ind, dataset in enumerate(datasets) if dataset.data_type == 'delta_true']
    if len(true_dataset_ind) != 1:
        print("Exactly one of the provided dataset should contain the true nearest neighbours")
        exit()
    else:
        true_dataset_ind = true_dataset_ind[0]
        true_dataset = datasets[true_dataset_ind]
        pdists_true = true_dataset.distances

    # n_cells = nn_mat_true.shape[0] + 1
    dataset_counter = -1
    avg_rel_diffs = []
    R_vals = {}
    for ind_dataset, dataset in enumerate(datasets):
        if ind_dataset == true_dataset_ind:
            continue
        dataset_counter += 1
        try:
            ax = axs.flatten()[dataset_counter]
            fontsize = 'x-small' if (len(axs.flatten()) > 4) else 'large'
        except AttributeError:
            ax = axs
            fontsize = 'x-small'
        try:
            ax2 = axs2.flatten()[dataset_counter]
            ax3 = axs3.flatten()[dataset_counter]
        except AttributeError:
            ax2 = axs2
            ax3 = axs3
        fontsize = 'large'
        print("Comparing pairwise distances of dataset: '" + dataset.data_type + "'.")
        pdists_data = dataset.distances
        if loglog_corr:
            log_truedists = np.log(pdists_true)
            nonzeros_true = pdists_true != 0
        else:
            log_truedists = pdists_true
            nonzeros_true = np.ones_like(pdists_true, dtype=bool)

        if dataset.data_type == 'bonsai':
            # avg_rel_diffs.append(np.mean(np.abs((pdists_data - pdists_true) / pdists_true)))
            avg_rel_diffs.append(np.mean((np.abs((np.log(pdists_data) - np.mean(np.log(pdists_data))) - (
                    np.log(pdists_true) - np.mean(np.log(pdists_true)))))))

        if make_fig:
            if not dataset.data_type == 'delta_true':
                if flip_axes:
                    ax.scatter(pdists_data, pdists_true, s=2.5,
                               color=dataset.data_type_color, zorder=0, alpha=.5)
                else:
                    ax.scatter(pdists_true, pdists_data, s=2.5,
                               color=dataset.data_type_color, zorder=0, alpha=.5)

            if ax2 is not None:
                ax2.hist(np.abs((pdists_data - pdists_true) / pdists_true), bins=100, density=True, histtype='step',
                         label=first_title, cumulative=-1)
            if ax3 is not None:
                ax3.hist(np.abs((np.log(pdists_data) - np.mean(np.log(pdists_data))) - (
                        np.log(pdists_true) - np.mean(np.log(pdists_true)))), bins=100, density=True, histtype='step',
                         label=first_title, cumulative=-1)
            # Get correlation between distances:
            if loglog_corr:
                nonzeros = pdists_data != 0
                nonzeros_all = nonzeros * nonzeros_true
                log_truedists_nonzeros = log_truedists[nonzeros_all]
                log_datadists = np.log(pdists_data[nonzeros_all])
            else:
                # nonzeros_all = np.ones_like(log_truedists, dtype=bool)
                log_truedists_nonzeros = log_truedists
                log_datadists = pdists_data

            # TODO: Maybe remove later
            # lt1 = log_truedists_nonzeros > -1e9
            # log_truedists_lt1 = log_truedists_nonzeros[lt1]
            # log_datadists_lt1 = log_datadists[lt1]

            # Clog = np.cov(np.vstack((log_truedists_nonzeros - np.mean(log_truedists_nonzeros), log_datadists - np.mean(log_datadists))))
            Clog = np.cov(np.vstack((log_truedists - np.mean(log_truedists), log_datadists - np.mean(log_datadists))))
            corrlog = Clog[0, 1] / np.sqrt(Clog[0, 0] * Clog[1, 1])
            dataset.PearsonR = corrlog
            eigVals, eigVecs = np.linalg.eig(Clog)
            max_eigval = np.argmax(eigVals)
            slopepca1log = eigVecs[1, max_eigval] / eigVecs[0, max_eigval]
            if flip_axes:
                Clog_flipped = np.cov(np.vstack(
                    (log_datadists - np.mean(log_datadists), log_truedists - np.mean(log_truedists))))
                eigVals_flipped, eigVecs_flipped = np.linalg.eig(Clog_flipped)
                max_eigval_flipped = np.argmax(eigVals_flipped)
                slopepca1log_flipped = eigVecs_flipped[1, max_eigval_flipped] / eigVecs_flipped[0, max_eigval_flipped]
            # regLineXlog = np.linspace(log_truedists_nonzeros.min(), log_truedists_nonzeros.max(), 20)
            regLineXlog = np.linspace(log_truedists.min(), log_truedists.max(), 20)
            # regLineYlog = slopepca1log * (regLineXlog - log_truedists_nonzeros.mean()) + log_datadists.mean()
            regLineYlog = slopepca1log * (regLineXlog - log_truedists.mean()) + log_datadists.mean()

            # linregress_res = linregress(x=log_truedists_nonzeros, y=log_datadists)
            # TODO: Add back later
            # linregress_res_Y_log = regLineXlog * linregress_res.slope + linregress_res.intercept

            # TODO: Remove later
            # from sklearn.linear_model import LinearRegression
            # x = np.array(log_truedists).reshape((-1, 1))
            # y = np.array(log_datadists)
            # model = LinearRegression(fit_intercept=False)
            # model.fit(x, y)
            # r_sq = model.score(x, y)
            # intercept = model.intercept_
            # slope = model.coef_
            # regLineYlog = intercept + slope * regLineXlog

            if loglog_corr:
                y_fit = np.exp(regLineYlog)
                x_fit = np.exp(regLineXlog)
            else:
                y_fit = regLineYlog
                x_fit = regLineXlog
            if not flip_axes:
                ax.plot(x_fit, y_fit, '--', lw=2, c='black', zorder=10)
            else:
                ax.plot(y_fit, x_fit, '--', lw=2, c='black', zorder=10)
            # ax.plot(np.exp(regLineXlog), np.exp(linregress_res_Y_log), '--', lw=2, c='black', zorder=12)
            # ax.text(0.98, 0.1, "Slope: {:.2f},\nR={:.2f}".format(linregress_res.slope, linregress_res.rvalue),
            #         horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
            if corr_measure == 'Rsq':
                corr_label = corrlog ** 2
            elif corr_measure == '-log(1-Rsq)':
                corr_label = -np.log10(1 - corrlog ** 2)
            if not flip_axes:
                ax.text(0.98, 0.01, "Slope: {:.2f},\n{}={:.2f}".format(slopepca1log, corr_measure, corr_label),
                        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                        fontsize=fontsize)
            else:
                ax.text(0.98, 0.01, "Slope: {:.2f},\n{}={:.2f}".format(slopepca1log_flipped, corr_measure, corr_label),
                        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                        fontsize=fontsize)
            # ax.text(0.98, 0.01, "-log(1-R^2)={:.2f}".format(-np.log(1 - corrlog ** 2)),
            #         horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)

            # ax.text("Correlation of %f. Fitted line: y = %f + %f x\" %(corr, regLineY[0], slopepca1))
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if set_lims:
                if loglog_corr:
                    xmin = .9 * np.exp(log_truedists_nonzeros.min())
                    xmax = .9 * np.exp(log_truedists_nonzeros.max())
                    ymin = 1.1 * np.exp(log_datadists.min())
                    ymax = 1.1 * np.exp(log_datadists.max())
                else:
                    xmin = 0
                    xmax = 1.1 * log_truedists_nonzeros.max()
                    ymin = 0
                    ymax = 1.1 * log_datadists.max()
                if not flip_axes:
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                else:
                    ax.set_ylim(xmin, xmax)
                    ax.set_xlim(ymin, ymax)
            if loglog_corr:
                ax.set_xscale('log')
                ax.set_yscale('log')
            # if dataset_counter == 0:
            if (dataset_counter == len(datasets) - 2) and XLABEL:
                ax.set_xlabel('True squared distances', fontsize=fontsize)
            fam = dataset.data_type.split('_')[0]
            if YLABEL:
                if fam != 'bonsai':
                    ax.set_ylabel('{}: Inferred \nsquared distances'.format(fam), fontsize=fontsize)
                else:
                    ax.set_ylabel('Bonsai: Summed \nbranch lengths', fontsize=fontsize)
            if first_title is not None:
                if ind_dataset == 0:
                    ax.set_title(first_title)
            else:
                ax.set_title(dataset.data_id)
    if return_Rvals:
        return avg_rel_diffs, R_vals
    else:
        return avg_rel_diffs


def compare_pdists_to_truth_per_cell(datasets, make_fig=True, axs=None, title='', YLABEL=True, XLABEL=True,
                                     first_title=None, set_lims=True, corr_measure='Rsq', return_Rvals=False,
                                     flip_axes=False, density=False, bins=20, share_y=False, loglog_corr=False):
    if make_fig and (axs is None):
        n_datasets = len(datasets) - 1  # Minus 1 to subtract for true dataset
        n_rows = int(np.ceil(np.sqrt(n_datasets)))
        n_cols = int(np.ceil(n_datasets / n_rows))
        fig, axs = plt.subplots(n_rows=n_rows, n_cols=n_cols, figsize=(8, 6))

    true_dataset_ind = [ind for ind, dataset in enumerate(datasets) if dataset.data_type == 'delta_true']
    if len(true_dataset_ind) != 1:
        print("Exactly one of the provided dataset should contain the true nearest neighbours")
        exit()
    else:
        true_dataset_ind = true_dataset_ind[0]
        true_dataset = datasets[true_dataset_ind]
        pdists_true = true_dataset.distances

    # n_cells = nn_mat_true.shape[0] + 1
    dataset_counter = -1
    avg_rel_diffs = [None]
    R_vals = {}
    for ind_dataset, dataset in enumerate(datasets):
        if ind_dataset == true_dataset_ind:
            continue
        dataset_counter += 1
        try:
            ax = axs.flatten()[dataset_counter]
            fontsize = 'x-small' if (len(axs.flatten()) > 4) else 'large'
        except AttributeError:
            ax = axs
            fontsize = 'x-small'
        fontsize = 'large'
        print("Comparing pairwise distances of dataset: '" + dataset.data_type + "'.")
        pdists_data = squareform(dataset.distances)
        true_dists = squareform(pdists_true)
        # log_truedists = squareform(np.log(pdists_true))

        # Remove the diagonal element, which is uninformative for distances
        n_cells = true_dists.shape[0]
        mask = ~np.eye(n_cells, dtype=bool)
        true_dists = true_dists[mask].reshape(n_cells, n_cells-1)
        pdists_data = pdists_data[mask].reshape(n_cells, n_cells-1)

        # if loglog_corr:
        #     log_truedists = np.log(true_dists)
        # else:
        #     log_truedists = true_dists

        recalc_pearsonRs = False
        if dataset.pearsonRs is None:
            dataset.pearsonRs = np.zeros(n_cells)
            dataset.avg_rel_errors = np.zeros(n_cells)
            recalc_pearsonRs = True
        if make_fig:
            if recalc_pearsonRs:
                # Get correlation between distances:
                for cell_ind in range(n_cells):
                    # THIS_FIG = (cell_ind % int(np.ceil(n_cells / 10)) == 0)
                    THIS_FIG = False
                    pdists_data_cell = pdists_data[cell_ind, :]
                    # log_truedists_cell = log_truedists[cell_ind, :]
                    # pdists_true_cell = squareform(pdists_true)[cell_ind, :]
                    pdists_true_cell = true_dists[cell_ind, :]

                    if THIS_FIG:
                        fig_cell, ax_cell = plt.subplots()
                        if not dataset.data_type == 'delta_true':
                            if flip_axes:
                                ax_cell.scatter(pdists_data_cell, pdists_true[cell_ind, :], s=2.5,
                                                color=dataset.data_type_color, zorder=0, alpha=.5)
                            else:
                                ax_cell.scatter(pdists_true_cell, pdists_data_cell, s=2.5,
                                                color=dataset.data_type_color, zorder=0, alpha=.5)

                    nonzeros_true_cell = pdists_true_cell != 0
                    nonzeros_cell = pdists_data_cell != 0
                    nonzeros_all = nonzeros_cell * nonzeros_true_cell
                    # log_truedists_nonzeros_cell = log_truedists_cell[nonzeros_all]
                    if loglog_corr:
                        truedists_transformed_cell = np.log(pdists_true_cell[nonzeros_all])
                        datadists_transformed_cell = np.log(pdists_data_cell[nonzeros_all])
                    else:
                        truedists_transformed_cell = pdists_true_cell
                        datadists_transformed_cell = pdists_data_cell

                    # # TODO: Maybe remove later
                    # lt1 = log_truedists_nonzeros > -1e9
                    # log_truedists_lt1 = log_truedists_nonzeros[lt1]
                    # log_datadists_lt1 = log_datadists[lt1]

                    # Clog = np.cov(np.vstack((log_truedists_nonzeros - np.mean(log_truedists_nonzeros), log_datadists - np.mean(log_datadists))))
                    Clog = np.cov(np.vstack(
                        (truedists_transformed_cell - np.mean(truedists_transformed_cell),
                         datadists_transformed_cell - np.mean(datadists_transformed_cell))))
                    pearsonR = Clog[0, 1] / np.sqrt(Clog[0, 0] * Clog[1, 1])
                    dataset.pearsonRs[cell_ind] = pearsonR
                    dataset.avg_rel_errors[cell_ind] = np.mean(np.abs(
                        (pdists_true_cell[nonzeros_all] - pdists_data_cell[nonzeros_all]) / pdists_true_cell[nonzeros_all]))
                    # dataset.PearsonR = corrlog
                    if THIS_FIG:
                        eigVals, eigVecs = np.linalg.eig(Clog)
                        max_eigval = np.argmax(eigVals)
                        slopepca1log = eigVecs[1, max_eigval] / eigVecs[0, max_eigval]
                        if flip_axes:
                            Clog_flipped = np.cov(np.vstack(
                                (datadists_transformed_cell - np.mean(datadists_transformed_cell),
                                 truedists_transformed_cell - np.mean(truedists_transformed_cell))))
                            eigVals_flipped, eigVecs_flipped = np.linalg.eig(Clog_flipped)
                            max_eigval_flipped = np.argmax(eigVals_flipped)
                            slopepca1log_flipped = eigVecs_flipped[1, max_eigval_flipped] / eigVecs_flipped[
                                0, max_eigval_flipped]
                        # regLineXlog = np.linspace(log_truedists_nonzeros.min(), log_truedists_nonzeros.max(), 20)
                        regLineXlog = np.linspace(truedists_transformed_cell.min(), truedists_transformed_cell.max(), 20)
                        # regLineYlog = slopepca1log * (regLineXlog - log_truedists_nonzeros.mean()) + log_datadists.mean()
                        regLineYlog = slopepca1log * (
                                regLineXlog - truedists_transformed_cell.mean()) + datadists_transformed_cell.mean()

                    # linregress_res = linregress(x=log_truedists_nonzeros, y=log_datadists)
                    # TODO: Add back later
                    # linregress_res_Y_log = regLineXlog * linregress_res.slope + linregress_res.intercept

                    # TODO: Remove later
                    # from sklearn.linear_model import LinearRegression
                    # x = np.array(log_truedists).reshape((-1, 1))
                    # y = np.array(log_datadists)
                    # model = LinearRegression(fit_intercept=False)
                    # model.fit(x, y)
                    # r_sq = model.score(x, y)
                    # intercept = model.intercept_
                    # slope = model.coef_
                    # regLineYlog = intercept + slope * regLineXlog
                    if THIS_FIG:
                        if not flip_axes:
                            if loglog_corr:
                                ax_cell.plot(np.exp(regLineXlog), np.exp(regLineYlog), '--', lw=2, c='black', zorder=10)
                            else:
                                ax_cell.plot(regLineXlog, regLineYlog, '--', lw=2, c='black', zorder=10)
                        else:
                            if loglog_corr:
                                ax_cell.plot(np.exp(regLineYlog), np.exp(regLineXlog), '--', lw=2, c='black', zorder=10)
                            else:
                                ax_cell.plot(regLineYlog, regLineXlog, '--', lw=2, c='black', zorder=10)
                        # ax.plot(np.exp(regLineXlog), np.exp(linregress_res_Y_log), '--', lw=2, c='black', zorder=12)
                        # ax.text(0.98, 0.1, "Slope: {:.2f},\nR={:.2f}".format(linregress_res.slope, linregress_res.rvalue),
                        #         horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
                        if corr_measure == 'Rsq':
                            corr_label = pearsonR ** 2
                        elif corr_measure == '-log(1-Rsq)':
                            corr_label = -np.log10(1 - pearsonR ** 2)
                        if not flip_axes:
                            ax_cell.text(0.98, 0.01,
                                         "Slope: {:.2f},\n{}={:.2f}".format(slopepca1log, corr_measure, corr_label),
                                         horizontalalignment='right', verticalalignment='bottom',
                                         transform=ax_cell.transAxes,
                                         fontsize=fontsize)
                        else:
                            ax_cell.text(0.98, 0.01,
                                         "Slope: {:.2f},\n{}={:.2f}".format(slopepca1log_flipped, corr_measure, corr_label),
                                         horizontalalignment='right', verticalalignment='bottom',
                                         transform=ax_cell.transAxes,
                                         fontsize=fontsize)
                    # ax.text(0.98, 0.01, "-log(1-R^2)={:.2f}".format(-np.log(1 - corrlog ** 2)),
                    #         horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)

                    # ax.text("Correlation of %f. Fitted line: y = %f + %f x\" %(corr, regLineY[0], slopepca1))
                    # box = ax.get_position()
                    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    if set_lims and THIS_FIG:
                        if not flip_axes:
                            ax_cell.set_xlim(np.exp(truedists_transformed_cell.min()),
                                             np.exp(truedists_transformed_cell.max()))
                            # ax.set_ylim(min(np.exp(log_truedists_nonzeros.min()), np.exp(regLineYlog.min())), max(np.exp(log_truedists_nonzeros.max()), np.exp(regLineYlog.max())))
                            ax_cell.set_ylim(np.exp(truedists_transformed_cell.min()),
                                             np.exp(truedists_transformed_cell.max()))
                        else:
                            ax_cell.set_ylim(np.exp(truedists_transformed_cell.min()),
                                             np.exp(truedists_transformed_cell.max()))
                            # ax.set_ylim(min(np.exp(log_truedists_nonzeros.min()), np.exp(regLineYlog.min())), max(np.exp(log_truedists_nonzeros.max()), np.exp(regLineYlog.max())))
                            ax_cell.set_xlim(np.exp(truedists_transformed_cell.min()),
                                             np.exp(truedists_transformed_cell.max()))
                    if THIS_FIG and loglog_corr:
                        ax_cell.set_xscale('log')
                        ax_cell.set_yscale('log')

            ax.hist(dataset.pearsonRs, bins=bins, color=dataset.data_type_color, range=(-0.1, 1), density=density)
            # if dataset_counter == 0:
            if (dataset_counter == len(datasets) - 2) and XLABEL:
                ax.set_xlabel('R-squared values between\ninferred and true distances per cell', fontsize=fontsize)
            fam = dataset.data_type.split('_')[0]
            if YLABEL:
                ax.set_ylabel("Numbers of cells")
                # if fam != 'bonsai':
                #     ax.set_ylabel('{}: Inferred \nsquared distances'.format(fam), fontsize=fontsize)
                # else:
                #     ax.set_ylabel('Bonsai: Summed \nbranch lengths', fontsize=fontsize)
            if first_title is not None:
                if ind_dataset == 0:
                    ax.set_title(first_title)
            else:
                ax.set_title(dataset.data_id)
    if return_Rvals:
        return avg_rel_diffs, R_vals
    else:
        return avg_rel_diffs


def compare_pdists_to_truth_per_cell_adjusted(datasets, make_fig=True, axs=None, title='', YLABEL=True, XLABEL=True,
                                              first_title=None, set_lims=True, corr_measure='Rsq', return_Rvals=False,
                                              flip_axes=False, density=False, bins=20, share_y=False, loglog_corr=False,
                                              histtype='bar'):
    if make_fig and (axs is None):
        n_datasets = len(datasets) - 1  # Minus 1 to subtract for true dataset
        n_rows = int(np.ceil(np.sqrt(n_datasets)))
        n_cols = int(np.ceil(n_datasets / n_rows))
        fig, axs = plt.subplots(n_rows=n_rows, n_cols=n_cols, figsize=(8, 6))

    true_dataset_ind = [ind for ind, dataset in enumerate(datasets) if dataset.data_type == 'delta_true']
    if len(true_dataset_ind) != 1:
        print("Exactly one of the provided dataset should contain the true nearest neighbours")
        exit()
    else:
        true_dataset_ind = true_dataset_ind[0]
        true_dataset = datasets[true_dataset_ind]
        pdists_true = true_dataset.distances

    dataset_counter = -1
    avg_rel_diffs = [None]
    R_vals = {}
    for ind_dataset, dataset in enumerate(datasets):
        if ind_dataset == true_dataset_ind:
            continue
        dataset_counter += 1
        try:
            ax = axs.flatten()[dataset_counter]
            fontsize = 'x-small' if (len(axs.flatten()) > 4) else 'large'
        except AttributeError:
            ax = axs
            fontsize = 'x-small'
        fontsize = 'large'
        print("Comparing pairwise distances of dataset: '" + dataset.data_type + "'.")
        pdists_data = squareform(dataset.distances)
        if loglog_corr:
            log_truedists = squareform(np.log(pdists_true))
        else:
            log_truedists = squareform(pdists_true)
        n_cells = log_truedists.shape[0]
        dataset.pearsonRs = np.zeros(n_cells)
        dataset.avg_rel_errors = np.zeros(n_cells)
        if make_fig:
            # Get correlation between distances:
            for cell_ind in range(n_cells):
                # THIS_FIG = (cell_ind % int(np.ceil(n_cells / 10)) == 0)
                # THIS_FIG = (cell_ind in [33, 9, 27]) and (int(first_title.split('-')[0]) in [1,20])
                THIS_FIG = False
                pdists_data_cell = pdists_data[cell_ind, :]
                log_truedists_cell = log_truedists[cell_ind, :]
                pdists_true_cell = squareform(pdists_true)[cell_ind, :]

                if THIS_FIG:
                    fig_cell, ax_cell = plt.subplots()
                    if not dataset.data_type == 'delta_true':
                        if flip_axes:
                            ax_cell.scatter(pdists_data_cell, pdists_true_cell, s=2.5,
                                            color=dataset.data_type_color, zorder=0, alpha=.5)
                            ax_cell.set_ylabel('True distances')
                            ax_cell.set_xlabel('Inferred distances')
                        else:
                            ax_cell.scatter(pdists_true_cell, pdists_data_cell, s=2.5,
                                            color=dataset.data_type_color, zorder=0, alpha=.5)
                            ax_cell.set_xlabel('True distances')
                            ax_cell.set_ylabel('Inferred distances')

                nonzeros_true_cell = pdists_true_cell != 0
                nonzeros_cell = pdists_data_cell != 0
                nonzeros_all = nonzeros_cell * nonzeros_true_cell
                log_truedists_nonzeros_cell = log_truedists_cell[nonzeros_all]
                if loglog_corr:
                    log_datadists_cell = np.log(pdists_data_cell[nonzeros_all])
                else:
                    log_datadists_cell = pdists_data_cell[nonzeros_all]

                # # TODO: Maybe remove later
                # lt1 = log_truedists_nonzeros > -1e9
                # log_truedists_lt1 = log_truedists_nonzeros[lt1]
                # log_datadists_lt1 = log_datadists[lt1]

                # Clog = np.cov(np.vstack((log_truedists_nonzeros - np.mean(log_truedists_nonzeros), log_datadists - np.mean(log_datadists))))
                Clog = np.cov(np.vstack(
                    (log_truedists_nonzeros_cell - np.mean(log_truedists_nonzeros_cell),
                     log_datadists_cell - np.mean(log_datadists_cell))))
                pearsonR = Clog[0, 1] / np.sqrt(Clog[0, 0] * Clog[1, 1])
                dataset.pearsonRs[cell_ind] = pearsonR
                dataset.avg_rel_errors[cell_ind] = np.mean(np.abs(
                    (pdists_true_cell[nonzeros_all] - pdists_data_cell[nonzeros_all]) / pdists_true_cell[nonzeros_all]))
                if THIS_FIG:
                    eigVals, eigVecs = np.linalg.eig(Clog)
                    max_eigval = np.argmax(eigVals)
                    slopepca1log = eigVecs[1, max_eigval] / eigVecs[0, max_eigval]
                    if flip_axes:
                        Clog_flipped = np.cov(np.vstack(
                            (log_datadists_cell - np.mean(log_datadists_cell),
                             log_truedists_nonzeros_cell - np.mean(log_truedists_nonzeros_cell))))
                        eigVals_flipped, eigVecs_flipped = np.linalg.eig(Clog_flipped)
                        max_eigval_flipped = np.argmax(eigVals_flipped)
                        slopepca1log_flipped = eigVecs_flipped[1, max_eigval_flipped] / eigVecs_flipped[
                            0, max_eigval_flipped]
                    # regLineXlog = np.linspace(log_truedists_nonzeros.min(), log_truedists_nonzeros.max(), 20)
                    regLineXlog = np.linspace(log_truedists_nonzeros_cell.min(), log_truedists_nonzeros_cell.max(), 20)
                    # regLineYlog = slopepca1log * (regLineXlog - log_truedists_nonzeros.mean()) + log_datadists.mean()
                    regLineYlog = slopepca1log * (
                            regLineXlog - log_truedists_nonzeros_cell.mean()) + log_datadists_cell.mean()

                # linregress_res = linregress(x=log_truedists_nonzeros, y=log_datadists)
                # TODO: Add back later
                # linregress_res_Y_log = regLineXlog * linregress_res.slope + linregress_res.intercept

                # TODO: Remove later
                # from sklearn.linear_model import LinearRegression
                # x = np.array(log_truedists).reshape((-1, 1))
                # y = np.array(log_datadists)
                # model = LinearRegression(fit_intercept=False)
                # model.fit(x, y)
                # r_sq = model.score(x, y)
                # intercept = model.intercept_
                # slope = model.coef_
                # regLineYlog = intercept + slope * regLineXlog
                if THIS_FIG:
                    if not flip_axes:
                        if loglog_corr:
                            ax_cell.plot(np.exp(regLineXlog), np.exp(regLineYlog), '--', lw=2, c='black', zorder=10)
                        else:
                            ax_cell.plot(regLineXlog, regLineYlog, '--', lw=2, c='black', zorder=10)
                    else:
                        if loglog_corr:
                            ax_cell.plot(np.exp(regLineYlog), np.exp(regLineXlog), '--', lw=2, c='black', zorder=10)
                        else:
                            ax_cell.plot(regLineYlog, regLineXlog, '--', lw=2, c='black', zorder=10)
                # ax.plot(np.exp(regLineXlog), np.exp(linregress_res_Y_log), '--', lw=2, c='black', zorder=12)
                # ax.text(0.98, 0.1, "Slope: {:.2f},\nR={:.2f}".format(linregress_res.slope, linregress_res.rvalue),
                #         horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
                if corr_measure == 'Rsq':
                    corr_label = pearsonR ** 2
                elif corr_measure == '-log(1-Rsq)':
                    corr_label = -np.log10(1 - pearsonR ** 2)
                if THIS_FIG:
                    if not flip_axes:
                        ax_cell.text(0.98, 0.01,
                                     "Slope: {:.2f},\n{}={:.2f}".format(slopepca1log, corr_measure, corr_label),
                                     horizontalalignment='right', verticalalignment='bottom',
                                     transform=ax_cell.transAxes,
                                     fontsize=fontsize)
                    else:
                        ax_cell.text(0.98, 0.01,
                                     "Slope: {:.2f},\n{}={:.2f}".format(slopepca1log_flipped, corr_measure, corr_label),
                                     horizontalalignment='right', verticalalignment='bottom',
                                     transform=ax_cell.transAxes,
                                     fontsize=fontsize)
                # ax.text(0.98, 0.01, "-log(1-R^2)={:.2f}".format(-np.log(1 - corrlog ** 2)),
                #         horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)

                # ax.text("Correlation of %f. Fitted line: y = %f + %f x\" %(corr, regLineY[0], slopepca1))
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                if set_lims and THIS_FIG:
                    if not flip_axes:
                        ax_cell.set_xlim(np.exp(log_truedists_nonzeros_cell.min()),
                                         np.exp(log_truedists_nonzeros_cell.max()))
                        # ax.set_ylim(min(np.exp(log_truedists_nonzeros.min()), np.exp(regLineYlog.min())), max(np.exp(log_truedists_nonzeros.max()), np.exp(regLineYlog.max())))
                        ax_cell.set_ylim(np.exp(log_truedists_nonzeros_cell.min()),
                                         np.exp(log_truedists_nonzeros_cell.max()))
                    else:
                        ax_cell.set_ylim(np.exp(log_truedists_nonzeros_cell.min()),
                                         np.exp(log_truedists_nonzeros_cell.max()))
                        # ax.set_ylim(min(np.exp(log_truedists_nonzeros.min()), np.exp(regLineYlog.min())), max(np.exp(log_truedists_nonzeros.max()), np.exp(regLineYlog.max())))
                        ax_cell.set_xlim(np.exp(log_truedists_nonzeros_cell.min()),
                                         np.exp(log_truedists_nonzeros_cell.max()))
                if THIS_FIG:
                    if loglog_corr:
                        ax_cell.set_xscale('log')
                        ax_cell.set_yscale('log')
                    ax_cell.set_title("Cell: {} in {}".format(cell_ind, first_title))

            ax.hist(dataset.pearsonRs, bins=bins, color=dataset.data_type_color, range=(-0.1, 1), density=density,
                    histtype=histtype, label=dataset.data_id)
            # if dataset_counter == 0:
            if (dataset_counter == len(datasets) - 2) and XLABEL:
                # ax.set_xlabel('R-squared values between\ninferred and true distances per cell', fontsize=fontsize)
                ax.set_xlabel('Pearson R-values between\ninferred and true distances \n of each cell to all others',
                              fontsize=fontsize)
            fam = dataset.data_type.split('_')[0]
            if YLABEL:
                ax.set_ylabel("Numbers of cells")
                # if fam != 'bonsai':
                #     ax.set_ylabel('{}: Inferred \nsquared distances'.format(fam), fontsize=fontsize)
                # else:
                #     ax.set_ylabel('Bonsai: Summed \nbranch lengths', fontsize=fontsize)
            if first_title is not None:
                if ind_dataset == 0:
                    ax.set_title(first_title)
            else:
                ax.set_title(dataset.data_id)
    if return_Rvals:
        return avg_rel_diffs, R_vals
    else:
        return avg_rel_diffs


def pdist_custom(coords):
    """

    :param coords: 2d-array with as rows different observations (cells), and columns different features (genes)
    :return: pdist-vector (in compact scipy-format)
    """
    n_cell = coords.shape[0]
    total_dists = int(n_cell * (n_cell - 1) / 2)
    pdists = np.zeros(total_dists)
    counter = 0
    for ind in range(n_cell - 1):
        if ind % 1000 == 0:
            print("Calculated distances from cell %d, %.2f percent." % (ind, (counter / total_dists)))
        cell_coords = coords[ind, :]
        larger_inds_left = n_cell - ind - 1
        pdists[counter: counter + larger_inds_left] = np.sum((coords[ind + 1:, :] - cell_coords) ** 2, axis=1)
        counter += larger_inds_left
    return pdists
