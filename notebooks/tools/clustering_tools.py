import time
import numpy as np
import scipy
import sklearn
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs


def plot_raw_data(X, labels):
    plt.scatter(X[:, 0], X[:, 1])
    for i, label in enumerate(labels):
        plt.text(X[i, 0] - 0.1, X[i, 1] + 0.1, label, fontsize=10, ha='right', va='bottom')

    plt.show()


def plot_clusters(
    X,
    cluster_labels,
    labels=None,
    special_indices=None,
    cluster_centers=None,
    text_shift=0.1,
    fontsize=10,
    s=None,
    new_fig=True,
    figsize=(6.4, 4.8),
    title=None,
    xlabel=None,
    ylabel=None,
    show_legend=True,
    show_plot=True,
):
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10')
    big_s = s * 3 if s is not None else None

    if new_fig:
        plt.figure(figsize=figsize)

    for i, cluster in enumerate(unique_clusters):
        cluster_points = X[cluster_labels == cluster]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            label=f'Cluster {cluster}', color=colors(i), alpha=0.6, s=s
        )

    if cluster_centers is not None:
        plt.scatter(
            cluster_centers[:, 0], cluster_centers[:, 1],
            c=[colors(i) for i in unique_clusters], marker='D', s=big_s, label='Centroids'
        )

    if labels is not None and special_indices is None:
        for i, label in enumerate(labels):
            plt.text(
                X[i, 0] - text_shift, X[i, 1] + text_shift, label,
                fontsize=fontsize, ha='right', va='bottom'
            )

    if special_indices is not None:
        plt.scatter(
            X[special_indices, 0], X[special_indices, 1],
            c='orange', marker='s', s=big_s, label='Special'
        )
        for i in special_indices:
            plt.text(
                X[i, 0] - text_shift, X[i, 1] + text_shift, labels[i],
                fontsize=fontsize, ha='right', va='bottom'
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_legend:
        plt.legend()

    if show_plot:
        plt.show()


def plot_dendrogram(children, distances, labels, title, figsize=(6.4, 4.8)):
    linkage_matrix = np.column_stack([children, distances, np.zeros(len(children))]).astype(float)

    plt.figure(figsize=figsize)
    scipy.cluster.hierarchy.dendrogram(linkage_matrix, labels=labels, count_sort='descending')
    plt.title(title)
    plt.ylabel("Distance")
    plt.show()


def generate_toy_datasets(n_samples, random_state_1=30, random_state_2=170):
    circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state_1)[0]
    moons = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state_1)[0]
    blobs = make_blobs(n_samples=n_samples, centers=2, random_state=random_state_1)[0]
    rng = np.random.RandomState(random_state_1)
    no_structure = rng.rand(n_samples, 2)

    X = make_blobs(n_samples=n_samples, centers=2, random_state=random_state_2)[0]
    transformation = np.array([[0.4, -0.4], [-0.4, 0.8]])
    aniso = np.dot(X, transformation)

    varied = make_blobs(
        n_samples=n_samples, centers=2, cluster_std=[1.0, 0.5], random_state=random_state_2
    )[0]

    datasets = [
        ["noisy_circles", circles],
        ["noisy_moons", moons],
        ["varied", varied],
        ["aniso", aniso],
        ["blobs", blobs],
        ["no_structure", no_structure],
    ]
    scaler = sklearn.preprocessing.StandardScaler()
    datasets = [(s, scaler.fit_transform(X)) for s, X in datasets]

    return datasets


def plot_toy_datasets(datasets, clusterers, figsize, axis_lim=2.5):
    fig, axes = plt.subplots(len(datasets), len(clusterers), figsize=figsize)
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.98, wspace=0.01, hspace=0.01
    )

    for row, (dataset_str, X) in enumerate(datasets):
        for col, (clusterer_str, clusterer) in enumerate(clusterers):
            ax = axes[row, col]

            start_time = time.perf_counter()
            clusterer.fit(X)
            fit_time = time.perf_counter() - start_time
            y_pred = clusterer.labels_

            plt.sca(ax)
            plot_clusters(X, y_pred, s=10, show_legend=False, new_fig=False, show_plot=False)

            if row == 0:
                ax.set_title(clusterer_str)
            plt.xlim(-axis_lim, axis_lim)
            plt.ylim(-axis_lim, axis_lim)
            plt.xticks(())
            plt.yticks(())
            ax.text(
                0.99, 0.01, ("%.2fs" % fit_time).lstrip("0"),
                transform=ax.transAxes, size=10, horizontalalignment="right",
            )

    plt.show()
