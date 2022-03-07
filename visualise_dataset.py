from matplotlib import pyplot as plt
import numpy as np
from utils import get_split_graphset, get_split_dataset


if __name__ == '__main__':
    train_graphs, train_features, train_y, val_graphs, val_features, val_y, test_graphs, test_features, test_y = \
        get_split_graphset('dataset/hela_100', 64, 1024, ['1'])

    fig, ax = plt.subplots()
    ax.matshow(train_graphs[0], cmap="YlOrBr", interpolation='none')
    plt.sca(ax)
    plt.axis('off')
    plt.savefig('tmp/hela_graph.pdf')

    fig, ax = plt.subplots()
    ax.matshow(train_y[0], interpolation='none')
    plt.sca(ax)
    plt.axis('off')
    plt.savefig('tmp/hela_y.pdf')



    train_graphs, train_features, train_y, val_graphs, val_features, val_y, test_graphs, test_features, test_y = \
        get_split_graphset('dataset/hela_100_pl', 64, 1024, ['1'])

    fig, ax = plt.subplots()
    ax.matshow(train_graphs[0], cmap="YlOrBr", interpolation='none')
    plt.sca(ax)
    plt.axis('off')
    plt.savefig('tmp/hela_pl_graph.pdf')

    fig, ax = plt.subplots()
    ax.matshow(train_y[0], interpolation='none')
    plt.sca(ax)
    plt.axis('off')
    plt.savefig('tmp/hela_pl_y.pdf')
