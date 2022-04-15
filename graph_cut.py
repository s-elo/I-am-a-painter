import maxflow
import numpy as np
import math


def get_mask(image, SIGMA):
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes(image.shape)

    K = -float("inf")
    # Add non-terminal edges.
    max_row, max_col = image.shape
    for row in range(max_row):
        for col in range(max_col):
            # pixel below
            if (row + 1 < max_row):
                weight = get_weights(
                    image[row, col], image[row + 1, col], SIGMA)
                g.add_edge(nodeids[row, col],
                           nodeids[row + 1, col], weight, weight)
                K = max(K, weight)
            # pixel to the right
            if (col + 1 < max_col):
                weight = get_weights(
                    image[row, col], image[row, col + 1], SIGMA)
                g.add_edge(nodeids[row, col],
                           nodeids[row, col + 1], weight, weight)
                K = max(K, weight)

    # Add the terminal edges.
    # g.add_grid_tedges(nodeids, 255-sm, sm)
    for row in range(max_row):
        for col in range(max_col):
            pixel_val = image[row, col]

            if (pixel_val > 5):
                g.add_tedge(nodeids[row, col], K, 0)
            else:
                g.add_tedge(nodeids[row, col], 0, K)

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    # sgm.shape == nodeids.shape
    sgm = g.get_grid_segments(nodeids)
    # The labels should be 1 where sgm is False and 0 otherwise.
    ret = np.int_(np.logical_not(sgm))

    return ret


def get_cut_img(img, mask):
    mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    cut_img = mask * img

    return cut_img


def get_weights(ip, iq, SIGMA):
    w = 100 * math.exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return w
