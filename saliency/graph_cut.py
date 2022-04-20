import maxflow
import numpy as np
import math
import cv2
from sklearn.mixture import GaussianMixture


def get_mask(saliency_map, img, background_quantile, foreground_quantile):
    SIGMA = 30

    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes(saliency_map.shape)

    # get GMM models
    flatten_channels = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    assert(flatten_channels.shape == (299*299, 3))

    gmm = GaussianMixture(n_components=2, random_state=0).fit(flatten_channels)

    # get the quantile thresholds
    background_threshold = np.quantile(saliency_map, background_quantile)
    foreground_threshold = np.quantile(saliency_map, foreground_quantile)

    K = -float("inf")
    # Add edges.
    max_row, max_col = saliency_map.shape
    for row in range(max_row):
        for col in range(max_col):
            # n-links
            # pixel below
            if (row + 1 < max_row):
                weight = get_weights(
                    img[row, col, :], img[row + 1, col, :], SIGMA)
                g.add_edge(nodeids[row, col],
                           nodeids[row + 1, col], weight, weight)
                K = max(K, weight)
            # pixel to the right
            if (col + 1 < max_col):
                weight = get_weights(
                    img[row, col, :], img[row, col + 1, :], SIGMA)
                g.add_edge(nodeids[row, col],
                           nodeids[row, col + 1], weight, weight)
                K = max(K, weight)

            # t-links
            saliency_pixel = saliency_map[row, col]
            pixel = img[row, col, :]

            probs = gmm.predict_proba([pixel])[0]
            max_prob = max(probs)
            # g.add_tedge(nodeids[row, col], probs[0]*K, probs[1]*K)

            if (saliency_pixel >= foreground_threshold):
                g.add_tedge(nodeids[row, col], K, 0)
            elif (saliency_pixel <= background_threshold):
                g.add_tedge(nodeids[row, col], 0, K)
            # else:
            #     if (probs[0] == 0):
            #         probs[0] = 0.00000001
            #     if (probs[1] == 0):
            #         probs[1] = 0.00000001
            #     g.add_tedge(nodeids[row, col], -
            #                 np.log(probs[0]), -np.log(probs[1]))

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    # sgm.shape == nodeids.shape
    sgm = g.get_grid_segments(nodeids)
    # The labels should be 1 where sgm is False and 0 otherwise.
    ret = np.int_(np.logical_not(sgm))

    return np.expand_dims(ret, 2).repeat(3, axis=2)


def get_cut_img(img, mask):
    # mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    cut_img = mask * img

    return cut_img


def graph_cut(imgs, saliency_map, background_quantile, foreground_quantile):
    cut_imgs = []
    masks = []

    for idx in range(0, len(imgs)):
        img = imgs[idx]
        sm = saliency_map[idx]
        mask = get_mask(sm, img, background_quantile, foreground_quantile)
        cut_img = get_cut_img(img, mask)

        masks.append(mask*255)
        cut_imgs.append(cut_img)

    return np.array(cut_imgs), np.array(masks)


def get_weights(ip, iq, SIGMA):
    w = 100 * math.exp(-np.linalg.norm(ip - iq) / (2 * pow(SIGMA, 2)))
    return w
