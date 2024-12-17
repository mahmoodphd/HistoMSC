
import os
import sys
import morse_smale 

import openslide
from tqdm.autonotebook import tqdm
from math import ceil
import matplotlib.pyplot as plt

import matplotlib.cm as cm

import geojson
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.geometry import Polygon


import numpy as np
from skimage import measure
from scipy import ndimage
from skimage.transform import resize

import fastai 
from fastai.vision import *
from fastai.torch_core import *
import pretrainedmodels
from fastai.callbacks import *
from efficientnet_pytorch import EfficientNet
from fastai.vision.models import *

from glob import glob
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from palettable.colorbrewer.qualitative import Set1_3
from palettable.colorbrewer.qualitative import Set1_5
from palettable.colorbrewer.qualitative import Set2_5
import pickle
import scipy.stats as st

from KDEpy import FFTKDE

from scipy import special
import json
import shutil
import cv2
import gzip
import argparse
import time
import pathlib
import random
import torch
from scipy import spatial
import warnings 
warnings.filterwarnings("ignore")

# Helper functions from original code
def compute_histogram(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def id_center(name):
    name = name.split('.')[-2]
    name = name.split('/')[-1]
    ixy = name.split('_')
    return int(ixy[0]), int(ixy[1]), int(ixy[2]), int(ixy[3]), int(ixy[4])

def colorRGB(c):
    red = int(255 * c[0])
    green = int(255 * c[1])
    blue = int(255 * c[2])
    alpha = 255
    RGBint = (alpha << 24) + (red << 16) + (green << 8) + blue
    return RGBint - 2**32

def squeezenet_inference(modeldir, imgdir):
    files = [[image.split('/')[-1], id_center(image)[0], id_center(image)[1], id_center(image)[2], id_center(image)[3], id_center(image)[4]] 
               for image in glob(str(imgdir) + '/*.jpg', recursive=True)]
    test = pd.DataFrame(files, columns=['name', 'id', 'x', 'y', 'w', 'h'])
    
    full_path = Path(modeldir).absolute()
    tester = load_learner(full_path,
                          test=ImageList.from_df(path=imgdir, df=test)
                         ).to_fp16()
    p = tester.get_preds(ds_type=DatasetType.Test)[0]
    prob = special.softmax(np.array(p).reshape(-1, len(tester.data.classes)), axis=1)
    l = np.argmax(prob, axis=1)
    i = [int(v) for v in test['id']]
    x = [int(v) for v in test['x']]
    y = [int(v) for v in test['y']]
    w = [int(v) for v in test['w']]
    h = [int(v) for v in test['h']]
    return [i, x, y, w, h, l, prob]

def read_csv_inference(csv_file):
    df = pd.read_csv(csv_file)
    i = df["i"].tolist()
    x = df["x"].tolist()
    y = df["y"].tolist()
    w = df["w"].tolist()
    h = df["h"].tolist()
    l = df["l"].tolist()
    prob = np.array([df["p0"].tolist(), df["p1"].tolist(), df["p2"].tolist(), df["p3"].tolist(), df["p4"].tolist()]).T.tolist()
    return [i, x, y, w, h, l, prob]

def write_csv_inference(i, x, y, w, h, l, prob, csv_file):
    prob = np.array(prob)
    data = {"i": i, "x": x, "y": y, "w": w, "h": h, "l": l, 
            "p0": prob[:, 0], "p1": prob[:, 1], "p2": prob[:, 2], "p3": prob[:, 3], "p4": prob[:, 4]} 
    df = pd.DataFrame(data)
    df.to_csv(csv_file)
    return

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1    
            else:
                m2 = x
    return m2 if count >= 2 else None    

def _extract_partition_contours(p, apron=10, norm=True):
    labels = set(p.flatten())
    contours = []
    for l in labels:
        contours.append(_extract_contour(l, p, apron, norm))
    return contours

def _extract_contour(l, p, apron=10, norm=True):
    mask = np.zeros(shape=[p.shape[0], p.shape[1]])  
    delta = apron // 2 
    mask[p == l] = 255.0
    mask_apron = np.zeros(shape=[p.shape[0] + apron, p.shape[1] + apron])
    if delta > 0:
        mask_apron[delta:-delta, delta:-delta] = mask
    else:
        mask_apron = mask
    cc = measure.find_contours(mask_apron, 254.0)
    scalex = scaley = 1
    if norm:
        scalex = 1.0 / float(p.shape[0])
        scaley = 1.0 / float(p.shape[1])
    for pp in cc[0]:
        pp[0] = float(pp[0] - delta) * scalex
        pp[1] = float(pp[1] - delta) * scaley
    return cc[0] 

def _compute_center(p, l):
    mask = np.zeros(shape=[p.shape[0], p.shape[1]])    
    mask[p == l] = 1.0
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return float(cX) / float(p.shape[0]), float(cY) / float(p.shape[1])

def _extract_msc_signature_data(p, f, lamda):
    msc_nodes = {}
    msc_edges = {}
    gamma_plus = f > lamda
    gamma_minus = f < -lamda 
    labels = set(p.flatten())
    maxvol = 0.0
    minvol = 1.0e10
    for l in labels:
        vol = float(np.count_nonzero(p == l))
        vol_plus = float(np.count_nonzero(p[gamma_plus] == l))
        vol_minus = float(np.count_nonzero(p[gamma_minus] == l))
        r_plus = vol_plus / vol
        r_minus = vol_minus / vol
        rtot = r_plus + r_minus
        if rtot > 0.0:
            cx, cy = _compute_center(p, l)
            msc_nodes[l] = {"r_plus": r_plus, "r_minus": r_minus, "cx": cx, "cy": cy, "vol": vol}
            maxvol = max(rtot * vol, maxvol)
            minvol = min(rtot * vol, minvol)

    maxborder = 0.0
    for l in msc_nodes:
        contour = _extract_contour(l, p, apron=0, norm=False)
        mark = np.zeros(contour.shape[0] - 1, dtype=np.int32)  # Changed from np.int to np.int32
        dx = contour[1:] - contour[:-1]
        for i, x in enumerate(contour[:-1]):
            n = np.cross([0.0, 0.0, 1.0], [dx[i][0], dx[i][1], 0.0])[:-1]
            if p[int(x[0] + n[0])][int(x[1] + n[1])] != l:
                mark[i] = int(p[int(x[0] + n[0])][int(x[1] + n[1])])
            else:
                mark[i] = int(p[int(x[0] - n[0])][int(x[1] - n[1])])
        for m in set(mark):
            if m != l and m in msc_nodes:
                edge = (int(min(l, m)), int(max(l, m)))
                if edge not in msc_edges:
                    border = float(np.count_nonzero(mark == m))
                    if border > 5.0:
                        msc_edges[edge] = border
                        maxborder = max(border, maxborder)
    return msc_nodes, msc_edges, minvol, maxvol, maxborder

def _msc_graph_node(c, e, tau, maxvol, config):
    roi = config['roi']
    name = config['classnames'][c]
    cmap = cm.get_cmap(config['kde_colormap'][c])
    color = cmap(tau)
    color_rgb = colorRGB(cmap(1.0))
    rtot = e['r_plus'] + e['r_minus']
    rho = 0.01 * max(roi[3] - roi[2], roi[1] - roi[0])
    radius = rho * (3.0 + 4.0 * rtot * e['vol'] / maxvol)
    x0 = roi[0] + e['cx'] * (roi[1] - roi[0]) 
    y0 = roi[2] + e['cy'] * (roi[3] - roi[2]) 
    pol = []
    for s in range(32):
        theta = 2.0 * float(s) / 31 * np.pi
        pol.append([x0 + radius * np.cos(theta), y0 + radius * np.sin(theta)])
    gp = geojson.Polygon([pol])
    properties = {}
    properties['color'] = [int(255 * color[0]), int(255 * color[1]), int(255 * color[2])]
    properties['classification'] = {'name': name, 'colorRGB': color_rgb}
    properties['isLocked'] = False
    properties['measurements'] = []
    feature = geojson.Feature(geometry=gp, id='PathAnnotationObject', properties=properties)
    return feature

def _msc_signature_glyph(c, r, e, maxvol, config):
    roi = config['roi']
    rtot = e['r_plus'] + e['r_minus']
    rho = 0.01 * max(roi[3] - roi[2], roi[1] - roi[0])
    h = rho * (1.0 + 2.0 * rtot * e['vol'] / maxvol)
    w = rho * (3.0 + 6.0 * rtot * e['vol'] / maxvol)
    
    name = config['classnames'][c]
    color = cm.get_cmap(config['kde_colormap'][c])(1.0)
    color_rgb = colorRGB(color)
    ratio = r / rtot
    y0 = roi[2] + e['cy'] * (roi[3] - roi[2]) 
    if r != e['r_plus']: 
        y0 -= h 
    x0 = roi[0] + e['cx'] * (roi[1] - roi[0]) - w / 2
    pol = [[x0, y0], [x0 + ratio * w, y0], 
           [x0 + ratio * w, y0 + h], [x0, y0 + h], [x0, y0]]
    gp = geojson.Polygon([pol])
    properties = {}
    properties['color'] = [int(255 * color[0]), int(255 * color[1]), int(255 * color[2])]
    properties['classification'] = {'name': name, 'colorRGB': color_rgb}
    properties['isLocked'] = False
    properties['measurements'] = []
    feature = geojson.Feature(geometry=gp, id='PathAnnotationObject', properties=properties)
    return feature

def _msc_graph_edge(edge, tau, c, nodes, config):
    roi = config['roi']
    name = config['classnames'][c]
    cmap = cm.get_cmap(config['kde_colormap'][c])
    color = cmap(tau)
    color_rgb = colorRGB(cmap(1.0))
    rho = 0.005 * max(roi[3] - roi[2], roi[1] - roi[0]) 
    delta = 6 * rho
    n0 = nodes[edge[0]]
    n1 = nodes[edge[1]]
    x0 = roi[0] + n0['cx'] * (roi[1] - roi[0]) 
    y0 = roi[2] + n0['cy'] * (roi[3] - roi[2]) 
    x1 = roi[0] + n1['cx'] * (roi[1] - roi[0]) 
    y1 = roi[2] + n1['cy'] * (roi[3] - roi[2])
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    v = [(x1 - x0) / length, (y1 - y0) / length]
    n = np.cross([0.0, 0.0, 1.0], [v[0], v[1], 0.0])[:-1]
    hw = rho * (0.25 + tau * 0.5)    
    poly = [[x0 + delta * v[0] - hw * n[0], y0 + delta * v[1] - hw * n[1]], 
            [x0 + delta * v[0] + hw * n[0], y0 + delta * v[1] + hw * n[1]],
            [x1 - delta * v[0] + hw * n[0], y1 - delta * v[1] + hw * n[1]], 
            [x1 - delta * v[0] - hw * n[0], y1 - delta * v[1] - hw * n[1]],
            [x0 + delta * v[0] - hw * n[0], y0 + delta * v[1] - hw * n[1]]]
    gl = geojson.Polygon([poly])
    properties = {}
    properties['color'] = [int(255 * color[0]), int(255 * color[1]), int(255 * color[2])]
    properties['classification'] = {'name': name, 'colorRGB': color_rgb}
    properties['isLocked'] = False
    properties['measurements'] = []
    feature = geojson.Feature(geometry=gl, id='PathAnnotationObject', properties=properties)
    return feature

def msc_signature(c0, c1, f0, f1, config):
    msc_sig = []
    
    grid_points = 2**8  # Grid points in each dimension
    roi = config['roi']
    xmin, xmax = roi[0], roi[1]
    ymin, ymax = roi[2], roi[3]
    x_space = np.linspace(xmin, xmax, grid_points)
    y_space = np.linspace(ymin, ymax, grid_points)
    x_sample, y_sample = [np.array(a.flat) for a in np.meshgrid(x_space, y_space)]
    f = f0 - f1 
    maxdensity = np.max(f)
    mindensity = np.min(f)
    f_lin = f.T.reshape(f.shape[0] * f.shape[1])
    sample_points = np.stack([x_sample, y_sample]) 

    msc = morse_smale.nn_partition(sample_points, f_lin, k_neighbors=100, persistence_level=0.0)
    p = msc.partitions.reshape(f.shape[0], f.shape[1])
    msc_min = sample_points[:, msc.min_indices].T
    msc_max = sample_points[:, msc.max_indices].T
    
    msc_nodes, msc_edges, minvol, maxvol, maxborder = _extract_msc_signature_data(p.T, f.T, lamda=config['density_threshold'] * min(-mindensity, maxdensity))
    
    for k in msc_nodes.keys():
        if in_border(msc_min[k], roi):
            continue
        e = msc_nodes[k]
        if e['r_plus'] > e['r_minus']:
            msc_sig.append(_msc_graph_node(c0, e, e['r_plus'] * e['vol'] / maxvol, maxvol, config))
        else:
            msc_sig.append(_msc_graph_node(c1, e, e['r_minus'] * e['vol'] / maxvol, maxvol, config))
        if e['r_plus'] > 0.0: 
            msc_sig.append(_msc_signature_glyph(c0, e['r_plus'], e, maxvol, config))
        if e['r_minus'] > 0.0:
            msc_sig.append(_msc_signature_glyph(c1, e['r_minus'], e, maxvol, config))
        for edge in msc_edges.keys():
            if in_border(msc_min[edge[0]], roi) or in_border(msc_min[edge[1]], roi):
                continue
            r0 = msc_nodes[edge[0]]['r_plus'] + msc_nodes[edge[1]]['r_plus']
            r1 = msc_nodes[edge[0]]['r_minus'] + msc_nodes[edge[1]]['r_minus']
            c = c0 if r0 > r1 else c1
            msc_sig.append(_msc_graph_edge(edge, msc_edges[edge] / maxborder, c, msc_nodes, config))
    return msc_sig

def kde_contours_fft(x, y, l, w, channel_id, roi, threshold=0.0, filtering=True, weighting=True, cmap='Reds', debug_pic='test_kde.png', kernel='box', norm=2, bw=384, gamma=1.0, contrast=True):
    xf = np.array(x)
    yf = np.array(y)
    wf = np.array(w).reshape(xf.shape[0], -1)
    
    if filtering:
        filter = np.array(l) == channel_id
        xf = xf[filter]
        yf = yf[filter]
        wf = wf[filter, :]
        wc = wf[:, channel_id]
        if contrast:
            for i in range(wf.shape[0]):
                second_prob = max(0.01, second_largest(wf[i, :]))
                wc[i] = (wc[i] / second_prob) ** gamma 
            wf = wc    
        else:
            wf = wf[:, channel_id]

    if xf.shape[0] == 0:
        return 0, 0

    xmin, xmax = roi[0], roi[1]
    ymin, ymax = roi[2], roi[3]
    grid_points = 2**8  
    x_space = np.linspace(xmin, xmax, grid_points)
    y_space = np.linspace(ymin, ymax, grid_points)
    x_sample, y_sample = [np.array(a.flat) for a in np.meshgrid(x_space, y_space)]
    xx, yy = np.mgrid[xmin:xmax:grid_points*1j, ymin:ymax:grid_points*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([xf, yf])
    if weighting:
        kernel = st.gaussian_kde(values, weights=wf)
    else:
        kernel = st.gaussian_kde(values)
    
    f = np.reshape(kernel(positions).T, xx.shape)
 
    f_lin = f.T.reshape(f.shape[0] * f.shape[1])
    sample_points = np.stack([x_sample, y_sample]) 
    msc = morse_smale.nn_partition(sample_points, f_lin, k_neighbors=100, persistence_level=0.0)
    p = msc.partitions.reshape(f.shape[0], f.shape[1])
    msc_cont = _extract_partition_contours(p.T)
    kernel.set_bandwidth(bw_method=kernel.factor * bw)
    f = np.reshape(kernel(positions).T, xx.shape)
   
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca()
    ax.set_aspect('equal')
    plt.axis('off')
    ax.imshow(f.T, extent=[xmin, xmax, ymax, ymin])
    contour_set = ax.contour(xx, yy, f, cmap=cmap)
    
    if weighting:
        ax.scatter(xf, yf, c=wf, marker="+", cmap=cmap)
    else:
        ax.scatter(xf, yf, c='gray', marker="+")

    ax.scatter(sample_points[0, msc.min_indices], sample_points[1, msc.min_indices],  c='red', s=100)
    ax.scatter(sample_points[0, msc.max_indices], sample_points[1, msc.max_indices],  c='blue', s=100)

    msc_dmin = f_lin[msc.min_indices]
    msc_dmax = f_lin[msc.max_indices]
    msc_min = sample_points[:, msc.min_indices].T
    msc_max = sample_points[:, msc.max_indices].T
    
    plt.gca().invert_yaxis()
    plt.savefig(debug_pic, bbox_inches='tight', pad_inches=0)
    plt.close()

    extent = [xmin, xmax, ymax, ymin]
    
    return contour_set.allsegs, contour_set.levels, msc_cont, msc_min, msc_max, msc_dmin, msc_dmax, f, extent

def polygon_from_box(b):
    pol = [[b[0] - b[2] / 2, b[1] - b[3] / 2], [b[0] + b[2] / 2, b[1] - b[3] / 2], 
           [b[0] + b[2] / 2, b[1] + b[3] / 2], [b[0] - b[2] / 2, b[1] + b[3] / 2], 
           [b[0] - b[2] / 2, b[1] - b[3] / 2]] 
    gp = geojson.Polygon([pol])
    return gp

def polygon_glyph(p, color, name, color_rgb, size=400.0):
    pol = [[p[0] - size / 2, p[1]], [p[0], p[1] - size / 2], 
           [p[0] + size / 2, p[1]], [p[0], p[1] + size / 2], 
           [p[0] - size / 2, p[1]]]
    gp = geojson.Polygon([pol])
    properties = {}
    properties['color'] = color
    properties['classification'] = {'name': name, 'colorRGB': color_rgb}
    properties['isLocked'] = False
    properties['measurements'] = []
    feature = geojson.Feature(geometry=gp, id='PathAnnotationObject', properties=properties)
    return feature

def kde_contours_all(x, y, labels, p_all, config):
    annotations = []
    partitions = []
    critical_points = []
    signatures = []
    class_ids = config['class_id']
    classnames = config['classnames']
    colors = [colorRGB(cm.get_cmap(config['kde_colormap'][i])(1.0)) for i in range(len(classnames))]
    density = []  # Initialize the density list

    for colornum, class_id in tqdm(enumerate(class_ids), desc="Isocontours"):
        kde_contours, kde_levels, msc_contours, msc_min, msc_max, msc_dmin, msc_dmax, d, extent = kde_contours_fft(
            x, y, labels, p_all, class_id, roi=config['roi'], threshold=config['threshold'], filtering=config['kde_filtering'],
            weighting=config['kde_weighting'], cmap=config['kde_colormap'][class_id],
            kernel=config['kde_kernel'], bw=config['kde_bandwidth'],
            norm=config['kde_norm'], gamma=config['gamma'], contrast=config['contrast']
        )
        density.append(d)  # Append the density to the list
        if kde_contours == 0:
            continue
        num_levels = config['num_levels']
        cmap = cm.get_cmap(config['kde_colormap'][class_id])
        level_colors = cmap(np.linspace(config['threshold'], 1, num_levels))

        dmin = np.min(msc_dmin)
        dmax = np.max(msc_dmax)

        for n, l in enumerate(kde_contours[-num_levels - 1:-1]):
            for i, c in enumerate(l):
                if len(c) > 1:
                    pol = []
                    for p in c:
                        pol.append([int(p[0]), int(p[1])])
                    if float(pol[0][0] - pol[-1][0]) ** 2 + float(pol[0][1] - pol[-1][1]) ** 2 < 0.5:
                        gp = geojson.Polygon([pol])
                        properties = {}
                        properties['color'] = [int(255 * level_colors[n][0]), int(255 * level_colors[n][1]), int(255 * level_colors[n][2])]
                        properties['classification'] = {'name': classnames[class_id], 'colorRGB': colors[colornum]}
                        properties['isLocked'] = False
                        properties['measurements'] = []
                        feature = geojson.Feature(geometry=gp, id='PathAnnotationObject', properties=properties)
                        annotations.append(feature)

        x0 = extent[0]
        dx = extent[1] - extent[0]
        y0 = extent[3]
        dy = extent[2] - extent[3]
        for n, c in enumerate(msc_contours):
            if in_border(msc_min[n], extent):
                continue
            tmin = (msc_dmin[n] - dmin) / (dmax - dmin)
            tmax = (msc_dmax[n] - dmin) / (dmax - dmin)
            cmin = cmap(0.5)
            cmax = cmap(1.0)
            cmini = [int(255 * cmin[0]), int(255 * cmin[1]), int(255 * cmin[2])]
            cmaxi = [int(255 * cmax[0]), int(255 * cmax[1]), int(255 * cmax[2])]
            glyph_min = polygon_glyph(msc_min[n], cmini, classnames[class_id], colors[colornum], size=100.0 + tmin * 300.0)
            glyph_max = polygon_glyph(msc_max[n], cmaxi, classnames[class_id], colors[colornum], size=100.0 + tmax * 300.0)
            critical_points.append(glyph_min)
            critical_points.append(glyph_max)

            tavg = 0.5 * (tmin + tmax)
            cavg = cmap(tavg)
            cavgi = [int(255 * cavg[0]), int(255 * cavg[1]), int(255 * cavg[2])]

            if len(c) > 1:
                pol = []
                for p in c:
                    pol.append([int(x0 + p[0] * dx), int(y0 + p[1] * dy)])
                if float(pol[0][0] - pol[-1][0]) ** 2 + float(pol[0][1] - pol[-1][1]) ** 2 < 0.5:
                    gp = geojson.Polygon([pol])
                    properties = {}
                    properties['color'] = cavgi
                    properties['classification'] = {'name': classnames[class_id], 'colorRGB': colors[colornum]}
                    properties['isLocked'] = False
                    properties['measurements'] = []
                    feature = geojson.Feature(geometry=gp, id='PathAnnotationObject', properties=properties)
                    partitions.append(feature)

        for p in config['msc_signatures']:
            if p[0] < len(density) and p[1] < len(density):
                signatures.append(msc_signature(p[0], p[1], density[p[0]], density[p[1]], config))

    return annotations, partitions, critical_points, signatures

def extract_sample(CX, CY, img, size=(144, 144, 3)):
    ret = np.zeros(shape=size, dtype=img.dtype)
    DX, DY = size[0] / 2, size[1] / 2
    Xmin, Ymin = int(max(0, CX - DX)), int(max(0, CY - DY))
    Xmax, Ymax = int(min(CX + DX, img.shape[0] - 1)), int(min(CY + DY, img.shape[1] - 1))
    org = int((size[0] / 2 - (CX - Xmin))), int((size[1] / 2 - (CY - Ymin)))
    ret[org[0]:org[0] + Xmax - Xmin, org[1]:org[1] + Ymax - Ymin, :] = np.copy(img[Xmin:Xmax, Ymin:Ymax, :])
    return ret

def in_border(x, roi):
    return x[0] == roi[0] or x[0] == roi[1] or x[1] == roi[2] or x[1] == roi[3]

OPENCV_METHOD = cv2.HISTCMP_BHATTACHARYYA

if __name__ == '__main__':
    overall_start = time.time()

    if len(sys.argv) < 2:
        print('Usage: histo_msc.py wsi_image [config json] [square_size]')
        sys.exit(1)

    wsi_fname = sys.argv[1]
    config_json = sys.argv[2] if len(sys.argv) > 2 else 'config_msc.json'
    current_size = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    with open(config_json) as json_file:
        config = json.load(json_file)

    # Load slide
    osh = openslide.OpenSlide(wsi_fname)
    xdim, ydim = osh.level_dimensions[0]
    openslidelevel = config['openslidelevel'] 
    tilesize = config['tilesize'] 
    patchsize = config['patchsize']  
    model_dir = config['model_dir']
    classnames = config['classnames']
    colors = [colorRGB(cm.get_cmap(config['kde_colormap'][i])(1.0)) for i in range(len(classnames))]

    roi = config['roi']
    image_resolution = config['image_resolution']

    startx = int(roi[0] / image_resolution)
    endx = int(roi[1] / image_resolution)
    starty = int(roi[2] / image_resolution)
    endy = int(roi[3] / image_resolution)

    if endx == 0:
        endx = xdim
    if endy == 0:
        endy = ydim

    config['roi'] = [startx, endx, starty, endy]

    scalefactor = int(osh.level_downsamples[openslidelevel])
    paddingsize = patchsize // 2 * scalefactor
    nucleus_size = config['nucleus_size']
    img_topleft = np.asarray(osh.read_region((0, 0), openslidelevel, 
                                               (nucleus_size, nucleus_size)), dtype=np.float32)[..., :3]
    hist_background_ref = compute_histogram(img_topleft)
    step = round(tilesize * scalefactor)

    nuclei_dir = os.path.split(wsi_fname)[0] + '/' + os.path.splitext(os.path.split(wsi_fname)[1])[0] + '_nuclei'
    json_critical_fname = os.path.split(wsi_fname)[0] + '/' + os.path.splitext(os.path.split(wsi_fname)[1])[0] + '_cp.json'
    json_partitions_fname = os.path.split(wsi_fname)[0] + '/' + os.path.splitext(os.path.split(wsi_fname)[1])[0] + '_msc.json'
    json_contours_fname = os.path.split(wsi_fname)[0] + '/' + os.path.splitext(os.path.split(wsi_fname)[1])[0] + '_cont.json'
    json_annotated_fname = os.path.split(wsi_fname)[0] + '/' + os.path.splitext(os.path.split(wsi_fname)[1])[0] + '_ann.json'
    csv_inference_fname = os.path.split(wsi_fname)[0] + '/' + os.path.splitext(os.path.split(wsi_fname)[1])[0] + '_inf.csv'
    json_signatures_basename = os.path.split(wsi_fname)[0] + '/' + os.path.splitext(os.path.split(wsi_fname)[1])[0]

    if not os.path.isdir(nuclei_dir):
        os.mkdir(nuclei_dir)
    else:
        if not config['skip_detect']: 
            shutil.rmtree(nuclei_dir)
            os.mkdir(nuclei_dir)

    # Start detection timing
    detection_start = time.time()
    if not config['skip_detect']:
        detect_path = config['detect_model_dir'] + '/' + config['detect_model']  
        detect_model = torch.hub.load("ultralytics/yolov5", 'custom', path=detect_path, force_reload=True)
        id = 0    
        border_boxes_per_tile = {}

        border_threshold = nucleus_size / 3
        imgsize = tilesize + 2 * paddingsize
        
        for y in tqdm(range(starty, endy, step), desc="Detecting nuclei"):
            for x in range(startx, endx, step):    
                img = np.asarray(osh.read_region((x - paddingsize, y - paddingsize), openslidelevel, 
                                                  (tilesize + 2 * paddingsize, tilesize + 2 * paddingsize)), dtype=np.float32)[..., :3]

                hist_tile = compute_histogram(img)
                dt = cv2.compareHist(hist_tile, hist_background_ref, OPENCV_METHOD)
                boxes = {}
                if dt >= config['background_threshold']:
                    results = detect_model(img)
                    df = results.pandas().xyxy[0]
                    for i_row, row in df.iterrows():
                        if row['xmin'] < border_threshold or row['xmax'] > imgsize - border_threshold or  row['ymin'] < border_threshold or row['ymax'] > imgsize - border_threshold :
                            continue
                        dx = (row['xmax'] - row['xmin']) / 2
                        dy = (row['ymax'] - row['ymin']) / 2
                        px = row['xmin'] + dx
                        py = row['ymin'] + dy
                        patch = extract_sample(px, py, img, size=(config['nucleus_size'], config['nucleus_size'], 3))
                        xn = int(px) * scalefactor + (x - paddingsize)
                        yn = int(py) * scalefactor + (y - paddingsize)
                        wn = int(2 * dx)
                        hn = int(2 * dy)
                        plt.imsave(nuclei_dir + '/' + str(id) + '_' + str(xn)
                                   + '_' + str(yn) + '_' + str(wn) + '_' + str(hn) + '.jpg', np.uint8(patch))  
                        if row['xmin'] <= patchsize  or row['xmax'] >= imgsize - patchsize  or row['ymin'] <= patchsize  or row['ymax'] >= imgsize - patchsize:
                            boxes[id] = [xn, yn, wn, hn]
                        id += 1
        border_boxes_per_tile[(y, x)] = boxes
        del detect_model
        # We skip border clean for now as in original code it was optional.
        detection_end = time.time()

  # Classification timing
    classification_start = time.time()
    if not config['skip_inference']:
        i, xv, yv, wv, hv, l, p = squeezenet_inference(model_dir, nuclei_dir)
        write_csv_inference(i, xv, yv, wv, hv, l, p, csv_inference_fname)
    else:
        i, xv, yv, wv, hv, l, p = read_csv_inference(csv_inference_fname)
    classification_end = time.time()

    # Annotation timing
    annotation_start = time.time()
    allobjects = []
    for idx in range(len(xv)):
        b = [xv[idx], yv[idx], wv[idx], hv[idx]]
        gp = polygon_from_box(b)
        label = l[idx]
        c = cm.get_cmap(config['kde_colormap'][label])(p[idx][label])
        properties = {}
        properties['color'] = [int(255 * c[0]), int(255 * c[1]), int(255 * c[2])]
        properties['classification'] = {'name': classnames[label], 'colorRGB': colors[label]}
        properties['isLocked'] = False
        properties['measurements'] = []
        feature = geojson.Feature(geometry=gp, id='PathAnnotationObject', properties=properties)
        allobjects.append(feature)

    annotations, partitions, critical, signatures = kde_contours_all(xv, yv, l, p, config)

    with open(json_contours_fname, 'w') as outfile:
        geojson.dump(annotations, outfile)

    with open(json_partitions_fname, 'w') as outfile:
        geojson.dump(partitions, outfile)

    with open(json_critical_fname, 'w') as outfile:
        geojson.dump(critical, outfile)

    for n, s in enumerate(signatures):
        # Include ROI label in signature filename
        json_signatures_fname = f"{json_signatures_basename}_{sys.argv[3]}_sig_{n}.json"
        # Ensure polygons are closed
        for feature in s:
            if feature['geometry']['type'] == 'Polygon':
                for ring in feature['geometry']['coordinates']:
                    if ring[0] != ring[-1]:  # If not closed, close it
                        ring.append(ring[0])
        with open(json_signatures_fname, 'w') as outfile:
            geojson.dump(s, outfile)

    annotation_end = time.time()

    # Save annotation file based on current_size
    output_dir = os.path.split(wsi_fname)[0] + '/'
    annotation_fname = output_dir + f'annotation_{current_size}um.json'
    with open(annotation_fname, 'w') as outfile:
        geojson.dump(allobjects, outfile)

    # Compute times
    overall_end = time.time()
    total_time = overall_end - overall_start
    detection_time = detection_end - detection_start
    classification_time = classification_end - classification_start
    annotation_time = annotation_end - annotation_start

    # Append timings
    timings_file = output_dir + 'timings.json'
    timing_data = {
        "Square_Size": current_size,
        "Total_Time": total_time,
        "Detection_Time": detection_time,
        "Classification_Time": classification_time,
        "Annotation_Time": annotation_time
    }

    if os.path.exists(timings_file):
        with open(timings_file, 'r') as f:
            all_timings = json.load(f)
    else:
        all_timings = []

    all_timings.append(timing_data)
    with open(timings_file, 'w') as f:
        json.dump(all_timings, f)

    print(f"Completed run for {current_size}um square.")
    print(f"Total time = {total_time} s")

    # Print all generated files
    files_to_print = [json_contours_fname, json_partitions_fname, json_critical_fname, 
                      json_annotated_fname, annotation_fname, timings_file]
    files_to_print.extend([json_signatures_basename + str(n) + '_sig.json' for n in range(len(signatures))])
    for file_name in files_to_print:
        print(file_name)