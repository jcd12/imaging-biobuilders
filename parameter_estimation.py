#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage import io, morphology, color, util, filters, draw
from scipy import stats
import numpy as np
from PIL import Image
import pandas as pd
import sknw
import argparse

import warnings
warnings.filterwarnings('ignore')


# In[2]:


parser = argparse.ArgumentParser(description='Processes a set of fluorescent or segmented images of mycelia and extracts relevant features.')

parser.add_argument('--img_folder', dest="img_folder", type=str, required=True,
                    help='destination folder containing the image data set')
parser.add_argument('--overview_file', dest="overview_file", type=str, required=True,
                    help='csv file in img_folder containing image details. see github repo for an example.')


parser.add_argument('--fluorescence_dim', dest="fluorescence_dim", type=int, required=True,
                    help='colour channel to look for mycelium in (R=0, G=1, B=2).')
parser.add_argument('--hours', dest="n_hours", type=float, required=True,
                    help='number of hours the mycelia have been growing. Use to scale e.g. branch frequency.')

parser.add_argument('--hyphal_element_length', dest="hyphal_element_length_pixels", type=int, default=50,
                    help='estimated number of pixels per hyphal element. used for scaling parameters for generative mycelium model.')
parser.add_argument('--small_object_limit', dest="small_object_limit", type=int, required=False, default=10000,
                    help='minimum size of connected binary pixels to keep when preprocessing binary image. larger values remove more noise')
parser.add_argument('--small_object_limit_skeleton', dest="small_object_limit_skeleton", type=int, required=False, default=1000,
                    help='minimum size of connected binary pixels to keep when preprocessing binary, skeletonized image. larger values remove more noise')
parser.add_argument('--median_filter_kernel', dest="median_filter_kernel", type=int, required=False, default=5,
                    help='size of median filter kernel used in smoothing edges of mycelia during preprocessing.')
parser.add_argument('--genotype', dest="genotype", type=str, required=False, default="all",
                    help="""only use images connected to a specific genotype, as per the gene column in the image csv. "all" includes all images.""")


args = parser.parse_args()
#args = parser.parse_args(["--img_folder", "data/", "--overview_file", "image_overview.csv", "--fluorescence_dim", "2", "--hours", "24", "--genotype", "wt"])




img_folder = args.img_folder
overview_file = args.overview_file
pixels_per_hyphal_element = args.hyphal_element_length_pixels
n_hours = args.n_hours
small_object_limit = args.small_object_limit
small_object_limit_skeleton = args.small_object_limit_skeleton
median_filter_kernel = args.median_filter_kernel
fluorescence_dim = args.fluorescence_dim

genotype = args.genotype




def import_fluorescent_img(fname, fluorescence_dim):
    
    fl_img  = io.imread(fname)
    fl_img = fl_img[:,:,fluorescence_dim]
    fl_img = util.img_as_ubyte(fl_img)
    
    return fl_img

def preprocess_image(fluorescent_img, threshold, small_object_limit, small_object_limit_skeleton, median_filter_kernel):

    bin_img = fluorescent_img.copy()
    bin_img[bin_img >= threshold] = 255
    bin_img[bin_img < threshold] = 0

    bin_img = bin_img.astype(np.bool)

    bin_img = morphology.remove_small_objects(bin_img, min_size=small_object_limit, connectivity=2)

    bin_img = filters.median(bin_img, selem=morphology.square(median_filter_kernel))

    ske_img = morphology.skeletonize(bin_img)
    ske_img = morphology.remove_small_objects(ske_img, min_size=small_object_limit_skeleton, connectivity=2)
    
    return ske_img
    
    

def calculate_radius(p1, p2, p3):
    # from https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle/50974391#50974391
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return None

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return radius

def calculate_curvatures(points, dist=30):
    
    curvatures_all = list()
    
    for i in range(len(ps)):

        cur_curvatures = list()
        cur_line = ps[i]

        for i in range(dist, len(cur_line[:,0])-dist):

            xm1, x, xp1 = cur_line[i-dist,0], cur_line[i,0], cur_line[i+dist,0]
            ym1, y, yp1 = cur_line[i-dist,1], cur_line[i,1], cur_line[i+dist,1]

            r = calculate_radius((xm1, ym1), (x,y), (xp1, yp1))
            if r is not None:
                cur_curvatures.append(1/r)
                
        curvatures_all.extend(cur_curvatures)
    
    if len(curvatures_all) != 0:
        return curvatures_all
    else:
        return None
    
def calculate_angles(graph, n_points = 10, n_min = 10):

    nodes = graph.nodes()
    angles = list()

    for i in nodes:

        if len(graph.edges(i)) != 3:
            continue

        neighbors_idx = [j for j in graph.neighbors(i)]
        neighbors_pts = np.array([graph[i][j]['pts'] for j in neighbors_idx])
        neighbors_len = [len(j) for j in neighbors_pts]

        if sum(np.array(neighbors_len) > n_min) != 3:
            continue

        c = graph.nodes(data=True)[i]["o"]

        # find closest n points
        p = [0, 0, 0]
        for i in range(3):
            p[i] = neighbors_pts[i][np.argsort(np.sum((neighbors_pts[i] - c)**2, axis=1))[:n_points],:]

        # linear regression through all sets
        sets = [[0,1], [1,2], [0,2]]
        r_vals = list()
        for i in range(3):
            a, b = sets[i][0], sets[i][1]
            temp_data = np.concatenate((p[a], p[b]), axis=0)

            _, _, r_value, _, _ = stats.linregress(temp_data)
            r_squared = r_value**2

            r_vals.append(r_squared)

        best_idx = sets[np.argmax(r_vals)]
        last_idx = list(set([0,1,2]).difference(set(best_idx)))[0]

        # create two lines, one through two best sets and one through last set. 
        temp_data = np.concatenate((p[best_idx[0]], p[best_idx[1]]), axis=0)
        slope1, _, _, _, _ = stats.linregress(temp_data)
        slope2, _, _, _, _ = stats.linregress(p[last_idx][:,0], p[last_idx][:,1])
        
        if np.isnan(slope1): slope1=1000
        if np.isnan(slope2): slope2=1000

        alpha, beta = np.arctan(slope1)*180/3.14, np.arctan(slope2)*180/3.14
        angle = alpha - beta
        #angle = (m1 - m2)*180/3.14
        if angle < 0:
            angle = -angle
        if angle > 90:
            angle = 180 - angle

        angles.append(angle)
        
    return angles

def box_fractal_dim(Z, threshold=0.9):
    # From https://github.com/rougier/numpy-100 (#87)
    # Only for 2d image
    assert(len(Z.shape) == 2)
    
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def get_dist_params(measure, data, angle_max=40):
    if measure == "curvature":
        dist_name = "gamma"
        dist = getattr(stats, dist_name)
        max_curv = max(data)
        data = [angle_max*i/max_curv for i in data]
        fit = dist.fit(data)
        p = [fit[0], fit[2]] #a, scale in scipy
    elif measure == "branching":
        dist_name = "beta"
        max_angle = max(data)
        data = [i/max_angle for i in data]
        dist = getattr(stats, dist_name)
        p = dist.fit(data)[0:2] #a, b in scipy
    return p




img_overview = pd.read_csv(img_folder+overview_file)

if genotype != "all":
    img_overview = img_overview.loc[(img_overview["gene"] == genotype)].reset_index(drop=True)




branching_freqs = list()
curvatures_all = list()
angles_all = list()
internodal_all = list()
box_counting_dims_all = list()

for idx in range(len(img_overview)):


    fluorescent_img = import_fluorescent_img(img_folder + img_overview.at[idx,"target_img_fname"], fluorescence_dim=fluorescence_dim)
    
    skeleton = preprocess_image(fluorescent_img, img_overview.at[idx,"binary_threshold"],
                                small_object_limit, small_object_limit_skeleton, median_filter_kernel)
    
    # build graph from skeleton
    graph = sknw.build_sknw(skeleton)

    # draw node by o
    nodes = graph.nodes()
    bs = np.array([nodes[i]['o'] for i in nodes if len(graph.edges(i))>=3])
    ts = np.array([nodes[i]['o'] for i in nodes if len(graph.edges(i))<=2])
    
    ps = list()
    internodal_lengths = list()

    for (s,e) in graph.edges():
        ps.append(graph[s][e]['pts'].astype(np.int32))
        
        if len([j for j in graph.neighbors(s)]) == 3 and len([j for j in graph.neighbors(s)]) == 3:
            internodal_lengths.append(len(graph[s][e]['pts'].astype(np.int32)))
        
    curvatures = calculate_curvatures(ps)
        
    branch_occurence = (len(bs))/graph.size(weight="weight")    
    n_hyphal_elements = (graph.size(weight="weight"))/pixels_per_hyphal_element
    
    curvatures_all.append(curvatures)
    branching_freqs.append(branch_occurence*pixels_per_hyphal_element/n_hours)
    
    angles = calculate_angles(graph)
    angles_all.append(angles)
    
    internodal_all.append(internodal_lengths)
    
    box_counting_dims_all.append(box_fractal_dim(skeleton))





branching_freq_avg = np.mean(branching_freqs)
curvature_params = get_dist_params("curvature", [item for sublist in curvatures_all for item in sublist]) #gamma dist, params scale
angle_params = get_dist_params("branching", [item for sublist in angles_all for item in sublist]) #beta dist, params a, b





export_df = pd.DataFrame([["branching_frequency", "curvature_gamma_a", "curvature_gamma_scale", "angle_beta_a", "angle_beta_b"], 
                          [branching_freq_avg] + list(curvature_params) + list(angle_params)])

export_df = export_df.rename(columns=export_df.iloc[0]).drop(export_df.index[0])




export_df.to_csv("graph_parameters.csv")

