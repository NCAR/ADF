#%%
print()
import numpy as np
try : import wasserstein
except: 
    print('  Wasserstein package is not installed so wasserstein distance cannot be used. Attempting to use wassertein distance will raise an error.')
    print('  To use wasserstein distance please install the wasserstein package in your environment: https://pypi.org/project/Wasserstein/ ')
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from shapely.geometry import Point
import cartopy
from shapely.prepared import prep
import glob
from math import ceil
import time
import dask
import os


#global num_iter, n_samples, data, ds, ht_var_name, tau_var_name, k, height_or_pressure

def cloud_regime_analysis(adf, wasserstein_or_euclidean = "euclidean", data_product='all', premade_cloud_regimes=None, lat_range=None, lon_range=None, only_ocean_or_land=False):
    """
    This script/function is designed to generate 2-D lat/lon maps of Cloud Regimes (CRs), as well as plots of the CR 
    centers themselves. It can fit data into CRs using either Wassertstein (AKA Earth Movers Distance) or the more conventional 
    Euclidean distance. To use this script, the user should add the appropriate COSP variables to the diag_var_list in the yaml file.
    The appropriate variables are FISCCP1_COSP for ISCCP, CLD_MISR for MISR, and CLMODIS for MODIS. All three should be added to 
    diag_var_list if you wish to preform analysis on all three. The user can also specify to preform analysis for just one or for 
    all three of the data products (ISCCP, MODIS, and MISR) that there exists COSP output for. A user can also choose to use only 
    a specfic lat and lon range, or to use data only over water or over land. Lastly if a user has CRs that they have custom made, 
    these can be passed in and the script will fit data into them rather than the premade CRs that the script already points to. 
    There are a total of 6 sets of premade CRs, two for each data product. One set made with euclidean distance and one set made 
    with Wasserstein distance for ISCCP, MODIS, and MISR. Therefore when the wasserstein_or_euclidean variables is changes it is 
    important to undertand that not only the distance metric used to fit data into CRs is changing, but also the CRs themselves 
    unless the user is passing in a set of premade CRs with the premade_cloud_regimes variable. 

    Description of kwargs:
    wasserstein_or_euclidean        -> Whether to use wasserstein or euclidean distance to fit CRs, enter "wassertein" for wasserstein or 
                                       "euclidean" for euclidean. This also changes the default CRs that data is fit into from ones created
                                       with kmeans using euclidean distance to ones using kmeans with wassertein distance.  Default is euclidean distance. 
    data_product                    -> Which data product to preform analysis for. Enter "ISCCP", "MODIS", "MISR" or "all". Default is "all"
    premade_cloud_regimes           -> If the user wishes to use custom CRs rather than the pre-loaded ones, enter them here as a path to a numpy 
                                       array of shape (k, n_tau_bins * n_pressure_bins)
    lat_range                       -> Range of latitudes to use enetered as a list, Ex. [-30,30]. Default is use all available latitudes
    lon_range                       -> Range of longitudes to use enetered as a list, Ex. [-90,90]. Default is use all available longitudes
    only_ocean_or_land              -> Set to "O" to preform analysis with only points over water, "L" for only points over land, or False 
                                       to use data over land and water. Default is False
    """

    global k, ht_var_name, tau_var_name, var, mat, mat_b, mat_o
    dask.config.set({"array.slicing.split_large_chunks": False})

    # Compute cluster labels from precomputed cluster centers with appropriate distance
    def precomputed_clusters(mat, cl, wasserstein_or_euclidean, ds):

        if wasserstein_or_euclidean == 'euclidean':
            cluster_dists = np.sum((mat[:,:,None] - cl.T[None,:,:])**2, axis = 1)
            cluster_labels_temp = np.argmin(cluster_dists, axis = 1)

        if wasserstein_or_euclidean == 'wasserstein':

            # A function to convert mat into the form required for the EMD calculation
            @njit()
            def stacking(position_matrix, centroids):
                centroid_list = []

                for i in range(len(centroids)):
                    x = np.empty((3,len(mat[0]))).T
                    x[:,0] = centroids[i]
                    x[:,1] = position_matrix[0]
                    x[:,2] = position_matrix[1]
                    centroid_list.append(x)

                return centroid_list
            
            # setting shape
            n1 = len(ds[tau_var_name])
            n2 = len(ds[ht_var_name])

            # Calculating the max distance between two points to be used as hyperparameter in EMD
            # This is not necesarily the only value for this variable that can be used, see Wasserstein documentation
            # on R hyper-parameter for more information
            R = (n1**2+n2**2)**0.5

            # Creating a flattened position matrix to pass wasersstein.PairwiseEMD
            position_matrix = np.zeros((2,n1,n2))
            position_matrix[0] = np.tile(np.arange(n2),(n1,1))
            position_matrix[1] = np.tile(np.arange(n1),(n2,1)).T
            position_matrix = position_matrix.reshape(2,-1)

            # Initialising wasserstein.PairwiseEMD
            emds = wasserstein.PairwiseEMD(R = R, norm=True, dtype=np.float32, verbose=1, num_threads=162)

            # Rearranging mat to be in the format necesary for wasserstein.PairwiseEMD
            events = stacking(position_matrix, mat)
            centroid_list = stacking(position_matrix, cl)
            emds(events, centroid_list)
            print("       -Calculating Wasserstein distances")
            print("       -Warning: This can be slow, but scales very well with additional processors")
            distances = emds.emds()
            labels = np.argmin(distances, axis=1)

            cluster_labels_temp = np.argmin(distances, axis=1)
            
        return cluster_labels_temp

    # This function is no longer used, no need to check
    # Plot the CR cluster centers
    def plot_hists(cl, cluster_labels, ht_var_name, tau_var_name, adf):
        #defining number of clusters
        k = len(cl)

        # setting up plots
        ylabels = ds[ht_var_name].values
        xlabels = ds[tau_var_name].values
        X2,Y2 = np.meshgrid(np.arange(len(xlabels)+1), np.arange(len(ylabels)+1))
        p = [0,0.2,1,2,3,4,6,8,10,15,99]
        cmap = mpl.colors.ListedColormap(['white', (0.19215686274509805, 0.25098039215686274, 0.5607843137254902), (0.23529411764705882, 0.3333333333333333, 0.6313725490196078), (0.32941176470588235, 0.5098039215686274, 0.6980392156862745), (0.39215686274509803, 0.6, 0.43137254901960786), (0.44313725490196076, 0.6588235294117647, 0.21568627450980393), (0.4980392156862745, 0.6784313725490196, 0.1843137254901961), (0.5725490196078431, 0.7137254901960784, 0.16862745098039217), (0.7529411764705882, 0.8117647058823529, 0.2), (0.9568627450980393, 0.8980392156862745,0.1607843137254902)])
        norm = mpl.colors.BoundaryNorm(p,cmap.N)
        plt.rcParams.update({'font.size': 12})
        fig_height = 1 + 10/3 * ceil(k/3)
        fig, ax = plt.subplots(figsize = (17, fig_height), ncols=3, nrows=ceil(k/3), sharex='all', sharey = True)

        aa = ax.ravel()
        boundaries = p
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        aa[1].invert_yaxis()

        # creating weights area for area weighted RFOs
        weights = cluster_labels.stack(z=('time','lat','lon')).lat.values
        weights = np.cos(np.deg2rad(weights))
        weights = weights[valid_indicies]
        indicies = np.arange(len(mat))

        # Plotting each cluster center
        for i in range (k):

            # Area Weighted relative Frequency of occurence calculation
            total_rfo_num = cluster_labels == i 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100
            total_rfo = total_rfo.values

            # Area weighting each histogram belonging to a cluster and taking the mean
            # if clustering was preformed with wasserstein distance and area weighting on, mean of i = cl[i], however if clustering was preformed with
            # conventional kmeans or wasseerstein without weighting, these two will not be equal
            indicies_i = indicies[np.where(cluster_labels_temp == i)]
            mean = mat[indicies_i] * weights[indicies_i][:,np.newaxis]
            if len(indicies_i) > 0: mean = np.sum(mean, axis=0) / np.sum(weights[indicies_i])
            else: mean = np.zeros(len(xlabels)*len(ylabels))
            
            mean = mean.reshape(len(xlabels),len(ylabels)).T           # reshaping into original histogram shape
            if np.max(mean) <= 1:                                      # Converting fractional data to percent to plot properly
                mean *= 100

            im = aa[i].pcolormesh(X2,Y2,mean,norm=norm,cmap=cmap)
            aa[i].set_title(f"CR {i+1}, RFO = {np.round(total_rfo,1)}%")

        # setting titles, labels, etc
        if data == "MISR": height_or_pressure = 'h'
        else: height_or_pressure = 'p'
        if height_or_pressure == 'p': fig.supylabel(f'Cloud-top Pressure ({ds[ht_var_name].units})', fontsize = 12, x = 0.09 )
        if height_or_pressure == 'h': fig.supylabel(f'Cloud-top Height ({ds[ht_var_name].units})', fontsize = 12, x = 0.09  )
        # fig.supxlabel('Optical Depth', fontsize = 12, y=0.26 )
        cbar_ax = fig.add_axes([0.95, 0.38, 0.045, 0.45])
        cb = fig.colorbar(im, cax=cbar_ax, ticks=p)
        cb.set_label(label='Cloud Cover (%)', size =10)
        cb.ax.tick_params(labelsize=9)
        #aa[6].set_position([0.399, 0.125, 0.228, 0.215])
        #aa[6].set_position([0.33, 0.117, 0.36, 0.16])
        #aa[-2].remove()

        bbox = aa[1].get_position()
        p1 = bbox.p1
        p0 = bbox.p0
        fig.suptitle(f'{data} Cloud Regimes', x=0.5, y=p1[1]+(1/fig_height * 0.5), fontsize=15)

        bbox = aa[-2].get_position()
        p1 = bbox.p1
        p0 = bbox.p0
        fig.supxlabel('Optical Depth', fontsize = 12, y=p0[1]-(1/fig_height * 0.5) )


        # Removing extra plots
        for i in range(ceil(k/3)*3-k):
            aa[-(i+1)].remove()
        save_path = adf.plot_location[0] + f'/{data}_CR_centers'
        plt.savefig(save_path)

        if adf.create_html:
            adf.add_website_data(save_path + ".png", var, adf.get_baseline_info("cam_case_name"))


    # Plot the CR centers of obs, baseline and test case 
    def plot_hists_baseline(cl, cluster_labels, cluster_labels_o, ht_var_name, tau_var_name, adf):
        # #defining number of clusters
        k = len(cl)

        # setting up plots
        ylabels = ds[ht_var_name].values
        xlabels = ds[tau_var_name].values
        X2,Y2 = np.meshgrid(np.arange(len(xlabels)+1), np.arange(len(ylabels)+1))
        p = [0,0.2,1,2,3,4,6,8,10,15,99]
        cmap = mpl.colors.ListedColormap(['white', (0.19215686274509805, 0.25098039215686274, 0.5607843137254902), (0.23529411764705882, 0.3333333333333333, 0.6313725490196078), (0.32941176470588235, 0.5098039215686274, 0.6980392156862745), (0.39215686274509803, 0.6, 0.43137254901960786), (0.44313725490196076, 0.6588235294117647, 0.21568627450980393), (0.4980392156862745, 0.6784313725490196, 0.1843137254901961), (0.5725490196078431, 0.7137254901960784, 0.16862745098039217), (0.7529411764705882, 0.8117647058823529, 0.2), (0.9568627450980393, 0.8980392156862745,0.1607843137254902)])
        norm = mpl.colors.BoundaryNorm(p,cmap.N)
        plt.rcParams.update({'font.size': 14})
        fig_height = (1 + 10/3 * ceil(k/3))*3
        fig, ax = plt.subplots(figsize = (17, fig_height), ncols=3, nrows=k, sharex='all', sharey = True)

        aa = ax.ravel()
        boundaries = p
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        if data != 'MISR': aa[1].invert_yaxis()

        # creating weights area for area weighted RFOs
        weights = cluster_labels.stack(z=('time','lat','lon')).lat.values
        weights = np.cos(np.deg2rad(weights))
        weights = weights[valid_indicies]
        indicies = np.arange(len(mat))

        for i in range(k):

            im = ax[i,0].pcolormesh(X2,Y2,cl[i].reshape(len(xlabels),len(ylabels)).T,norm=norm,cmap=cmap)
            ax[i,0].set_title(f" Observation CR {i+1}")

        # Plotting each cluster center (baseline)
        for i in range (k):
            # Area Weighted relative Frequency of occurence calculation
            total_rfo_num = cluster_labels_b == i 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels_b.lat)))
            total_rfo_denom = cluster_labels_b >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels_b.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100
            total_rfo = total_rfo.values

            # Area weighting each histogram belonging to a cluster and taking the mean
            # if clustering was preformed with wasserstein distance and area weighting on, mean of i = cl[i], however if clustering was preformed with
            # conventional kmeans or wasseerstein without weighting, these two will not be equal
            indicies_i = indicies[np.where(cluster_labels_temp_b == i)]
            mean = mat_b[indicies_i] * weights[indicies_i][:,np.newaxis]
            if len(indicies_i) > 0: mean = np.sum(mean, axis=0) / np.sum(weights[indicies_i])
            else: mean = np.zeros(len(xlabels)*len(ylabels))
            
            mean = mean.reshape(len(xlabels),len(ylabels)).T           # reshaping into original histogram shape
            if np.max(mean) <= 1:                                      # Converting fractional data to percent to plot properly
                mean *= 100

            im = ax[i,1].pcolormesh(X2,Y2,mean,norm=norm,cmap=cmap)
            ax[i,1].set_title(f"Baseline Case CR {i+1}, RFO = {np.round(total_rfo,1)}%")
        
        # Plotting each cluster center (test_case)
        for i in range (k):

            # Area Weighted relative Frequency of occurence calculation
            total_rfo_num = cluster_labels == i 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100
            total_rfo = total_rfo.values

            # Area weighting each histogram belonging to a cluster and taking the mean
            # if clustering was preformed with wasserstein distance and area weighting on, mean of i = cl[i], however if clustering was preformed with
            # conventional kmeans or wasseerstein without weighting, these two will not be equal
            indicies_i = indicies[np.where(cluster_labels_temp == i)]
            mean = mat[indicies_i] * weights[indicies_i][:,np.newaxis]
            if len(indicies_i) > 0: mean = np.sum(mean, axis=0) / np.sum(weights[indicies_i])
            else: mean = np.zeros(len(xlabels)*len(ylabels))
            
            mean = mean.reshape(len(xlabels),len(ylabels)).T           # reshaping into original histogram shape
            if np.max(mean) <= 1:                                      # Converting fractional data to percent to plot properly
                mean *= 100

            im = ax[i,2].pcolormesh(X2,Y2,mean,norm=norm,cmap=cmap)
            ax[i,2].set_title(f"Test Case CR {i+1}, RFO = {np.round(total_rfo,1)}%")
        
        # setting titles, labels, etc
        if data == "MODIS":
            ylabels = [0, 180, 310, 440, 560, 680, 800, 1000]
            xlabels = [0, 0.3, 1.3, 3.6, 9.4, 23, 60, 150]
            ax[0,0].set_yticks(np.arange(8))
            ax[0,0].set_xticks(np.arange(8))
            ax[0,0].set_yticklabels(ylabels)
            ax[0,0].set_xticklabels(xlabels)
            xticks = ax[0,0].xaxis.get_major_ticks()
            xticks[0].set_visible(False)
            xticks[-1].set_visible(False)

        if data == "MISR":
            xlabels = [0.2, 0.8, 2.4, 6.5, 16.2, 41.5, 100]
            ylabels = [ 0.25,0.75,1.25,1.75,2.25,2.75,3.5, 4.5, 6,8, 10, 12, 14, 16, 20  ]
            ax[0,0].set_yticks(np.arange(0,16,2)+0.5)
            ax[0,0].set_yticklabels(ylabels[0::2])
            ax[0,0].set_xticks(np.array([1,2,3,4,5,6,7]) -0.5)
            ax[0,0].set_xticklabels(xlabels, fontsize = 16)
            xticks = ax[0,0].xaxis.get_major_ticks()
            xticks[0].set_visible(False)
            xticks[-1].set_visible(False)

        if data == 'ISCCP':
            xlabels = [  0, 1.3, 3.6, 9.4, 22.6, 60.4, 450 ]
            ylabels = [  10, 180, 310, 440, 560, 680, 800, 1025]
            yticks = aa[i].get_yticks().tolist()
            xticks = aa[i].get_xticks().tolist()
            aa[i].set_yticks(yticks)
            aa[i].set_xticks(xticks)
            aa[i].set_yticklabels(ylabels)
            aa[i].set_xticklabels(xlabels)
            xticks = aa[i].xaxis.get_major_ticks()
            xticks[0].label1.set_visible(False)
            xticks[-1].label1.set_visible(False)
        

        if data == "MISR": height_or_pressure = 'h'
        else: height_or_pressure = 'p'
        if height_or_pressure == 'p': fig.supylabel(f'Cloud-top Pressure ({ds[ht_var_name].units})', x = 0.07 )
        if height_or_pressure == 'h': fig.supylabel(f'Cloud-top Height ({ds[ht_var_name].units})', x = 0.07  )

        if data == "MODIS":
            ylabels = [0, 180, 310, 440, 560, 680, 800, 1000]
            xlabels = [0, 0.3, 1.3, 3.6, 9.4, 23, 60, 150]
        if data == "MISR":
            x=1



        # fig.supxlabel('Optical Depth', fontsize = 12, y=0.26 )
        cbar_ax = fig.add_axes([0.95, 0.38, 0.045, 0.45])
        cb = fig.colorbar(im, cax=cbar_ax, ticks=p)
        cb.set_label(label='Cloud Cover (%)', size =10)
        cb.ax.tick_params(labelsize=9)
        #aa[6].set_position([0.399, 0.125, 0.228, 0.215])
        #aa[6].set_position([0.33, 0.117, 0.36, 0.16])
        #aa[-2].remove()

        bbox = aa[1].get_position()
        p1 = bbox.p1
        p0 = bbox.p0
        fig.suptitle(f'{data} Cloud Regimes', x=0.5, y=p1[1]+(1/fig_height * 0.5)+0.007, fontsize=18)

        bbox = aa[-2].get_position()
        p1 = bbox.p1
        p0 = bbox.p0
        fig.supxlabel('Optical Depth', y=p0[1]-(1/fig_height * 0.5)-0.007 )


        save_path = adf.plot_location[0] + f'/{data}_CR_centers'
        plt.savefig(save_path)

        if adf.create_html:
            adf.add_website_data(save_path + ".png", var, adf.get_baseline_info("cam_case_name"))
        
        # Closing the figure
        plt.close()


    # Plot the CR centers of obs and test case 
    def plot_hists_obs(cl, cluster_labels, cluster_labels_o, ht_var_name, tau_var_name, adf):
        #defining number of clusters
        k = len(cl)
        # setting up plots
        ylabels = ds[ht_var_name].values
        xlabels = ds[tau_var_name].values
        X2,Y2 = np.meshgrid(np.arange(len(xlabels)+1), np.arange(len(ylabels)+1))
        p = [0,0.2,1,2,3,4,6,8,10,15,99]
        cmap = mpl.colors.ListedColormap(['white', (0.19215686274509805, 0.25098039215686274, 0.5607843137254902), (0.23529411764705882, 0.3333333333333333, 0.6313725490196078), (0.32941176470588235, 0.5098039215686274, 0.6980392156862745), (0.39215686274509803, 0.6, 0.43137254901960786), (0.44313725490196076, 0.6588235294117647, 0.21568627450980393), (0.4980392156862745, 0.6784313725490196, 0.1843137254901961), (0.5725490196078431, 0.7137254901960784, 0.16862745098039217), (0.7529411764705882, 0.8117647058823529, 0.2), (0.9568627450980393, 0.8980392156862745,0.1607843137254902)])
        norm = mpl.colors.BoundaryNorm(p,cmap.N)
        plt.rcParams.update({'font.size': 14})
        fig_height = (1 + 10/3 * ceil(k/3))*3
        fig, ax = plt.subplots(figsize = (12, fig_height), ncols=2, nrows=k, sharex='all', sharey = True)

        aa = ax.ravel()
        boundaries = p
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        if data != 'MISR': aa[1].invert_yaxis()

        # creating weights area for area weighted RFOs
        weights = cluster_labels.stack(z=('time','lat','lon')).lat.values
        weights = np.cos(np.deg2rad(weights))
        weights = weights[valid_indicies]
        indicies = np.arange(len(mat))

        # ax[0,0].set_xticklabels(xlabels)
        # ax[0,0].set_yticklabels(ylabels)

        for i in range(k):
            # Area Weighted relative Frequency of occurence calculation
            total_rfo_num = cluster_labels_o == i 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels_o.lat)))
            total_rfo_denom = cluster_labels_o >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels_o.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100
            total_rfo = total_rfo.values

            im = ax[i,0].pcolormesh(X2,Y2,cl[i].reshape(len(xlabels),len(ylabels)).T,norm=norm,cmap=cmap)
            ax[i,0].set_title(f" Observation CR {i+1}, RFO = {np.round(total_rfo,1)}%")

        # Plotting each cluster center (test_case)
        for i in range (k):

            # Area Weighted relative Frequency of occurence calculation
            total_rfo_num = cluster_labels == i 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100
            total_rfo = total_rfo.values

            # Area weighting each histogram belonging to a cluster and taking the mean
            # if clustering was preformed with wasserstein distance and area weighting on, mean of i = cl[i], however if clustering was preformed with
            # conventional kmeans or wasseerstein without weighting, these two will not be equal
            indicies_i = indicies[np.where(cluster_labels_temp == i)]
            mean = mat[indicies_i] * weights[indicies_i][:,np.newaxis]
            if len(indicies_i) > 0: mean = np.sum(mean, axis=0) / np.sum(weights[indicies_i])
            else: mean = np.zeros(len(xlabels)*len(ylabels))
            
            mean = mean.reshape(len(xlabels),len(ylabels)).T           # reshaping into original histogram shape
            if np.max(mean) <= 1:                                      # Converting fractional data to percent to plot properly
                mean *= 100

            im = ax[i,1].pcolormesh(X2,Y2,mean,norm=norm,cmap=cmap)
            ax[i,1].set_title(f"Test Case CR {i+1}, RFO = {np.round(total_rfo,1)}%")

        if data == "MODIS":
            ylabels = [0, 180, 310, 440, 560, 680, 800, 1000]
            xlabels = [0, 0.3, 1.3, 3.6, 9.4, 23, 60, 150]
            ax[0,0].set_yticks(np.arange(8))
            ax[0,0].set_xticks(np.arange(8))
            ax[0,0].set_yticklabels(ylabels)
            ax[0,0].set_xticklabels(xlabels)
            xticks = ax[0,0].xaxis.get_major_ticks()
            xticks[0].set_visible(False)
            xticks[-1].set_visible(False)

        if data == "MISR":
            xlabels = [0.2, 0.8, 2.4, 6.5, 16.2, 41.5, 100]
            ylabels = [ 0.25,0.75,1.25,1.75,2.25,2.75,3.5, 4.5, 6,8, 10, 12, 14, 16, 20  ]
            ax[0,0].set_yticks(np.arange(0,16,2)+0.5)
            ax[0,0].set_yticklabels(ylabels[0::2])
            ax[0,0].set_xticks(np.array([1,2,3,4,5,6,7]) -0.5)
            ax[0,0].set_xticklabels(xlabels, fontsize = 16)
            xticks = ax[0,0].xaxis.get_major_ticks()
            xticks[0].set_visible(False)
            xticks[-1].set_visible(False)

        if data == 'ISCCP':
            xlabels = [  0, 1.3, 3.6, 9.4, 22.6, 60.4, 450 ]
            ylabels = [  10, 180, 310, 440, 560, 680, 800, 1025]
            yticks = aa[i].get_yticks().tolist()
            xticks = aa[i].get_xticks().tolist()
            aa[i].set_yticks(yticks)
            aa[i].set_xticks(xticks)
            aa[i].set_yticklabels(ylabels)
            aa[i].set_xticklabels(xlabels)
            xticks = aa[i].xaxis.get_major_ticks()
            xticks[0].label1.set_visible(False)
            xticks[-1].label1.set_visible(False)
        
        # setting titles, labels, etc
        if data == "MISR": height_or_pressure = 'h'
        else: height_or_pressure = 'p'
        if height_or_pressure == 'p': fig.supylabel(f'Cloud-top Pressure ({ds[ht_var_name].units})', x = 0.05 )
        if height_or_pressure == 'h': fig.supylabel(f'Cloud-top Height ({ds[ht_var_name].units})', x = 0.05)
        # fig.supxlabel('Optical Depth', fontsize = 12, y=0.26 )
        cbar_ax = fig.add_axes([0.95, 0.38, 0.045, 0.45])
        cb = fig.colorbar(im, cax=cbar_ax, ticks=p)
        cb.set_label(label='Cloud Cover (%)')
        # cb.ax.tick_params(labelsize=9)


        bbox = aa[1].get_position()
        p1 = bbox.p1
        p0 = bbox.p0
        fig.suptitle(f'{data} Cloud Regimes', x=0.5, y=p1[1]+(1/fig_height * 0.5)+0.007, fontsize=18)

        bbox = aa[-2].get_position()
        p1 = bbox.p1
        p0 = bbox.p0
        fig.supxlabel('Optical Depth', y=p0[1]-(1/fig_height * 0.5)-0.007 )

        save_path = adf.plot_location[0] + f'/{data}_CR_centers'
        plt.savefig(save_path)

        if adf.create_html:
            adf.add_website_data(save_path + ".png", var, case_name = None, multi_case=True)

        # Closing the figure
        plt.close()

    # Plot LatLon plots of the frequency of occrence of the baseline/obs and test case 
    def plot_rfo_obs_base_diff(cluster_labels, cluster_labels_d, adf):

        COLOR = 'black'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR
        plt.rcParams.update({'font.size': 13})
        plt.rcParams['figure.dpi'] = 500
        fig_height = 7

        # Comparing obs or baseline?
        if adf.compare_obs == True:
            obs_or_base = 'Observation'
        else:
            obs_or_base = 'Baseline'

        for cluster in range(k):
            fig, ax = plt.subplots(ncols=2, nrows=2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (12,fig_height))#, sharex='col', sharey='row')
            plt.subplots_adjust(wspace=0.08, hspace=0.002)
            aa = ax.ravel()
            
            # Calculating and plotting rfo of baseline/obs
            X, Y = np. meshgrid(cluster_labels_d.lon,cluster_labels_d.lat)
            rfo_d = np.sum(cluster_labels_d==cluster, axis=0) / np.sum(cluster_labels_d >= 0, axis=0) * 100
            aa[0].set_extent([-180, 180, -90, 90])
            aa[0].coastlines()
            mesh = aa[0].pcolormesh(X, Y, rfo_d, transform=ccrs.PlateCarree(), rasterized = True, cmap="GnBu",vmin=0,vmax=100)
            total_rfo_num = cluster_labels_d == cluster 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels_d.lat)))
            total_rfo_denom = cluster_labels_d >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels_d.lat)))
            total_rfo_d = total_rfo_num  / total_rfo_denom * 100
            aa[0].set_title(f"{obs_or_base}, RFO = {round(float(total_rfo_d),1)}", pad=4)

            # Calculating and plotting rfo of test_case
            X, Y = np. meshgrid(cluster_labels.lon,cluster_labels.lat)
            rfo = np.sum(cluster_labels==cluster, axis=0) / np.sum(cluster_labels >= 0, axis=0) * 100
            aa[1].set_extent([-180, 180, -90, 90])
            aa[1].coastlines()
            mesh = aa[1].pcolormesh(X, Y, rfo, transform=ccrs.PlateCarree(), rasterized = True, cmap="GnBu",vmin=0,vmax=100)
            total_rfo_num = cluster_labels == cluster 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100
            aa[1].set_title(f"Test Case, RFO = {round(float(total_rfo),1)}", pad=4)

            # Making colorbar
            cax = fig.add_axes([aa[1].get_position().x1+0.01,aa[1].get_position().y0,0.02,aa[1].get_position().height])
            cb = plt.colorbar(mesh, cax=cax) 
            cb.set_label(label = 'RFO (%)')

            # Calculating and plotting difference
            # If observation/baseline is a higher resolution interpolate from obs/baseline to CAM grid
            if len(cluster_labels_d.lat) * len(cluster_labels_d.lon) > len(cluster_labels.lat) * len(cluster_labels.lon):
                rfo_d = rfo_d.interp_like(rfo, method="nearest")
            
            # If CAM is a higher resolution interpolate from CAM to obs/baseline grid
            if len(cluster_labels_d.lat) * len(cluster_labels_d.lon) <= len(cluster_labels.lat) * len(cluster_labels.lon):
                rfo = rfo.interp_like(rfo_d, method="nearest")
                X, Y = np. meshgrid(cluster_labels_d.lon,cluster_labels_d.lat)

            rfo_diff = rfo - rfo_d

            aa[2].set_extent([-180, 180, -90, 90])
            aa[2].coastlines()
            mesh = aa[2].pcolormesh(X, Y, rfo_diff, transform=ccrs.PlateCarree(), rasterized = True, cmap="coolwarm",vmin=-100,vmax=100)
            total_rfo_num = cluster_labels == cluster 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100
            aa[2].set_title(f"Test - {obs_or_base}, ΔRFO = {round(float(total_rfo-total_rfo_d),1)}", pad=4)


            # Setting yticks
            aa[0].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())        
            aa[2].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())        
            lat_formatter = LatitudeFormatter()
            aa[0].yaxis.set_major_formatter(lat_formatter)
            aa[2].yaxis.set_major_formatter(lat_formatter)


            # making colorbar for diff plot
            cax = fig.add_axes([aa[2].get_position().x1+0.01,aa[2].get_position().y0,0.02,aa[2].get_position().height])
            cb = plt.colorbar(mesh, cax=cax) 
            cb.set_label(label = 'ΔRFO (%)')

            # plotting x labels 
            aa[1].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            aa[1].xaxis.set_major_formatter(lon_formatter)
            aa[2].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            aa[2].xaxis.set_major_formatter(lon_formatter)

            bbox = aa[1].get_position()
            p1 = bbox.p1
            plt.suptitle(f"CR{cluster+1} Relative Frequency of Occurence", y= p1[1]+(1/fig_height * 0.5))#, {round(cl[cluster,23],4)}")

            aa[-1].remove()

            save_path = adf.plot_location[0] + f'/{data}_CR{cluster+1}_LatLon_mean'
            plt.savefig(save_path)

            if adf.create_html:
                adf.add_website_data(save_path + ".png", var, case_name = None, multi_case=True)
        
            # Closing the figure
            plt.close()

    # This function is no longer used, no reason to check it
    # Plot RFO maps of the CRss
    def plot_rfo(cluster_labels, adf):
        #defining number of clusters

        COLOR = 'black'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR
        plt.rcParams.update({'font.size': 10})
        fig_height = 2.2 * ceil(k/2)
        plt.rcParams['figure.dpi'] = 500
        fig, ax = plt.subplots(ncols=2, nrows=int(k/2 + k%2), subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (10,fig_height))#, sharex='col', sharey='row')
        plt.subplots_adjust(wspace=0.13, hspace=0.05)
        aa = ax.ravel()

        X, Y = np. meshgrid(ds.lon,ds.lat)

        # Plotting the rfo of each cluster
        tot_rfo_sum = 0 
        
        for cluster in range(k): #range(0,k+1):
            # Calculating rfo
            rfo = np.sum(cluster_labels==cluster, axis=0) / np.sum(cluster_labels >= 0, axis=0) * 100
            # tca_explained = np.sum(cluster_labels == cluster) * np.sum(init_clusters[cluster]) / total_cloud_amnt * 100
            # tca_explained = round(float(tca_explained.values),1)
            aa[cluster].set_extent([-180, 180, -90, 90])
            aa[cluster].coastlines()
            mesh = aa[cluster].pcolormesh(X, Y, rfo, transform=ccrs.PlateCarree(), rasterized = True, cmap="GnBu",vmin=0,vmax=100)
            #total_rfo = np.sum(cluster_labels==cluster) / np.sum(cluster_labels >= 0) * 100
            # total_rfo_num = np.sum(cluster_labels == cluster * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_num = cluster_labels == cluster 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))

            total_rfo = total_rfo_num  / total_rfo_denom * 100
            tot_rfo_sum += total_rfo
            aa[cluster].set_title(f"CR {cluster+1}, RFO = {round(float(total_rfo),1)}", pad=4)
            # aa[cluster].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            # x_label_plot_list = [4,5,6]
            # y_label_plot_list = [0,2,4,6]
            # if cluster in x_label_plot_list:


            if cluster % 2 == 0:
                aa[cluster].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())        
                lat_formatter = LatitudeFormatter()
                aa[cluster].yaxis.set_major_formatter(lat_formatter)

        #aa[7].set_title(f"Weathersdfasdfa State {i+1}, RFO = {round(float(total_rfo),1)}", pad=-40)
        cb = plt.colorbar(mesh, ax = ax, anchor =(-0.28,0.83), shrink = 0.6)
        cb.set_label(label = 'RFO (%)', labelpad=-3)

        x_ticks_indicies = np.array([-1,-2])

        if k%2 == 1:
            aa[-1].remove()
            x_ticks_indicies -= 1

            #aa[-2].set_position([0.27, 0.11, 0.31, 0.15])

        # plotting x labels on final two plots
        aa[x_ticks_indicies[0]].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        aa[x_ticks_indicies[0]].xaxis.set_major_formatter(lon_formatter)
        aa[x_ticks_indicies[1]].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        aa[x_ticks_indicies[1]].xaxis.set_major_formatter(lon_formatter)

        bbox = aa[1].get_position()
        p1 = bbox.p1
        plt.suptitle(f"CR Relative Frequency of Occurence", x= 0.43, y= p1[1]+(1/fig_height * 0.5))#, {round(cl[cluster,23],4)}")

        # Saving
        save_path = adf.plot_location[0] + f'/{data}_RFO'
        plt.savefig(save_path)

        if adf.create_html:
            adf.add_website_data(save_path + ".png", var, adf.get_baseline_info("cam_case_name"))

    # This function is no longer used, no reason to check it
    # Plot RFO maps of the CRs
    def plot_rfo_diff(cluster_labels, cluster_labels_o, adf):

        # Setting plot parameters
        COLOR = 'black'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR
        plt.rcParams.update({'font.size': 10})
        fig_height = 2.2 * ceil(k/2)
        fig, ax = plt.subplots(ncols=2, nrows=int(k/2 + k%2), subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (10,fig_height))#, sharex='col', sharey='row')
        plt.subplots_adjust(wspace=0.13, hspace=0.05)
        aa = ax.ravel()
        plt.rcParams['figure.dpi'] = 500

        # CReating lat-lon mesh
        X, Y = np. meshgrid(ds.lon,ds.lat)

        # Plotting the difference in relative frequency of occurence (rfo) of each cluster
        for cluster in range(k): 

            # Calculating rfo
            rfo = np.sum(cluster_labels==cluster, axis=0) / np.sum(cluster_labels >= 0, axis=0) * 100
            rfo_o = np.sum(cluster_labels_o==cluster, axis=0) / np.sum(cluster_labels_o >= 0, axis=0) * 100
            
            # If observation/baseline is a higher resolution interpolate from obs/baseline to CAM grid
            if len(cluster_labels_o.lat) * len(cluster_labels_o.lon) > len(cluster_labels.lat) * len(cluster_labels.lon):
                rfo_o = rfo_o.interp_like(rfo, method="nearest")
            
            # If CAM is a higher resolution interpolate from CAM to obs/baseline grid
            if len(cluster_labels_o.lat) * len(cluster_labels_o.lon) <= len(cluster_labels.lat) * len(cluster_labels.lon):
                rfo = rfo.interp_like(rfo_o, method="nearest")

            # difference in RFO 
            rfo_diff = rfo - rfo_o 

            # Setting up subplots and plotting
            aa[cluster].set_extent([-180, 180, -90, 90])
            aa[cluster].coastlines()
            mesh = aa[cluster].pcolormesh(X, Y, rfo_diff, transform=ccrs.PlateCarree(), rasterized = True, cmap="coolwarm",vmin=-100,vmax=100)
    

            # Calucating area weighted rfo difference for the title of subplots
            total_rfo_num = cluster_labels == cluster 
            total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
            total_rfo = total_rfo_num  / total_rfo_denom * 100

            total_rfo_num_o = cluster_labels_o == cluster 
            total_rfo_num_o = np.sum(total_rfo_num_o * np.cos(np.deg2rad(cluster_labels_o.lat)))
            total_rfo_denom_o = cluster_labels_o >= 0
            total_rfo_denom_o = np.sum(total_rfo_denom_o * np.cos(np.deg2rad(cluster_labels_o.lat)))

            total_rfo_o = total_rfo_num_o  / total_rfo_denom_o * 100

            # Setting title
            aa[cluster].set_title(f"CR {cluster+1}, RFO Diff = {round(float(total_rfo-total_rfo_o),1)}", pad=4)

            # Put latitude labels on even numbered subplots
            if cluster % 2 == 0:
                aa[cluster].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())        
                lat_formatter = LatitudeFormatter()
                aa[cluster].yaxis.set_major_formatter(lat_formatter)

        # Setting colorbar
        cb = plt.colorbar(mesh, ax = ax, anchor =(-0.28,0.83), shrink = 0.6)
        cb.set_label(label = 'Diff in RFO (%)', labelpad=-3)

        # Removing extra subplot if k is an odd number
        x_ticks_indicies = np.array([-1,-2])
        if k%2 == 1:
            aa[-1].remove()
            x_ticks_indicies -= 1

        # plotting x labels on final two plots
        aa[x_ticks_indicies[0]].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        aa[x_ticks_indicies[0]].xaxis.set_major_formatter(lon_formatter)
        aa[x_ticks_indicies[1]].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        aa[x_ticks_indicies[1]].xaxis.set_major_formatter(lon_formatter)

        # Setting suptitle
        bbox = aa[1].get_position()
        p1 = bbox.p1
        plt.suptitle(f"CR Relative Frequency of Occurence", x= 0.43, y= p1[1]+(1/fig_height * 0.5))#, {round(cl[cluster,23],4)}")

        # Saving
        save_path = adf.plot_location[0] + f'/{data}_RFO'
        plt.savefig(save_path)

        if adf.create_html:
            adf.add_website_data(save_path + ".png", var, multi_case=True)

    # Create a one hot matrix where lat lon coordinates are over land using cartopy
    def create_land_mask(ds):
        
        # Get land data and prep polygons
        land_110m = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
        land_polygons = list(land_110m.geometries())
        land_polygons = [prep(land_polygon) for land_polygon in land_polygons]

        # Make lat-lon grid
        lats = ds.lat.values
        lons = ds.lon.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        points = [Point(point) for point in zip(lon_grid.ravel(), lat_grid.ravel())]

        # Creating list of cordinates that are over land
        land = []
        for land_polygon in land_polygons:
            land.extend([tuple(point.coords)[0] for point in filter(land_polygon.covers, points)])


        landar = np.asarray(land)
        lat_lon = np.empty((len(lats)*len(lons),2))
        oh_land = np.zeros((len(lats)*len(lons)))
        lat_lon[:,0] = lon_grid.flatten()
        lat_lon[:,1] = lat_grid.flatten()

        # Function to (somewhat) quickly test if a lat-lon point is over land
        @njit()
        def test (oh_land, lat_lon, landar):
            for i in range(len(oh_land)):
                check = lat_lon[i] == landar
                if np.max(np.sum(check,axis=1)) == 2:
                    oh_land[i] = 1
            return oh_land
        
        # Turn oh_land into a one hot matrix
        oh_land = test (oh_land, lat_lon, landar)

        # Reshape into original shape(n_lat, n_lon)
        oh_land=oh_land.reshape((len(lats),len(lons)))

        return oh_land
    
    # Checking if kwargs have been entered correctly
    if wasserstein_or_euclidean not in ['euclidean', 'wasserstein']: 
        print('  WARNING: Invalid option for wasserstein_or_euclidean. Please enter "wasserstein" or "euclidean". Proceeding with default of euclidean distance') 
        wasserstein_or_euclidean = 'euclidean'
    if data_product not in ['ISCCP', "MODIS", 'MISR', 'all']: 
        print('  WARNING: Invalid option for data_product. Please enter "ISCCP" or "MODIS", "MISR" or "all". Proceeding with default of "all"') 
        data_product = 'all'
    if premade_cloud_regimes != None:
        if type(premade_cloud_regimes) != str:
            print('  WARNING: Invalid option for premade_cloud_regimes. Please enter a path to a numpy array of Cloud Regime centers of shape (n_clusters, n_dimensions_of_data). Proceeding with default clusters') 
            premade_cloud_regimes = None
    if lat_range != None:
        if type(lat_range) != list or len(lat_range) != 2:
            print('  WARNING: Invalid option for lat_range. Please enter two values in square brackets sperated by a comma. Example: [-30,30]. Proceeding with entire latitude range') 
            lat_range = None
    if lon_range != None:
        if type(lon_range) != list or len(lon_range) != 2:
            print('  WARNING: Invalid option for lon_range. Please enter two values in square brackets sperated by a comma. Example: [0,90]. Proceeding with entire longitude range') 
            lon_range = None
    if only_ocean_or_land not in [False, 'L', 'O']:
        print('  WARNING: Invalid option for only_ocean_or_land. Please enter "L" for land only, "O" for ocean only. Set to False or leave blank for both land and water. Proceeding with default of False') 
        only_ocean_or_land = False
    
    # Checking if the path we wish to save our plots to exists, and if it doesnt creating it
    if not os.path.isdir(adf.plot_location[0]):
        os.makedirs(adf.plot_location[0])
    
    # path to h0 files
    h0_data_path = adf.get_cam_info("cam_hist_loc", required=True)[0] + "/*h0*.nc"

    # Time Range min and max, or None for all time
    time_range = [str(adf.get_cam_info("start_year")[0]), str(adf.get_cam_info("end_year")[0])] 

    files = glob.glob(h0_data_path)
    # Opening an initial dataset
    init_ds = xr.open_mfdataset(files[0])

    # defining dicts for variable names for each data set
    data_var_dict = {'ISCCP':'FISCCP1_COSP', "MISR":'CLD_MISR', "MODIS":"CLMODIS" }
    ht_var_dict = {'ISCCP':'cosp_prs', "MISR":'cosp_htmisr', "MODIS":"cosp_prs" }
    tau_var_dict = {'ISCCP':'cosp_tau', "MISR":'cosp_tau', "MODIS":"cosp_tau_modis" }

    # geting names of cosp data variables for all data products that will get processed
    if data_product == 'all':
        var_name = list(data_var_dict.values())
    else:
        var_name = [data_var_dict[data_product]]

    # looping through to do analysis on each data product selected
    for var in var_name:

        # Getting data name corresponding to the variable being opened
        key_list = list(data_var_dict.keys())
        val_list = list(data_var_dict.values())
        position = val_list.index(var)
        data = key_list[position]
        ht_var_name = ht_var_dict[data]
        tau_var_name = tau_var_dict[data]

        print(f'\n  Beginning {data} Cloud Regime analysis') #testing

        # variable that gets set to true if var is missing in the data file, and is used to skip that dataset
        missing_var = False

        # Trying to open time series files from cam)ts_loc
        try: ds = xr.open_mfdataset(adf.get_cam_info("cam_ts_loc", required=True)[0] + f"/*{var}*")

        # If that doesnt work trying to open the variables from the h0 files
        except: 
            print(f"      -WARNING: {data} time series file does not exist, was {var} added to the diag_var_list?")
            print("      -Attempting to use h0 files from cam_hist_loc, but this will be slower" )
            # Creating a list of all the variables in the dataset
            remove = list(init_ds.keys())
            try: 
                # Deleting the variables we want to keep in our dataset, all remaining variables will be dropped upon opening the files, this allows for faster opening of large files
                remove.remove(var)
                # If there's a LANDFRAC variable keep it in the dataset
                landfrac_present = True
                try: remove.remove('LANDFRAC')
                except: landfrac_present = False

                ds = xr.open_mfdataset(files, drop_variables = remove)

            # If variables are not present in h0 tell the user the variables do not exist, and that there is not COSP output for this data
            except: 
                print(f'      -{var} does not exist in h0 files, does this run have {data} COSP output? Skipping {data} for now')
                missing_var = True # used to skip the code below and move onto the next var name

        # executing further analysis on this data 
        finally:

            # Skipping var if its not present in data files
            if missing_var:
                continue

            # Adjusting lon to run from -180 to 180 if it doesnt already
            if np.max(ds.lon) > 180: 
                ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
                ds = ds.sortby(ds.lon)

          
            # Selecting only points over ocean or points over land if only_ocean_or_land has been used
            if only_ocean_or_land != False:
                # If LANDFRAC variable is present, use it to mask
                if landfrac_present == True:
                    if only_ocean_or_land == 'L': ds = ds.where(ds.LANDFRAC == 1)
                    elif only_ocean_or_land == 'O': ds = ds.where(ds.LANDFRAC == 0)
                # Otherwise use cartopy
                else:
                    land = create_land_mask(ds)
                    dims = ds.dims

                    # Inserting new axis to make land a broadcastable shape with ds
                    for n in range(len(dims)):
                        if dims[n] != 'lat' and dims[n] != 'lon':
                            land = np.expand_dims(land, n)

                    # Masking out the land or water
                    if only_ocean_or_land == 'L': ds = ds.where(land == 1)
                    elif only_ocean_or_land == 'O': ds = ds.where(land == 0)
                    else: raise Exception('Invalid option for only_ocean_or_land: Please enter "O" for ocean only, "L" for land only, or set to False for both land and water')
                    
            # Selecting lat range
            if lat_range != None:
                if ds.lat[0] > ds.lat[-1]:
                    ds = ds.sel(lat=slice(lat_range[1],lat_range[0]))
                else:
                    ds = ds.sel(lat=slice(lat_range[0],lat_range[1]))

            # Selecting Lon range
            if lon_range != None:
                if ds.lon[0] > ds.lon[-1]:
                    ds = ds.sel(lon=slice(lon_range[1],lon_range[0]))
                else:
                    ds = ds.sel(lon=slice(lon_range[0],lon_range[1]))

            # Selecting time range
            if time_range != ["None","None"]:
                # Need these if statements to be robust if the adf obj only has start_year or end_year
                if time_range[0] == "None":
                    start = ds.time[0]
                    end = time_range[1]
                elif time_range[1] == "None":
                    start = time_range[0]
                    end = ds.time[-1]
                else:
                    start = time_range[0]
                    end = time_range[1]

                ds = ds.sel(time=slice(start,end))
            
            # Turning dataset into a dataarray
            ds = ds[var]

            # Selecting only valid tau and height/pressure range
            # Many data products have a -1 bin for failed retreivals, we do not wish to include this
            tau_selection = {tau_var_name:slice(0,9999999999999)}
            # Making sure this works for pressure which is ordered largest to smallest and altitude which is ordered smallest to largest
            if ds[ht_var_name][0] > ds[ht_var_name][-1]: ht_selection = {ht_var_name:slice(9999999999999,0)}
            else: ht_selection = {ht_var_name:slice(0,9999999999999)}
            ds = ds.sel(tau_selection)
            ds = ds.sel(ht_selection)
            

            # Opening cluster centers
            # Using premade clusters if they have been provided
            if type(premade_cloud_regimes) == str:
                cl = np.load(premade_cloud_regimes)
                # Checking if the shape is what we'd expect
                if cl.shape[1] != len(ds[tau_var_name]) * len(ds[ht_var_name]):
                    if data == 'ISCCP' and cl.shape[1] == 42:
                        None
                    elif data == 'MISR' and cl.shape[1] == 105:
                        None
                    else:
                        raise Exception (f'premade_cloud_regimes is the wrong shape. premade_cloud_regimes.shape = {cl.shape}, but must be shape (k, {len(ds[tau_var_name]) * len(ds[ht_var_name])}) to fit the loaded {data} data')
                print('      -Using premade cloud regimes:')
            
            # If custom CRs havent been passed, use either the emd or euclidean premade ones
            elif wasserstein_or_euclidean == "wasserstein":
                obs_data_loc = adf.get_basic_info('obs_data_loc') + '/'
                cluster_centers_path = adf.variable_defaults[f"{data}_emd_centers"]['obs_file']
                cl = np.load(obs_data_loc + cluster_centers_path)

            elif wasserstein_or_euclidean == "euclidean":
                obs_data_loc = adf.get_basic_info('obs_data_loc') + '/'
                cluster_centers_path = adf.variable_defaults[f"{data}_euclidean_centers"]['obs_file']
                cl = np.load(obs_data_loc + cluster_centers_path)

            # Defining k, the number of clusters
            k = len(cl)

            print(f'      -Preprocessing data') 

            # COSP ISCCP data has one extra tau bin than the satellite data, and misr has an extra height bin. This checks roughly if we are comparing against the 
            # satellite data, and if so removes the extra tau or ht bin. If a user passes home made CRs from CESM data, no data will be removed
            if data == 'ISCCP' and cl.shape[1] == 42:
                    # a slightly hacky way to drop the smallest tau bin, but is robust incase tau is flipped in a future version
                    ds = ds.sel(cosp_tau=slice(np.min(ds.cosp_tau)+1e-11,np.inf))
                    print("      -Dropping smallest tau bin to be comparable with observational cloud regimes")
            if data == 'MISR' and cl.shape[1] == 105:
                    # a slightly hacky way to drop the lowest height bin, but is robust incase height is flipped in a future version
                    ds = ds.sel(cosp_htmisr=slice(np.min(ds.cosp_htmisr)+1e-11,np.inf))
                    print("      -Dropping lowest height bin to be comparable with observational cloud regimes")

            # Selcting only the relevant data and stacking it to shape n_histograms, n_tau * n_pc
            dims = list(ds.dims)
            dims.remove(tau_var_name)
            dims.remove(ht_var_name)
            histograms = ds.stack(spacetime=(dims), tau_ht=(tau_var_name, ht_var_name))
            weights = np.cos(np.deg2rad(histograms.lat.values)) # weights array to use with emd-kmeans

            # Turning into a numpy array for clustering
            mat = histograms.values

            # Removing all histograms with 1 or more nans in them
            indicies = np.arange(len(mat))
            is_valid = ~np.isnan(mat.mean(axis=1))
            is_valid = is_valid.astype(np.int32)
            valid_indicies = indicies[is_valid==1]
            mat=mat[valid_indicies]
            weights=weights[valid_indicies]

            print(f'      -Fitting data') 

            # Compute cluster labels
            cluster_labels_temp = precomputed_clusters(mat, cl, wasserstein_or_euclidean, ds)

            # taking the flattened cluster_labels_temp array, and turning it into a datarray the shape of ds.var_name, and reinserting NaNs in place of missing data
            cluster_labels = np.full(len(indicies), np.nan, dtype=np.int32)
            cluster_labels[valid_indicies]=cluster_labels_temp
            cluster_labels = xr.DataArray(data=cluster_labels, coords={"spacetime":histograms.spacetime},dims=("spacetime") )
            cluster_labels = cluster_labels.unstack()

            # Comparing to observation 
            if adf.compare_obs == True:
                # defining dicts for variable names for each data set
                obs_data_var_dict = {'ISCCP':'n_pctaudist', "MISR":'clMISR', "MODIS":"MODIS_CLD_HISTO" }
                obs_ht_var_dict = {'ISCCP':'levtau', "MISR":'tau', "MODIS":"COT" }
                obs_tau_var_dict = {'ISCCP':'levpc', "MISR":'cth', "MODIS":"PRES" }

                # Getting data name corresponding to the variable being opened
                key_list = list(obs_data_var_dict.keys())
                val_list = list(obs_data_var_dict.values())
                obs_var = obs_data_var_dict[data]
                position = val_list.index(obs_var)
                data = key_list[position]
                obs_ht_var_name = obs_ht_var_dict[data]
                obs_tau_var_name = obs_tau_var_dict[data]

                print(f'      -Starting {data} observation data') 

                # Opening observation files. The obs files have three variables, precomputed euclidean cluster labels, precomputed emd cluster labels
                # and then the raw data to use if custom CRs are passed in.
                obs_data_path = adf.var_obs_dict[var]['obs_file']

                # Opening the data
                ds_o = xr.open_dataset(obs_data_path)

                # Selecting either the appropriate pre-computed cluster_labels or the raw data
                if premade_cloud_regimes == None:
                    if wasserstein_or_euclidean == 'wasserstein':
                        ds_o = ds_o.emd_cluster_labels
                    else:
                        ds_o = ds_o.euclidean_cluster_labels
                else:
                    ds_o = ds_o[obs_var]

                # Adjusting lon to run from -180 to 180 if it doesnt already
                if np.max(ds_o.lon) > 180: 
                    ds_o.coords['lon'] = (ds_o.coords['lon'] + 180) % 360 - 180
                    ds_o = ds_o.sortby(ds_o.lon)

                # Selecting only points over ocean or points over land if only_ocean_or_land has been used
                if only_ocean_or_land != False:
                    land = create_land_mask(ds_o)
                    dims = ds_o.dims

                    # inserting new axis to make land a broadcastable shape with ds_o
                    for n in range(len(dims)):
                        if dims[n] != 'lat' and dims[n] != 'lon':
                            land = np.expand_dims(land, n)

                    # Masking out the land or water
                    if only_ocean_or_land == 'L': ds_o = ds_o.where(land == 1)
                    elif only_ocean_or_land == 'O': ds_o = ds_o.where(land == 0)
                    else: raise Exception('Invalid option for only_ocean_or_land: Please enter "O" for ocean only, "L" for land only, or set to False for both land and water')
                    
                # Selecting lat range
                if lat_range != None:
                    if ds_o.lat[0] > ds_o.lat[-1]:
                        ds_o = ds_o.sel(lat=slice(lat_range[1],lat_range[0]))
                    else:
                        ds_o = ds_o.sel(lat=slice(lat_range[0],lat_range[1]))

                # Selecting Lon range
                if lon_range != None:
                    if ds_o.lon[0] > ds_o.lon[-1]:
                        ds_o = ds_o.sel(lon=slice(lon_range[1],lon_range[0]))
                    else:
                        ds_o = ds_o.sel(lon=slice(lon_range[0],lon_range[1]))

                # Don't select time range for obsrvation, just compare to the full record
                # # Selecting time range
                # if time_range != ["None","None"]:
                #     if time_range[0] == "None":
                #         start = ds.time[0]
                #         end = time_range[1]
                #     elif time_range[1] == "None":
                #         start = time_range[0]
                #         end = ds.time[-1]
                #     else:
                #         start = time_range[0]
                #         end = time_range[1]

                #     ds = ds.sel(time=slice(start,end))

                if premade_cloud_regimes == None:
                    cluster_labels_o = ds_o
                    cluster_labels_o_temp = cluster_labels_o.stack(spacetime=("time", 'lat', 'lon'))
                else:
                    # Selecting only valid tau and height/pressure range
                    # Many data products have a -1 bin for failed retreivals, we do not wish to include this
                    tau_selection = {obs_tau_var_name:slice(0,9999999999999)}
                    # Making sure this works for pressure which is ordered largest to smallest and altitude which is ordered smallest to largest
                    if ds_o[obs_ht_var_name][0] > ds_o[obs_ht_var_name][-1]: ht_selection = {obs_ht_var_name:slice(9999999999999,0)}
                    else: ht_selection = {obs_ht_var_name:slice(0,9999999999999)}
                    ds_o = ds_o.sel(tau_selection)
                    ds_o = ds_o.sel(ht_selection)

                    # Selcting only the relevant data and stacking it to shape n_histograms, n_tau * n_pc
                    dims = list(ds_o.dims)
                    dims.remove(obs_tau_var_name)
                    dims.remove(obs_ht_var_name)
                    histograms_o = ds_o.stack(spacetime=(dims), tau_ht=(obs_ht_var_name, obs_tau_var_name))
                    weights_o = np.cos(np.deg2rad(histograms_o.lat.values)) # weights_o array to use with emd-kmeans

                    # Turning into a numpy array for clustering
                    mat_o = histograms_o.values

                    # Removing all histograms with 1 or more nans in them
                    indicies = np.arange(len(mat_o))
                    is_valid = ~np.isnan(mat_o.mean(axis=1))
                    is_valid = is_valid.astype(np.int32)
                    valid_indicies_o = indicies[is_valid==1]
                    mat_o=mat_o[valid_indicies_o]
                    weights_o=weights_o[valid_indicies_o]

                    if np.min(mat_o < 0):
                        raise Exception (f'Found negative value in ds_o.{var_name}, if this is a fill value for missing data, convert to nans and try again')

                    print(f'      -Fitting data') 

                    # Compute cluster labels
                    cluster_labels_temp_o = precomputed_clusters(mat_o, cl, wasserstein_or_euclidean, ds_o)

                    # Taking the flattened cluster_labels_temp_o array, and turning it into a datarray the shape of obs_ds.var_name, and reinserting NaNs in place of missing data
                    cluster_labels_o = np.full(len(indicies), np.nan, dtype=np.int32)
                    cluster_labels_o[valid_indicies_o]=cluster_labels_temp_o
                    cluster_labels_o = xr.DataArray(data=cluster_labels_o, coords={"spacetime":histograms_o.spacetime},dims=("spacetime") )
                    cluster_labels_o = cluster_labels_o.unstack()

                print(f'      -Plotting') 

                plot_hists_obs(cl, cluster_labels, cluster_labels_o, ht_var_name, tau_var_name, adf)
                plot_rfo_obs_base_diff(cluster_labels, cluster_labels_o, adf)

            # Comparing to CAM baseline if not comparing to obs
            else:
                # path to h0 files
                baseline_h0_data_path = adf.get_baseline_info("cam_hist_loc", required=True) + "/*h0*.nc"
                # Time Range min and max, or None for all time
                time_range_b = [str(adf.get_baseline_info("start_year")), str(adf.get_baseline_info("end_year"))] 
                # Creating a list of files
                files = glob.glob(baseline_h0_data_path)
                # Opening an initial dataset
                init_ds_b = xr.open_dataset(files[0])

                print(f'      -Starting {data} CAM baseline data') #testing

                # Variable that gets set to true if var is missing in the data file, and is used to skip processing that dataset
                missing_var = False

                # Trying to open time series files from cam)ts_loc
                try: ds_b = xr.open_mfdataset(adf.get_baseline_info("cam_ts_loc", required=True) + f"/*{var}*")

                # If that doesnt work trying to open the variables from the h0 files
                except: 
                    print(f"      -WARNING: {data} time series file does not exist, was {var} added to the diag_var_list?")
                    print("       Attempting to use h0 files from cam_hist_loc, but this will be slower" )
                    # Creating a list of all the variables in the dataset
                    remove = list(init_ds_b.keys())
                    try: 
                        # Deleting the variables we want to keep in our dataset, all remaining variables will be dropped upon opening the files, this allows for faster opening of large files
                        remove.remove(var)
                        # If there's a LANDFRAC variable keep it in the dataset
                        landfrac_present = True
                        try: remove.remove('LANDFRAC')
                        except: landfrac_present = False
                        
                        # Opening dataset and dropping irrelevant data
                        ds_b = xr.open_mfdataset(files, drop_variables = remove)

                    # If variables are not present in h0 tell the user the variables do not exist, and that there is not COSP output for this data
                    except: 
                        print(f'  {var} does not exist in h0 files, does this run have {data} COSP output? Skipping {data} for now')
                        missing_var = True # used to skip the code below and move onto the next var name

                # Executing further analysis on this data 
                finally:

                    # Skipping var if its not present in data files
                    if missing_var:
                        continue

                    # Adjusting lon to run from -180 to 180 if it doesnt already
                    if np.max(ds_b.lon) > 180: 
                        ds_b.coords['lon'] = (ds_b.coords['lon'] + 180) % 360 - 180
                        ds_b = ds_b.sortby(ds_b.lon)
                
                    # Selecting only points over ocean or points over land if only_ocean_or_land has been used
                    if only_ocean_or_land != False:
                        # If LANDFRAC variable is present, use it to mask
                        if landfrac_present == True:
                            if only_ocean_or_land == 'L': ds_b = ds_b.where(ds_b.LANDFRAC == 1)
                            elif only_ocean_or_land == 'O': ds_b = ds_b.where(ds_b.LANDFRAC == 0)
                        # Otherwise use cartopy
                        else:
                            land = create_land_mask(ds_b)
                            dims = ds_b.dims

                            # Inserting new axis to make land a broadcastable shape with ds_b
                            for n in range(len(dims)):
                                if dims[n] != 'lat' and dims[n] != 'lon':
                                    land = np.expand_dims(land, n)

                            # Masking out the land or water
                            if only_ocean_or_land == 'L': ds_b = ds_b.where(land == 1)
                            elif only_ocean_or_land == 'O': ds_b = ds_b.where(land == 0)
                            else: raise Exception('Invalid option for only_ocean_or_land: Please enter "O" for ocean only, "L" for land only, or set to False for both land and water')
                            
                    # Selecting lat range
                    if lat_range != None:
                        if ds_b.lat[0] > ds_b.lat[-1]:
                            ds_b = ds_b.sel(lat=slice(lat_range[1],lat_range[0]))
                        else:
                            ds_b = ds_b.sel(lat=slice(lat_range[0],lat_range[1]))

                    # Selecting Lon range
                    if lon_range != None:
                        if ds_b.lon[0] > ds_b.lon[-1]:
                            ds_b = ds_b.sel(lon=slice(lon_range[1],lon_range[0]))
                        else:
                            ds_b = ds_b.sel(lon=slice(lon_range[0],lon_range[1]))

                    # Selecting time range
                    if time_range_b != ["None","None"]:
                        # Need these if statements to be robust if the adf obj only has start_year or end_year
                        if time_range_b[0] == "None":
                            start = ds_b.time[0]
                            end = time_range_b[1]
                        elif time_range_b[1] == "None":
                            start = time_range_b[0]
                            end = ds_b.time[-1]
                        else:
                            start = time_range_b[0]
                            end = time_range_b[1]

                        ds_b = ds_b.sel(time=slice(start,end))
                    
                    # Turning dataset into a dataarray
                    ds_b = ds_b[var]

                    # Selecting only valid tau and height/pressure range
                    # Many data products have a -1 bin for failed retreivals, we do not wish to include this
                    tau_selection = {tau_var_name:slice(0,9999999999999)}
                    # Making sure this works for pressure which is ordered largest to smallest and altitude which is ordered smallest to largest
                    if ds_b[ht_var_name][0] > ds_b[ht_var_name][-1]: ht_selection = {ht_var_name:slice(9999999999999,0)}
                    else: ht_selection = {ht_var_name:slice(0,9999999999999)}
                    ds_b = ds_b.sel(tau_selection)
                    ds_b = ds_b.sel(ht_selection)

                    print(f'      -Preprocessing data') 

                    # COSP ISCCP data has one extra tau bin than the satellite data, and misr has an extra height bin. This checks roughly if we are comparing against the 
                    # satellite data, and if so removes the extra tau or ht bin. If a user passes home made CRs from CESM data, no data will be removed
                    if data == 'ISCCP' and cl.shape[1] == 42:
                            # A slightly hacky way to drop the smallest tau bin, but is robust incase tau is flipped in a future version
                            ds_b = ds_b.sel(cosp_tau=slice(np.min(ds_b.cosp_tau)+1e-11,np.inf))
                            print("      -Dropping smallest tau bin to be comparable with observational cloud regimes")
                    if data == 'MISR' and cl.shape[1] == 105:
                            # A slightly hacky way to drop the lowest height bin, but is robust incase height is flipped in a future version
                            ds_b = ds_b.sel(cosp_htmisr=slice(np.min(ds_b.cosp_htmisr)+1e-11,np.inf))
                            print("      -Dropping lowest height bin to be comparable with observational cloud regimes")

                    # Selcting only the relevant data and stacking it to shape n_histograms, n_tau * n_pc
                    dims = list(ds_b.dims)
                    dims.remove(tau_var_name)
                    dims.remove(ht_var_name)
                    histograms_b = ds_b.stack(spacetime=(dims), tau_ht=(tau_var_name, ht_var_name))
                    weights_b = np.cos(np.deg2rad(histograms_b.lat.values)) # weights_b array to use with emd-kmeans

                    # Turning into a numpy array for clustering
                    mat_b = histograms_b.values

                    # Removing all histograms with 1 or more nans in them
                    indicies = np.arange(len(mat_b))
                    is_valid = ~np.isnan(mat_b.mean(axis=1))
                    is_valid = is_valid.astype(np.int32)
                    valid_indicies_b = indicies[is_valid==1]
                    mat_b=mat_b[valid_indicies_b]
                    weights_b=weights_b[valid_indicies_b]

                    if np.min(mat_b < 0):
                        raise Exception (f'Found negative value in ds_b.{var_name}, if this is a fill value for missing data, convert to nans and try again')

                    print(f'      -Fitting data') 

                    # Compute cluster labels
                    cluster_labels_temp_b = precomputed_clusters(mat_b, cl, wasserstein_or_euclidean, ds_b)

                    # Taking the flattened cluster_labels_temp_b array, and turning it into a datarray the shape of ds.var_name, and reinserting NaNs in place of missing data
                    cluster_labels_b = np.full(len(indicies), np.nan, dtype=np.int32)
                    cluster_labels_b[valid_indicies_b]=cluster_labels_temp_b
                    cluster_labels_b = xr.DataArray(data=cluster_labels_b, coords={"spacetime":histograms_b.spacetime},dims=("spacetime") )
                    cluster_labels_b = cluster_labels_b.unstack()

                    print(f'      -Plotting') 

                    # Plotting
                    plot_hists_baseline(cl, cluster_labels, cluster_labels_b, ht_var_name, tau_var_name, adf)
                    plot_rfo_obs_base_diff(cluster_labels, cluster_labels_b, adf)

                


# %%
