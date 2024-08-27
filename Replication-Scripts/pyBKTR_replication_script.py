# Code to run BKTR examples #
# Lanthier, Lei, Sun and Labbe 2023 #


#################
#################
# Installation #
#################
#################

### From PyPI
# pip install pyBKTR 

### From Github (Latest Version)
# pip install git+https://github.com/julien-hec/pyBKTR.git

# Please note that multiple sections were ran on GPU
# If you don't have a GPU and want to run a given section,
# use 'cpu' instead of 'cuda' in the TSR$set_params() method
# Also the output of the following code ran in notebooks
# can be found (by section) at:
#    https://github.com/julien-hec/bktr-examples


# Load libraries
from pyBKTR.bktr import BKTRRegressor
from pyBKTR.examples.bixi import BixiData
from pyBKTR.kernels import KernelMatern, KernelParameter, KernelSE
from pyBKTR.tensor_ops import TSR
from pyBKTR.utils import simulate_spatiotemporal_data

# Following two librairies are for results manipulation
import numpy as np
import pandas as pd

###################################
###################################
# Intro: BKTR regressor class #
###################################
###################################

# This is a small example of how to use the BKTRRegressor class
TSR.set_params('float32', 'cpu', 1)

bixi_data = BixiData(is_light=True)

bktr_regressor = BKTRRegressor(
    data_df = bixi_data.data_df,
    spatial_positions_df = bixi_data.spatial_positions_df,
    temporal_positions_df = bixi_data.temporal_positions_df,
    burn_in_iter = 500,
    sampling_iter = 500,
)

bktr_regressor.mcmc_sampling()

print(bktr_regressor.summary)

#####################################
#####################################
# Section 5: Simulation-based study #
#####################################
#####################################

#######################
### 5.2: Imputation ###
#######################
df_res_arr = []
TSR.set_params('float32', 'cuda', 1)
BURN_IN_ITER = 500
SAMPLING_ITER = 500

for len_scale in [3, 6]:
    for miss_perc in [0.1, 0.5, 0.9]:
        for i in range(1, 11):
            spatial_kernel = KernelMatern(
                smoothness_factor = 5,
                lengthscale = KernelParameter(value = len_scale)
            )
            temporal_kernel = KernelSE(
                lengthscale = KernelParameter(value = len_scale)
            )

            simu_data = simulate_spatiotemporal_data(
                nb_locations=100,
                nb_time_points=150,
                nb_spatial_dimensions=2,
                spatial_scale=10,
                time_scale=10,
                spatial_covariates_means=[0, 2, 4],
                temporal_covariates_means=[1, 3],
                spatial_kernel=spatial_kernel,
                temporal_kernel=temporal_kernel,
                noise_variance_scale=1
            )

            data_df = simu_data['data_df'].copy()
            spatial_positions_df = simu_data['spatial_positions_df']
            temporal_positions_df = simu_data['temporal_positions_df']
            index_choices_tsr = TSR.tensor(list(range(len(data_df))))
            nb_miss_index = round(miss_perc * len(data_df))
            na_index = TSR.rand_choice(
                index_choices_tsr, nb_miss_index, use_replace=False
            ).cpu().numpy().astype(int)
            data_df.iloc[na_index, 0] = pd.NA

            bktr_regressor = BKTRRegressor(
                data_df = data_df,
                rank_decomp = 10,
                burn_in_iter = BURN_IN_ITER,
                sampling_iter = SAMPLING_ITER,
                spatial_kernel = KernelMatern(smoothness_factor = 5),
                spatial_positions_df = simu_data['spatial_positions_df'],
                temporal_kernel = KernelSE(),
                temporal_positions_df = simu_data['temporal_positions_df'],
                has_geo_coords = False
            )
            bktr_regressor.mcmc_sampling()

            y_err = (
                bktr_regressor.imputed_y_estimates.iloc[
                    na_index
                ][['y']].to_numpy()
                - simu_data['data_df'].iloc[na_index][['y']].to_numpy()
            )
            beta_err = (
                np.abs(bktr_regressor.beta_estimates.to_numpy()
                - simu_data['beta_df'].to_numpy())
            )
            y_rmse = float(np.sqrt(np.mean(y_err**2)))
            y_mae = float(np.mean(abs(y_err)))
            beta_rmse = float(np.sqrt(np.mean(beta_err**2)))
            beta_mae = float(np.mean(abs(beta_err)))

            df_res_arr.append([
                len_scale,
                miss_perc,
                i,
                beta_mae,
                beta_rmse,
                y_mae,
                y_rmse,
                bktr_regressor.result_logger.total_elapsed_time
            ])

print('## Iterations dataframe ##')
df = pd.DataFrame(df_res_arr, columns=[
    'Lengthscale', 'Missing', 'Iter', 'B_MAE',
    'B_RMSE', 'Y_MAE', 'Y_RMSE', 'Time'
])
print(df)

# Aggregate results (Table 5)
print('## Aggregated dataframe ##')
agg_df = df.groupby(['Lengthscale', 'Missing'])[[
    'B_MAE', 'B_RMSE', 'Y_MAE', 'Y_RMSE', 'Time'
]].agg(['mean', 'std']).reset_index()
print(agg_df)



##########################
### 5.3: Interpolation ###
##########################

TSR.set_params('float32', 'cuda', 1)

BURN_IN_ITER = 500
SAMPLING_ITER = 500

nb_aside_locs = 4
nb_aside_times = 6

df_res_arr = []

for ds_type in ['Smaller', 'Larger']:
    for len_scale in [3, 6]:
        for i in range(1, 11):
            matern_lengthscale = KernelParameter(value = len_scale)
            se_lengthscale = KernelParameter(value = len_scale)
            spatial_kernel = KernelMatern(
                lengthscale = matern_lengthscale, smoothness_factor = 5
            )
            temporal_kernel = KernelSE(lengthscale = se_lengthscale)

            nb_locs = 20 if ds_type == 'Smaller' else 100
            nb_times = 30 if ds_type == 'Smaller' else 150
            spa_cov_means = [0, 2] if ds_type == 'Smaller' else [0, 2, 4]
            tem_cov_means = [1] if ds_type == 'Smaller' else [1, 3]

            simu_data = simulate_spatiotemporal_data(
                nb_locations=nb_locs,
                nb_time_points=nb_times,
                nb_spatial_dimensions=2,
                spatial_scale=10,
                time_scale=10,
                spatial_covariates_means=spa_cov_means,
                temporal_covariates_means=tem_cov_means,
                spatial_kernel=spatial_kernel,
                temporal_kernel=temporal_kernel,
                noise_variance_scale=1
            )

            data_df = simu_data['data_df'].copy()
            spatial_positions_df = simu_data['spatial_positions_df']
            temporal_positions_df = simu_data['temporal_positions_df']

            obs_nb_locs = nb_locs - nb_aside_locs
            obs_nb_times = nb_times - nb_aside_times

            all_locs = data_df.index.get_level_values(0).unique().to_list()
            all_times = data_df.index.get_level_values(1).unique().to_list()

            locs_indx_sample = list(TSR.rand_choice(
                TSR.tensor(range(1, len(all_locs) + 1)),
                obs_nb_locs
            ).cpu().numpy())
            obs_locs = [all_locs[int(i) - 1] for i in locs_indx_sample]
            new_locs = list(set(all_locs) - set(obs_locs))

            times_indx_sample = list(TSR.rand_choice(TSR.tensor(
                range(1, len(all_times) + 1)),
                obs_nb_times
            ).cpu().numpy())
            obs_times = [all_times[int(i) - 1] for i in times_indx_sample]
            new_times = list(set(all_times) - set(obs_times))

            obs_data_df = data_df.drop(index=new_locs, level='location')
            obs_data_df = obs_data_df.drop(index=new_times, level='time')
            obs_spatial_pos_df = spatial_positions_df.drop(index=new_locs,)
            obs_temporal_pos_df = temporal_positions_df.drop(index=new_times,)

            new_data_df = data_df[
                (data_df.index.get_level_values(0).isin(new_locs)) |
                (data_df.index.get_level_values(1).isin(new_times))
            ].copy()
            new_beta_data_df = simu_data['beta_df'][
                simu_data['beta_df'].index.get_level_values(0).isin(new_locs) |
                simu_data['beta_df'].index.get_level_values(1).isin(new_times)
            ].copy()
            new_spatial_pos_df = spatial_positions_df[
                spatial_positions_df.index.isin(new_locs)
            ].copy()
            new_temporal_pos_df = temporal_positions_df[
                temporal_positions_df.index.isin(new_times)
            ].copy()


            bktr_regressor = BKTRRegressor(
                data_df = obs_data_df,
                rank_decomp = 10,
                burn_in_iter = BURN_IN_ITER,
                sampling_iter = SAMPLING_ITER,
                spatial_kernel = KernelMatern(smoothness_factor = 5),
                spatial_positions_df = obs_spatial_pos_df,
                temporal_kernel = KernelSE(),
                temporal_positions_df = obs_temporal_pos_df,
                has_geo_coords = False
            )
            bktr_regressor.mcmc_sampling()

            preds_y_df, preds_beta_df = bktr_regressor.predict(
                new_data_df,
                new_spatial_pos_df,
                new_temporal_pos_df
            )

            preds_y_df.sort_index(inplace=True)
            new_data_df.sort_index(inplace=True)
            preds_beta_df.sort_index(inplace=True)
            new_beta_data_df.sort_index(inplace=True)
            preds_y_err = (
                new_data_df['y'].to_numpy() - preds_y_df['y'].to_numpy()
            )
            preds_beta_err = (
                new_beta_data_df.to_numpy() - preds_beta_df.to_numpy()
            )
            df_res_arr.append([
                ds_type,
                len_scale,
                i,
                np.mean(np.abs(preds_beta_err)),
                np.sqrt(np.mean(np.square(preds_beta_err))),
                np.mean(np.abs(preds_y_err)),
                np.sqrt(np.mean(np.square(preds_y_err))),
            ])

print('## Iterations dataframe ##')
df = pd.DataFrame(df_res_arr, columns=[
    'Dataset_Type', 'Lengthscale', 'Iter',
    'B_MAE', 'B_RMSE', 'Y_MAE', 'Y_RMSE'
])
print(df)


# Aggregate results (Table 6)
print('## Aggregated dataframe ##')
agg_df = df.groupby(['Dataset_Type', 'Lengthscale'])[[
    'B_MAE', 'B_RMSE', 'Y_MAE', 'Y_RMSE'
]].agg(['mean', 'std']).reset_index()
print(agg_df)



##################################
##################################
### App E: Mercator Projection ###
##################################
##################################

import plotly.express as px

bixi_data = BixiData()

bktr_regressor = BKTRRegressor(
    data_df = bixi_data.data_df,
    spatial_positions_df = bixi_data.spatial_positions_df,
    temporal_positions_df = bixi_data.temporal_positions_df
)

print('# Initial dataframe (longitude, latitude) #')
print(bktr_regressor.geo_coords_projector.ini_df.head())
print()
print('# Mercator Projection dataframe ([-5, 5] scaled) #')
print(bktr_regressor.geo_coords_projector.scaled_ini_df.head())


FIG_WIDTH = 550
# Scaled Scatter Plot Figure 10 (right)
fig_scale = px.scatter(
    bktr_regressor.geo_coords_projector.scaled_ini_df.reset_index(),
    x='lon_x', y='lat_y', hover_name='location',
    width = FIG_WIDTH
)
fig_scale.update_xaxes(range=[-5.5, 5.5])
fig_scale.update_yaxes(range=[-5.5, 5.5])
fig_scale.show()

# Map Scatter Plot (longitude, latitude) Figure 10 (left)
fig_map = px.scatter_mapbox(
    bktr_regressor.geo_coords_projector.ini_df,
    lat='latitude', lon='longitude', zoom=9.9,
    mapbox_style='carto-positron',
    width = FIG_WIDTH
)
fig_map.show()
