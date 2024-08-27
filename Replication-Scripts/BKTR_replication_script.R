# Code to run BKTR examples #
# Lanthier, Lei, Sun and Labbe 2023 #


#################
#################
# Installation #
#################
#################

### From CRAN
# install.packages("panelView") 
# install.packages("ggpubr") # To place kernel plots side by side (in Figure 3)
# install.packages("BKTR")

### From Github (Latest Version)
# install.packages("devtools") # if not installed
# devtools::install_github("julien-hec/BKTR", ref = "main")

# Please note that multiple sections were ran on GPU
# If you don't have a GPU and want to run a given section,
# use 'cpu' instead of 'cuda' in the TSR$set_params() method
# Also the output of the following code ran in notebooks
# can be found (by section) at:
#    https://github.com/julien-hec/bktr-examples

library('BKTR')
library(data.table)
library(ggplot2)
library('ggpubr')

# To mitigate the effect of the brown out of staten maps in ggmap
#    https://github.com/dkahle/ggmap/issues/350
# Users need to use their own STADIA API key or Google Maps API key
# See https://cran.r-project.org/web/packages/ggmap/readme/README.html
# for more information
# stadia_api_token <- '_YOUR_STADIA_API_TOKEN_'
stadia_api_token <- NULL

###################################
###################################
# Section 3: BKTR regressor class #
###################################
###################################

# Input data presentation
bixi_data <- BixiData$new()
ex_locs <- c('7114 - Smith / Peel', '6435 - Victoria Hall')
ex_times <- c('2019-04-17', '2019-04-18', '2019-04-19')
# Looking at a valid spatial_positions_df
print(bixi_data$spatial_positions_df[location %in% ex_locs])
# Looking at a valid temporal_positions_df
print(bixi_data$temporal_positions_df[time %in% ex_times])
# Looking at a valid data_df
print(bixi_data$data_df[
  location %in% ex_locs & time %in% ex_times,
  c(1:4, 17)
])

# Set seed for results consistency
TSR$set_params(seed = 1)
bixi_light <- BixiData$new(is_light = TRUE)
bktr_regressor <- BKTRRegressor$new(
  data_df = bixi_light$data_df,
  spatial_positions_df = bixi_light$spatial_positions_df,
  temporal_positions_df = bixi_light$temporal_positions_df)

# Launch MCMC sampling
bktr_regressor$mcmc_sampling()


###########################
###########################
# Section 4: BKTR Kernels #
###########################
###########################

MIN_DAY <- 1
MAX_DAY <- 21
pos_df <- data.frame(day=MIN_DAY:MAX_DAY, pos = as.numeric(MIN_DAY:MAX_DAY))

# Create three kernels: Periodic, SE and Local Periodic
se_lengthscale <- KernelParameter$new(value = 10)
per_length <- KernelParameter$new(value = 7, is_fixed = TRUE)
k_periodic <- KernelPeriodic$new(period_length = per_length)
k_se <- KernelSE$new(lengthscale = se_lengthscale)
k_local_periodic <- k_periodic * k_se
k_se$set_positions(pos_df)
k_periodic$set_positions(pos_df)
k_local_periodic$set_positions(pos_df)
# Set their covariance matrices
cov_per <- k_periodic$kernel_gen()
cov_se <- k_se$kernel_gen()
cov_lper <- k_local_periodic$kernel_gen()

# Create plot scales
sxd <- scale_x_discrete(
  limits = factor(MIN_DAY:MAX_DAY),
  breaks = as.character(seq(MIN_DAY, MAX_DAY, by = 2))
)
syd <- scale_y_discrete(
  limits = factor(MAX_DAY:MIN_DAY),
  breaks = as.character(seq(MAX_DAY, MIN_DAY, by = -2))
)
sfv <- scale_fill_viridis_c()

# Plot each kernels
p_se <- k_se$plot(show_figure = FALSE)
p_per <- k_periodic$plot(show_figure = FALSE)
p_lper <- k_local_periodic$plot(show_figure = FALSE)

# Replace scales on all figure
for (p in list(p_se, p_per, p_lper)) {
  p$scales$scales <- list(sxd, syd, sfv)
}

# For good proportions in a notebook, uncomment the following line
# options(repr.plot.width = 11, repr.plot.height = 4.5)

# Make one plot with all kernels (Figure 3)
ggarrange(p_se, p_per, p_lper + ggtitle('Local Periodic Kernel'),
          common.legend = TRUE, ncol = 3, legend = 'bottom')

#####################################
#####################################
# Section 5: Simulation-based study #
#####################################
#####################################

#########################################
### 5.1: Estimation of the parameters ###
#########################################

# Set seed and calculation params
TSR$set_params(seed = 1, fp_type = 'float64', fp_device = 'cpu')

# Use specific covariance matrix for simulation
matern_lenscale <- KernelParameter$new(value = 14)
se_lenscale <- KernelParameter$new(value = 5)
spatial_kernel <- KernelMatern$new(lengthscale = matern_lenscale)
temporal_kernel <- KernelSE$new(lengthscale = se_lenscale)

# Simulate data
simu_data <- simulate_spatiotemporal_data(
  nb_locations = 100,
  nb_time_points = 150,
  nb_spatial_dimensions = 2,
  spatial_scale = 10,
  time_scale = 10,
  spatial_covariates_means = c(0, 2, 4),
  temporal_covariates_means = c(1, 3),
  spatial_kernel = spatial_kernel,
  temporal_kernel = temporal_kernel,
  noise_variance_scale = 1)

# Create Regressor
bktr_regressor <- BKTRRegressor$new(
  data_df = simu_data$data_df,
  spatial_kernel = KernelMatern$new(),
  spatial_positions_df = simu_data$spatial_positions_df,
  temporal_kernel = KernelSE$new(),
  temporal_positions_df = simu_data$temporal_positions_df,
  has_geo_coords = FALSE)

# MCMC sampling
bktr_regressor$mcmc_sampling()

# Print Summary
summary(bktr_regressor)

# Print Beta Errors (Removing Time and Location columns)
beta_err <- unlist(abs(
    bktr_regressor$beta_estimates[, -c(1, 2)]
    - simu_data$beta_df[, -c(1, 2)]
))
print(sprintf('Beta RMSE: %.4f', sqrt(mean(beta_err^2))))
print(sprintf('Beta MAE: %.4f', mean(abs(beta_err))))

# Create Traceplot (Figure 4)
# options(repr.plot.width = 7, repr.plot.height = 4, repr.plot.res = 200)
fig <- plot_hyperparams_traceplot(bktr_regressor, c(
    'Spatial - Matern 5/2 Kernel - lengthscale',
    'Temporal - SE Kernel - lengthscale'
    ), show_figure = FALSE)

col_1 <- '#f87d76'; col_2 <- '#00bfc4';
fig +
    scale_colour_manual(name = 'Name', values = c(col_1, col_2)) +
    geom_hline(yintercept = matern_lenscale$value,
        linetype = 'dashed', col = col_1) +
    geom_hline(yintercept = se_lenscale$value,
        linetype = 'dashed', col = col_2)

# Plot y estimates (Figure 5)
plot_y_estimates(bktr_regressor, fig_title = NULL)


#######################
### 5.2: Imputation ###
#######################
### WARNING: This section may take a long time to run ###
###          and it requires a gpu to run with cuda.  ###

df_res_arr <- c()
res_colnames <- c(
    'Lengthscale', 'Missing', 'Iter', 'B_MAE',
    'B_RMSE', 'Y_MAE', 'Y_RMSE', 'Time'
)

RANK_DECOMP <- 10
BURN_IN_ITER <- 500
SAMPLING_ITER <- 500

# Set seed and calculation params
TSR$set_params(seed = 1, fp_type = 'float32', fp_device = 'cuda')

# Run simulation for different lengthscale and missing percentage
for (len_scale in c(3, 6)) {
  for (miss_perc in c(0.1, 0.5, 0.9)) {
    for (i in 1:10) {
      spatial_kernel <- KernelMatern$new(
        smoothness_factor = 5,
        lengthscale = KernelParameter$new(value = len_scale)
      )
      temporal_kernel <- (
        KernelSE$new(lengthscale = KernelParameter$new(value = len_scale))
      )

      simu_data <- simulate_spatiotemporal_data(
        100, 150, 2, 10, 10,
        c(0, 2, 4), c(1, 3),
        spatial_kernel, temporal_kernel, 1
      )

      data_df <- simu_data$data_df
      index_choices_tsr <- TSR$tensor(1:nrow(data_df))
      nb_miss_index <- round(miss_perc * nrow(data_df))
      na_index <- as.numeric(
        TSR$rand_choice(index_choices_tsr, nb_miss_index)$cpu()
      )
      data_df$y[na_index] <- NA

      bktr_regressor <- BKTRRegressor$new(
        data_df = data_df,
        rank_decomp = RANK_DECOMP,
        burn_in_iter = BURN_IN_ITER,
        sampling_iter = SAMPLING_ITER,
        spatial_kernel = KernelMatern$new(smoothness_factor = 5),
        spatial_positions_df = simu_data$spatial_positions_df,
        temporal_kernel = KernelSE$new(),
        temporal_positions_df = simu_data$temporal_positions_df,
        has_geo_coords = FALSE
      )

      # Hide output of sampling because its volume is too large
      .unused_out <- capture.output(bktr_regressor$mcmc_sampling())

      # Calc Beta Errors
      y_err <- (
        bktr_regressor$imputed_y_estimates$y[na_index]
        - simu_data$data_df$y[na_index]
      )
      beta_err <- unlist(abs(
        lapply(bktr_regressor$beta_estimates[, -c(1, 2)], as.numeric)
        - simu_data$beta_df[, -c(1, 2)]
      ))
      y_rmse <- sqrt(mean(y_err^2))
      y_mae <- mean(abs(y_err))
      beta_rmse <- sqrt(mean(beta_err^2))
      beta_mae <- mean(abs(beta_err))

      # Formatting Values
      df_res_arr <- c(
        df_res_arr,
        len_scale,
        miss_perc,
        sprintf('%04d', i),
        sprintf('%.4f', beta_mae),
        sprintf('%.4f', beta_rmse),
        sprintf('%.4f', y_mae),
        sprintf('%.4f', y_rmse),
        sprintf('%.3f', as.numeric(
            bktr_regressor$result_logger$total_elapsed_time, units = "secs"
        ))
      )
    }
  }
}
df <- as.data.table(
    matrix(df_res_arr, ncol = length(res_colnames), byrow = TRUE)
)
colnames(df) <- res_colnames
print(df)

# Aggregate results (Table 5)
mean_fmt <- function(x) sprintf('%.4f', mean(x))
sd_fmt <- function(x) sprintf('%.4f', sd(x))

df <- df[, lapply(.SD, as.numeric), by = list(Lengthscale, Missing)]
df <- df[, .(
  B_MAE_avg = mean_fmt(B_MAE),
  B_MAE_sd = sd_fmt(B_MAE),
  B_RMSE_avg = mean_fmt(B_RMSE),
  B_RMSE_sd = sd_fmt(B_RMSE),
  Y_MAE_avg = mean_fmt(Y_MAE),
  Y_MAE_sd = sd_fmt(Y_MAE),
  Y_RMSE_avg = mean_fmt(Y_RMSE),
  Y_RMSE_sd = sd_fmt(Y_RMSE),
  Time_avg = mean_fmt(Time),
  Time_sd = sd_fmt(Time)
), by = list(Lengthscale, Missing)]
setkey(df, Lengthscale, Missing)
print(df)


##########################
### 5.3: Interpolation ###
##########################
### WARNING: This section may take a long time to run ###
###          and it requires a gpu to run with cuda.  ###

# Part for Table 6, influence of lengthscale values and dataset size
nb_aside_locs <- 4
nb_aside_times <- 6

# Set seed and calculation params
TSR$set_params(seed = 2, fp_type = 'float32', fp_device = 'cuda')
res_colnames <- c(
    'Dataset_Type', 'Lengthscale', 'Iter', 'B_MAE',
    'B_RMSE', 'Y_MAE', 'Y_RMSE', 'Time'
)
nb_res_cols <- length(res_colnames)
res_vals <- c()

# Run simulation for different lengthscale and dataset size
for (ds_type in c('Smaller', 'Larger')) {
  for (len_scale in c(3, 6)) {
    for (i in 1:10) {
      matern_lengthscale <- KernelParameter$new(value = len_scale)
      se_lengthscale <- KernelParameter$new(value = len_scale)
      spatial_kernel <- KernelMatern$new(
        lengthscale = matern_lengthscale, smoothness_factor = 5)
      temporal_kernel <- KernelSE$new(lengthscale = se_lengthscale)

      is_small_ds <- ds_type == 'Smaller'
      nb_locs <- ifelse(is_small_ds, 20, 100)
      nb_times <- ifelse(is_small_ds, 30, 150)
      spa_cov_means <- if (is_small_ds) c(0, 2) else c(0, 2, 4)
      tem_cov_means <- if (is_small_ds) c(1) else c(1, 3)

      simu_data <- simulate_spatiotemporal_data(
        nb_locations = nb_locs,
        nb_time_points = nb_times,
        nb_spatial_dimensions = 2,
        spatial_scale = 10,
        time_scale = 10,
        spatial_covariates_means = spa_cov_means,
        temporal_covariates_means = tem_cov_means,
        spatial_kernel = spatial_kernel,
        temporal_kernel = temporal_kernel,
        noise_variance_scale = 1
      )

      # Set some values aside for M_new locs and N_new times
      obs_nb_locs <- nb_locs - nb_aside_locs
      obs_nb_times <- nb_times - nb_aside_times

      data_df <- simu_data$data_df
      spatial_pos_df <- simu_data$spatial_positions_df
      temporal_pos_df <- simu_data$temporal_positions_df

      all_locs <- spatial_pos_df$location
      all_times <- temporal_pos_df$time

      locs_indx_sample <- TSR$rand_choice(
        TSR$tensor(1:length(all_locs)), obs_nb_locs)
      obs_locs <- all_locs[as.numeric(locs_indx_sample$cpu())]
      new_locs <- setdiff(all_locs, obs_locs)

      times_indx_sample <- TSR$rand_choice(
        TSR$tensor(1:length(all_times)), obs_nb_times)
      obs_times <- all_times[as.numeric(times_indx_sample$cpu())]
      new_times <- setdiff(all_times, obs_times)

      obs_data_df <- data_df[
        data_df[, .I[location %in% obs_locs & time %in% obs_times]], ]
      obs_spatial_pos_df <- spatial_pos_df[
        spatial_pos_df[, .I[location %in% obs_locs]], ]
      obs_temporal_pos_df <- temporal_pos_df[
        temporal_pos_df[, .I[time %in% obs_times]], ]

      new_data_df <- data_df[
        data_df[, .I[location %in% new_locs | time %in% new_times]], ]
      new_spatial_positions_df <- spatial_pos_df[
        spatial_pos_df[, .I[location %in% new_locs]], ]
      new_temporal_positions_df <- temporal_pos_df[
        temporal_pos_df[, .I[time %in% new_times]], ]

      # Run mcmc sampling
      bktr_regressor <- BKTRRegressor$new(
        data_df = obs_data_df,
        rank_decomp = 10,
        burn_in_iter = 500,
        sampling_iter = 500,
        spatial_kernel = KernelMatern$new(smoothness_factor = 5),
        spatial_positions_df = obs_spatial_pos_df,
        temporal_kernel = KernelSE$new(),
        temporal_positions_df = obs_temporal_pos_df,
        has_geo_coords = FALSE
      )
      # Hide output of sampling because its volume creates notebook errors
      .unused_out <- capture.output(bktr_regressor$mcmc_sampling())

      # Run interpolation
      preds <- bktr_regressor$predict(
        new_data_df,
        new_spatial_positions_df,
        new_temporal_positions_df
      )

      # Align both datasets
      sim_data_df <- simu_data$data_df
      pred_y_df <- preds$new_y_df
      beta_data_df <- simu_data$beta_df
      beta_pred_df <- preds$new_beta_df
      setkey(beta_pred_df, location, time)
      sim_y_df <- sim_data_df[
        sim_data_df[, .I[location %in% new_locs | time %in% new_times]],
        c('location', 'time', 'y')
      ]
      setorderv(pred_y_df, c('location', 'time'))
      setorderv(sim_y_df, c('location', 'time'))

      # Calc Errors
      preds_y_err <- (
        sim_data_df[
          sim_data_df[, .I[location %in% new_locs | time %in% new_times]],
          'y']
        - pred_y_df[
            pred_y_df[, .I[location %in% new_locs | time %in% new_times]],
            'y_est']
      )
      preds_y_err <- unlist(preds_y_err)
      preds_beta_err <- (
        beta_data_df[
          beta_data_df[, .I[location %in% new_locs | time %in% new_times]],
          -c('location', 'time')]
        - beta_pred_df[
            beta_pred_df[, .I[location %in% new_locs | time %in% new_times]],
            -c('location', 'time')]
      )
      preds_beta_err <- unlist(preds_beta_err)

      y_rmse <- sqrt(mean(preds_y_err^2))
      y_mae <- mean(abs(preds_y_err))
      beta_rmse <- sqrt(mean(preds_beta_err^2))
      beta_mae <- mean(abs(preds_beta_err))

      # Formatting Values
      res_vals <- c(
        res_vals,
        ds_type,
        len_scale,
        sprintf('%04d', i),
        sprintf('%.4f', beta_mae),
        sprintf('%.4f', beta_rmse),
        sprintf('%.4f', y_mae),
        sprintf('%.4f', y_rmse),
        sprintf('%.3f', as.numeric(
          bktr_regressor$result_logger$total_elapsed_time,units="secs"
        ))
      )
    }
  }
}
df <- as.data.table(matrix(res_vals, ncol = nb_res_cols, byrow = TRUE))
colnames(df) <- res_colnames
print(df)

# Aggregate results (Table 6)
mean_fmt <- function(x) sprintf('%.4f', mean(x))
sd_fmt <- function(x) sprintf('%.4f', sd(x))
df <- df[, lapply(.SD, as.numeric), by = list(Dataset_Type, Lengthscale)]
df <- df[, .(
  B_MAE_avg = mean_fmt(B_MAE),
  B_MAE_sd = sd_fmt(B_MAE),
  B_RMSE_avg = mean_fmt(B_RMSE),
  B_RMSE_sd = sd_fmt(B_RMSE),
  Y_MAE_avg = mean_fmt(Y_MAE),
  Y_MAE_sd = sd_fmt(Y_MAE),
  Y_RMSE_avg = mean_fmt(Y_RMSE),
  Y_RMSE_sd = sd_fmt(Y_RMSE),
  Time_avg = mean_fmt(Time),
  Time_sd = sd_fmt(Time)
), by = list(Dataset_Type, Lengthscale)]
setkey(df, Dataset_Type, Lengthscale)
print(df)



# Part for Table 7, Interpolation results by interpolated segments
# Functions for formatting purposes
get_df_errors <- function(
  l_df, l_col, r_df, r_col, loc_subset, time_subset, use_and = TRUE
) {
  if (use_and) {
    l_indx <- which(
      l_df[, location] %in% loc_subset & l_df[, time] %in% time_subset)
    r_indx <- which(
      r_df[, location] %in% loc_subset & r_df[, time] %in% time_subset)
  } else {
    l_indx <- which(
      l_df[, location] %in% loc_subset | l_df[, time] %in% time_subset)
    r_indx <- which(
      r_df[, location] %in% loc_subset | r_df[, time] %in% time_subset)
  }
  err <- unlist(l_df[l_indx, ..l_col] - r_df[r_indx, ..r_col])
  return(c(mean(abs(err)), sqrt(mean(err ** 2))))
}

get_all_errors <- function(
    segment_name,
    i,
    sim_data_df,
    pred_y_df,
    beta_data_df,
    beta_pred_df,
    loc_subset,
    time_subset,
    use_and = TRUE
) {
  y_err <- get_df_errors(sim_data_df, 'y', pred_y_df, 'y_est',
                         loc_subset, time_subset, use_and = use_and)
  beta_colnames <- setdiff(colnames(beta_pred_df), c('location', 'time'))
  beta_err <- get_df_errors(beta_data_df, beta_colnames, beta_pred_df,
      beta_colnames, loc_subset, time_subset, use_and = use_and)

  return(c(
    segment_name,
    sprintf('%04d', i),
    sprintf('%.4f', beta_err[1]),
    sprintf('%.4f', beta_err[2]),
    sprintf('%.4f', y_err[1]),
    sprintf('%.4f', y_err[2])
  ))
}

# Simulation Params
nb_aside_locs <- 10
nb_aside_times <- 20
nb_locs <- 100
nb_times <- 150
len_scale <- 6
spa_cov_means <- c(0, 2, 4)
tem_cov_means <- c(1, 3)

# Set seed and calculation params
TSR$set_params(seed = 1, fp_type = 'float32', fp_device = 'cuda')
res_colnames <- c(
  'Interpol_Segment', 'Iter', 'B_MAE',
  'B_RMSE', 'Y_MAE', 'Y_RMSE'
)
nb_res_cols <- length(res_colnames)
res_vals <- c()

# Run Large simulation for multiple iterations
for (i in 1:10) {
  matern_lengthscale <- KernelParameter$new(value = len_scale)
  se_lengthscale <- KernelParameter$new(value = len_scale)
  spatial_kernel <- KernelMatern$new(lengthscale = matern_lengthscale)
  temporal_kernel <- KernelSE$new(lengthscale = se_lengthscale)

  simu_data <- simulate_spatiotemporal_data(
    nb_locations = nb_locs,
    nb_time_points = nb_times,
    nb_spatial_dimensions = 2,
    spatial_scale = 10,
    time_scale = 10,
    spatial_covariates_means = spa_cov_means,
    temporal_covariates_means = tem_cov_means,
    spatial_kernel = spatial_kernel,
    temporal_kernel = temporal_kernel,
    noise_variance_scale = 1
  )

  # Set some values aside for M_new locs and N_new times
  obs_nb_locs <- nb_locs - nb_aside_locs
  obs_nb_times <- nb_times - nb_aside_times

  data_df <- simu_data$data_df
  spatial_pos_df <- simu_data$spatial_positions_df
  temporal_pos_df <- simu_data$temporal_positions_df

  all_locs <- spatial_pos_df$location
  all_times <- temporal_pos_df$time
  obs_locs <- sample(all_locs, obs_nb_locs)
  new_locs <- setdiff(all_locs, obs_locs)
  obs_times <- sample(all_times, obs_nb_times)
  new_times <- setdiff(all_times, obs_times)

  obs_data_df <- data_df[
    data_df[, .I[location %in% obs_locs & time %in% obs_times]], ]
  obs_spatial_pos_df <- spatial_pos_df[
    spatial_pos_df[, .I[location %in% obs_locs]], ]
  obs_temporal_pos_df <- temporal_pos_df[
    temporal_pos_df[, .I[time %in% obs_times]], ]

  new_data_df <- data_df[
    data_df[, .I[location %in% new_locs | time %in% new_times]], ]
  new_spatial_positions_df <- spatial_pos_df[
    spatial_pos_df[, .I[location %in% new_locs]], ]
  new_temporal_positions_df <- temporal_pos_df[
    temporal_pos_df[, .I[time %in% new_times]], ]

  # Run mcmc sampling
  bktr_regressor <- BKTRRegressor$new(
    data_df = obs_data_df,
    spatial_kernel = KernelMatern$new(),
    spatial_positions_df = obs_spatial_pos_df,
    temporal_kernel = KernelSE$new(),
    temporal_positions_df = obs_temporal_pos_df,
    burn_in_iter = 1000,
    sampling_iter = 500,
    has_geo_coords = FALSE
  )
  # Hide output of sampling because its volume creates notebook errors
  .unused_out <- capture.output(bktr_regressor$mcmc_sampling())

  # Run interpolation
  preds <- bktr_regressor$predict(
    new_data_df,
    new_spatial_positions_df,
    new_temporal_positions_df
  )

  # Align both datasets
  sim_data_df <- simu_data$data_df
  pred_y_df <- preds$new_y_df
  beta_data_df <- simu_data$beta_df
  beta_pred_df <- preds$new_beta_df
  setkey(beta_pred_df, location, time)
  sim_y_df <- sim_data_df[
    sim_data_df[, .I[location %in% new_locs | time %in% new_times]],
    c('location', 'time', 'y')
  ]
  setorderv(pred_y_df, c('location', 'time'))
  setorderv(sim_y_df, c('location', 'time'))

  # Formatting Values
  res_vals <- c(
    res_vals,
    get_all_errors('1_new_spa',  i, sim_data_df, pred_y_df, beta_data_df,
                   beta_pred_df, new_locs, obs_times),
    get_all_errors('2_new_temp', i, sim_data_df, pred_y_df, beta_data_df,
                   beta_pred_df, obs_locs, new_times),
    get_all_errors('3_new_both', i, sim_data_df, pred_y_df, beta_data_df,
                   beta_pred_df, new_locs, new_times),
    get_all_errors('new_total', i, sim_data_df, pred_y_df, beta_data_df,
                   beta_pred_df, new_locs, new_times, use_and = FALSE)
  )
}
df <- as.data.table(matrix(res_vals, ncol = nb_res_cols, byrow = TRUE))
colnames(df) <- res_colnames
print(df)

# Aggregate results (Table 7)
mean_fmt <- function(x) sprintf('%.4f', mean(x))
sd_fmt <- function(x) sprintf('%.4f', sd(x))
df <- df[, lapply(.SD, as.numeric), by = list(Interpol_Segment)]
df <- df[, .(
    B_MAE_avg = mean_fmt(B_MAE),
    B_MAE_sd = sd_fmt(B_MAE),
    B_RMSE_avg = mean_fmt(B_RMSE),
    B_RMSE_sd = sd_fmt(B_RMSE),
    Y_MAE_avg = mean_fmt(Y_MAE),
    Y_MAE_sd = sd_fmt(Y_MAE),
    Y_RMSE_avg = mean_fmt(Y_RMSE),
    Y_RMSE_sd = sd_fmt(Y_RMSE)
), by = list(Interpol_Segment)]
setkey(df, Interpol_Segment)
print(df)


#################################
#################################
# Section 6: Experimental study #
#################################
#################################

#################################################
# WARNING: This section uses a large dataset.   #
#     It requires a GPU to run with cuda.       #
#     If also requires a fair amount of memory. #
#################################################

#####################
### 6.1: Analysis ###
#####################

library(BKTR)

TSR$set_params(seed = 0, fp_type = 'float32', fp_device = 'cuda')
bixi_data <- BixiData$new()

p_lgth <- KernelParameter$new(value = 7, is_fixed = TRUE)
k_local_periodic <- KernelSE$new() * KernelPeriodic$new(period_length = p_lgth)
bktr_regressor <- BKTRRegressor$new(
  formula = nb_departure ~ 1 + mean_temp_c + area_park + total_precip_mm,
  data_df = bixi_data$data_df,
  spatial_positions_df = bixi_data$spatial_positions_df,
  temporal_positions_df = bixi_data$temporal_positions_df,
  rank = 8,
  spatial_kernel = KernelMatern$new(smoothness_factor = 5),
  temporal_kernel = k_local_periodic,
  burn_in_iter = 1000,
  sampling_iter = 500)
bktr_regressor$mcmc_sampling()

# Print Summary of BKTR regressor
summary(bktr_regressor)

# Plot BIXI betas through time (Figure 6)
plot_temporal_betas(
    bktr_regressor,
    plot_feature_labels = c('mean_temp_c', 'area_park', 'total_precip_mm'),
    spatial_point_label = '7114 - Smith / Peel',
    fig_width = 7, fig_height = 3.75)

# Plot BIXI betas through space (Figure 7)
plot_spatial_betas(
    bktr_regressor,
    plot_feature_labels = c('mean_temp_c', 'total_precip_mm'),
    temporal_point_label = '2019-07-01',
    nb_cols = 2,
    fig_width = 7, fig_height = 2.75,
    stadia_token = stadia_api_token)

# Plot BIXI y estimates (Figure 8)
plot_y_estimates(bktr_regressor, fig_title = NULL)

###############################
### 6.2: Imputation Example ###
###############################

y_is_na <- is.na(bixi_data$data_df$nb_departure)
nb_y_na <- sum(y_is_na)
sprintf(
  'There is %.d missing `nb_departure` values representing ~%.2f%%',
  nb_y_na,
  nb_y_na / length(y_is_na) * 100)

print(bixi_data$data_df[which(y_is_na)[1:3], 1:3])

print(bktr_regressor$imputed_y_estimates[which(y_is_na)[1:3]])

##################################
### 6.3: Interpolation Example ###
##################################

TSR$set_params(seed = 0, fp_type = 'float32', fp_device = 'cuda')
bixi_data <- BixiData$new()
data_df <- bixi_data$data_df
spa_df <- bixi_data$spatial_positions_df
tem_df <- bixi_data$temporal_positions_df

# Separate data in old vs new batches
new_s <- c(
  '4002 - Graham / Wicksteed',
  '7079 - Notre-Dame / Gauvin',
  '6236 - Laurier / de Bordeaux'
)
new_t <- c('2019-05-01', '2019-05-02')
new_t <- as.IDate(new_t)
# Get obs data
obs_s <- setdiff(unlist(spa_df$location), new_s)
obs_t <- as.IDate(setdiff(unlist(tem_df$time), new_t))

obs_data_df <- data_df[data_df[, .I[
  location %in% obs_s & time %in% obs_t]], ]
obs_spa_df <- spa_df[spa_df[, .I[location %in% obs_s]], ]
obs_tem_df <- tem_df[tem_df[,.I[time %in% obs_t]], ]
# Get new data
new_data_df <- data_df[data_df[, .I[
  location %in% new_s | time %in% new_t]], ]
new_spa_df <- spa_df[spa_df[, .I[location %in% new_s]], ]
new_tem_df <- tem_df[tem_df[, .I[time %in% new_t]], ]

bktr_regressor <- BKTRRegressor$new(
  data_df = obs_data_df,
  spatial_positions_df = obs_spa_df,
  temporal_positions_df = obs_tem_df,
  #... other parameters like section 6.1
  formula = nb_departure ~ 1 + mean_temp_c + area_park + total_precip_mm,
  rank = 8,
  spatial_kernel = KernelMatern$new(smoothness_factor = 5),
  temporal_kernel = (
    KernelSE$new() *
      KernelPeriodic$new(
        period_length = KernelParameter$new(value = 7, is_fixed = TRUE)
      )
  ),
  burn_in_iter = 1000,
  sampling_iter = 500)
bktr_regressor$mcmc_sampling()

# Prediction on unobserved data
preds <- bktr_regressor$predict(
  new_data_df,
  new_spa_df,
  new_tem_df
)
new_data_df <- data_df[
  data_df[, .I[location %in% new_s | time %in% new_t]],
  c('location', 'time', 'nb_departure')
]
pred_y_df <- preds$new_y_df
# Sort data for comparison and remove na values
setkey(new_data_df, location, time)
setkey(pred_y_df, location, time)
non_na_indices <- which(!is.na(new_data_df$nb_departure))

y_err <- (
    new_data_df$nb_departure[non_na_indices]
    - pred_y_df$y_est[non_na_indices]
)
sprintf('Predicting %d y values || MAE: %.4f || RMSE: %.4f',
        length(non_na_indices), mean(abs(y_err)), sqrt(mean(y_err ^ 2)))

# Plot y predicted vs y observed (Figure 9)
y_list <- new_data_df$nb_departure[non_na_indices]
y_est_list <- pred_y_df$y_est[non_na_indices]
min_y <- min(y_list)
max_y <- max(y_list)
df <- data.table(y = y_list, y_est = y_est_list)
fig <- (
    ggplot(df, aes(x = .data$y, y = .data$y_est))
    + geom_point(color = '#39a7d0', alpha = 0.6, shape = 21, fill = '#20a0d0')
    + geom_segment(aes(x = min_y, y = min_y, xend = max_y, yend = max_y),
                   color = 'black', linetype = 'twodash', linewidth = 1)
    + theme_bw()
    + ylab('Estimated y')
    + xlab('Observed y')
    + ggtitle('Interpolated y estimates vs observed y values')
)
print(fig)


################
################
#   Appendix   #
################
################

########################################
### Appendix B: Covariates reshaping ###
########################################

# Create full bixi data set
bixi_data <- BixiData$new()

# Ensure the data is reshaped to the right format
spatial_df <- bixi_data$spatial_features_df
temporal_df <- bixi_data$temporal_features_df
y_df <- bixi_data$departure_df
p_s <- ncol(spatial_df) - 1 # Not counting index column
p_t <- ncol(temporal_df) - 1
sprintf('Response M=%d and N=%d', nrow(y_df), ncol(y_df))
sprintf('Spatial features M=%d x p_s=%d', nrow(spatial_df), p_s)
sprintf('Temporal features N=%d x p_t=%d', nrow(temporal_df), p_t)
data_df <- reshape_covariate_dfs(spatial_df, temporal_df,
    y_df, 'nb_departure')
sprintf('Should obtain MN=%d x P=%d', nrow(spatial_df) *
   nrow(temporal_df), 1 + p_s + p_t)
sprintf('Reshaped MN=%d x P=%d', nrow(data_df), ncol(data_df) - 2)


#############################################################
# Appendix D: Influence of device and floating point format #
#############################################################
# WARNING: Very long to run                                 #
#############################################################
TSR$set_params(seed = 1)

res_colnames <- c(
  'Device', 'FP_Type', 'Iter', 'Y_RMSE',
  'Y_MAE', 'B_RMSE', 'B_MAE', 'Time'
)

nb_res_cols <- length(res_colnames)
res_vals <- c()
burn_in_iter <- 500
sampling_iter <- 500
for (fp_device in c('cuda', 'cpu')) {
  for (fp_type in c('float64', 'float32')) {
    for (i in 1:10) {
      TSR$set_params(fp_type = fp_type, fp_device = fp_device)
      matern_lengthscale <- KernelParameter$new(value = 14)
      se_lengthscale <- KernelParameter$new(value = 5)
      spatial_kernel <- KernelMatern$new(lengthscale = matern_lengthscale)
      temporal_kernel <- KernelSE$new(lengthscale = se_lengthscale)

      simu_data <- simulate_spatiotemporal_data(
        nb_locations = 100,
        nb_time_points = 150,
        nb_spatial_dimensions = 2,
        spatial_scale = 10,
        time_scale = 10,
        spatial_covariates_means = c(0, 2, 4),
        temporal_covariates_means = c(1, 3),
        spatial_kernel = spatial_kernel,
        temporal_kernel = temporal_kernel,
        noise_variance_scale = 1
      )

      bktr_regressor <- BKTRRegressor$new(
        data_df = simu_data$data_df,
        spatial_kernel = KernelMatern$new(),
        spatial_positions_df = simu_data$spatial_positions_df,
        temporal_kernel = KernelSE$new(),
        temporal_positions_df = simu_data$temporal_positions_df,
        burn_in_iter = burn_in_iter,
        sampling_iter = sampling_iter,
        has_geo_coords = FALSE
      )
      bktr_regressor$mcmc_sampling()

      # Calc Beta Errors
      beta_err <- unlist(abs(
        lapply(bktr_regressor$beta_estimates[, -c(1, 2)], as.numeric)
        - simu_data$beta_df[, -c(1, 2)]
      ))
      beta_rmse <- sqrt(mean(beta_err^2))
      beta_mae <- mean(abs(beta_err))
      # Formatting Values
      res_vals <- c(
        res_vals,
        fp_device,
        fp_type,
        sprintf('%04d', i),
        sprintf('%.4f', bktr_regressor$result_logger$error_metrics$RMSE),
        sprintf('%.4f', bktr_regressor$result_logger$error_metrics$MAE),
        sprintf('%.4f', beta_rmse),
        sprintf('%.4f', beta_mae),
        sprintf('%.3f', as.numeric(
          bktr_regressor$result_logger$total_elapsed_time, units = "secs"
        ))
      )
      df <- as.data.table(matrix(res_vals, ncol = nb_res_cols, byrow = TRUE))
      colnames(df) <- res_colnames
      print(df)
    }
  }
}

# Aggregate results (Table 8)
mean_fmt <- function(x) sprintf('%.4f', mean(x))
sd_fmt <- function(x) sprintf('%.4f', sd(x))
df <- df[, lapply(.SD, as.numeric), by = list(Device, FP_Type)]
df <- df[, .(
    Y_RMSE_avg = mean_fmt(Y_RMSE),
    Y_RMSE_sd = sd_fmt(Y_RMSE),
    Y_MAE_avg = mean_fmt(Y_MAE),
    Y_MAE_sd = sd_fmt(Y_MAE),
    B_RMSE_avg = mean_fmt(B_RMSE),
    B_RMSE_sd = sd_fmt(B_RMSE),
    B_MAE_avg = mean_fmt(B_MAE),
    B_MAE_sd = sd_fmt(B_MAE),
    Time_avg = mean_fmt(Time),
    Time_sd = sd_fmt(Time)
), by = list(Device, FP_Type)]
print(df)
