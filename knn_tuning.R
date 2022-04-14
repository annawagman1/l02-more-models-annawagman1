# 2. Nearest neighbors tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(janitor)

set.seed(3012)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
load("data/wf_dataprep.rda")

# Recipe adjustments
knn_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

# Define model ----
knn_model <- nearest_neighbor(mode = "classification",
                              neighbors = tune()) %>% 
  set_engine("kknn")

# set-up tuning grid ----
knn_params <- parameters(knn_model)

# define tuning grid
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow ----
knn_wflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(knn_recipe)

# Tuning/fitting ----
tic("K-Nearest Neighbor")

# tuning
knn_tune <- knn_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = knn_grid
  )

toc(log = TRUE)

# save run time info
knn_tt <- tic.log(format = TRUE) 

# Write out results & workflow
save(knn_tune, knn_wflow, knn_tt, file = "model_info/knn_results.rda")
