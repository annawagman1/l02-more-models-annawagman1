# 7. Single Layer Neural Network tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(janitor)
library(kernlab)

set.seed(3012)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
load("data/wf_dataprep.rda")

# Recipe adjustments
slnn_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

# Define model ----
slnn_model <- mlp(
  hidden_units = tune(),
  penalty = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("nnet")

# set-up tuning grid ----
parameters(slnn_model)
slnn_params <- parameters(slnn_model)

# define tuning grid
slnn_grid <- grid_regular(slnn_params, levels = 5)


# workflow ----
slnn_wflow <- workflow() %>% 
  add_model(slnn_model) %>% 
  add_recipe(slnn_recipe)

# Tuning/fitting ----
tic("Single layer neural network")

# tuning
slnn_tune <- slnn_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = slnn_grid
  )

toc(log = TRUE)

# save run time info
slnn_tt <- tic.log(format = TRUE) 

# Write out results & workflow
save(slnn_tune, slnn_wflow, slnn_tt, file = "model_info/slnn_results.rda")
