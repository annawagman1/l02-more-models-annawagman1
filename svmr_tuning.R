# 6. Support vector machine (radial) tuning ----

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
svmr_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

# Define model ----
svmr_model <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

# set-up tuning grid ----
parameters(svmr_model)
svmr_params <- parameters(svmr_model)

# define tuning grid
svmr_grid <- grid_regular(svmr_params, levels = 5)


# workflow ----
svmr_wflow <- workflow() %>% 
  add_model(svmr_model) %>% 
  add_recipe(svmr_recipe)

# Tuning/fitting ----
tic("Support vector machine (radial)")

# tuning
svmr_tune <- svmr_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = svmr_grid
  )

toc(log = TRUE)

# save run time info
svmr_tt <- tic.log(format = TRUE) 

# Write out results & workflow
save(svmr_tune, svmr_wflow, svmr_tt, file = "model_info/svmr_results.rda")
