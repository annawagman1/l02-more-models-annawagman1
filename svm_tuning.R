# 5. Support vector machine tuning ----

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
svm_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

# Define model ----
svm_model <- svm_poly(
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

# set-up tuning grid ----
parameters(svm_model)
svm_params <- parameters(svm_model)

# define tuning grid
svm_grid <- grid_regular(svm_params, levels = 5)


# workflow ----
svm_wflow <- workflow() %>% 
  add_model(svm_model) %>% 
  add_recipe(svm_recipe)

# Tuning/fitting ----
tic("Support vector machine")

# tuning
svm_tune <- svm_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = svm_grid
  )

toc(log = TRUE)

# save run time info
svm_tt <- tic.log(format = TRUE) 

# Write out results & workflow
save(svm_tune, svm_wflow, svm_tt, file = "model_info/svm_results.rda")
