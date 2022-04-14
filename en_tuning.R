# 1. Elastic net (logistic_reg) tuning ----

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
en_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_interact(wlf ~ .^2) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) 

# Define model ----
en_model <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# set-up tuning grid ----
en_params <- parameters(en_model)

# define tuning grid
en_grid <- grid_regular(en_params, levels = 5)

# workflow ----
en_wflow <- workflow() %>%
  add_model(en_model) %>%
  add_recipe(en_recipe)

# Tuning/fitting ----
tic("Elastic Net")

# tuning
en_tune <- en_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = en_grid
  )

toc(log = TRUE)

# save run time info
en_tt <- tic.log(format = TRUE) 

# Write out results & workflow
save(en_tune, en_wflow, en_tt, file = "model_info/en_results.rda")
