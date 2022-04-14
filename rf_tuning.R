# 3. Random Forest tuning ----

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
rf_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

# Define model ----
rf_model <- rand_forest(
  min_n = tune(),
  mtry = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

# set-up tuning grid ----
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 10)))

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_wflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rf_recipe)

# Tuning/fitting ----
tic("Random Forest")

# tuning
rf_tune <- rf_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = rf_grid
  )

toc(log = TRUE)

# save run time info
rf_tt <- tic.log(format = TRUE) 

#accuracy
rf_best <- rf_tune %>%
  collect_metrics(metric = "accuracy") %>%
  arrange(-mean) %>%
  dplyr::slice(1L) %>%
  select(mean) %>%
  mutate(model = "rf")

# Write out results & workflow
save(rf_tune, rf_wflow, rf_tt, rf_best, file = "model_info/rf_results.rda")
