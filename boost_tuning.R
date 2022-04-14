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
boost_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

# Define model ----
boost_model <- boost_tree(
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")

# set-up tuning grid ----
boost_params <- parameters(boost_model) %>% 
  update(mtry = mtry(range = c(2, 10)))

# define tuning grid
boost_grid <- grid_regular(boost_params, levels = 5)

# workflow ----
boost_wflow <- workflow() %>%
  add_model(boost_model) %>%
  add_recipe(boost_recipe)

# Tuning/fitting ----
tic("Boosted Tree")

# tuning
boost_tune <- boost_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = boost_grid
  )

toc(log = TRUE)

# save run time info
boost_tt <- tic.log(format = TRUE) 

# accuracy mean
boost_best <- boost_tune %>%
  collect_metrics(metric = "accuracy") %>%
  arrange(-mean) %>%
  dplyr::slice(1L) %>%
  select(mean) %>%
  mutate(model = "boost")

# Write out results & workflow
save(boost_tune, boost_wflow, boost_tt, boost_best, file = "model_info/boost_results.rda")
