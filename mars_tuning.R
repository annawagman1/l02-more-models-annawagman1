# 8. Multivariate adaptive regression splines tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(janitor)
library(earth)

set.seed(3012)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
load("data/wf_dataprep.rda")

# Recipe adjustments
mars_recipe <- recipe(wlf ~ . , data = wf_train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())

# Define model ----
mars_model <- mars(
  num_terms = tune(),
  prod_degree = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("earth")

# set-up tuning grid ----
mars_params <- parameters(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 25)))

# define tuning grid
mars_grid <- grid_regular(mars_params, levels = 5)


# workflow ----
mars_wflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(mars_recipe)

# Tuning/fitting ----
tic("Multivariate adaptive regression splines")

# tuning
mars_tune <- mars_wflow %>% 
  tune_grid(
    resamples = wf_folds,
    grid = mars_grid
  )

toc(log = TRUE)

# save run time info
mars_tt <- tic.log(format = TRUE) 

# Write out results & workflow
save(mars_tune, mars_wflow, mars_tt, file = "model_info/mars_results.rda")
