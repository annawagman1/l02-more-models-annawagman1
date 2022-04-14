### L02: More Models
## 4/12/22
# Anna Wagman

#load packages
library(tidymodels)
library(readr)
library(dplyr)
library(workflows)
library(rsample)
library(recipes)
library(tune)

tidymodels_prefer()


#import data set
wildfires <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))
  ) %>%
  select(-burned)

##prep data for modeling
#make testing and training sets
wf_split <- initial_split(wildfires, prop = .8)
wf_train <- training(wf_split)
wf_test <- testing(wf_split)

#cross fold validation
wf_folds <- wf_train %>%
  vfold_cv(v = 5, repeats = 3)

#recipe for wlf 
wf_recipe <- recipe(wlf ~ ., data = wf_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

wf_recipe %>%
  prep() %>%
  bake(new_data = NULL)

save(wildfires, wf_folds, wf_test, wf_train, wf_recipe, wf_split, file = "data/wf_dataprep.rda")