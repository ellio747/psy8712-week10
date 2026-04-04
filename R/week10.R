# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# library(haven) # Read SPSS files
library(tidyverse) # Data analysis tools
library(caret) # Machine learning; loaded per 2.7.1
# remotes::install_version("xgboost", version = "1.6.0.1", repos = "https://cran.r-project.org") # Compatible version of xgboost with caret - install this version in R Console before continuing. 
library(xgboost)

# Data Import and Cleaning
gss_tbl <- haven::read_sav("../data/GSS2016.sav", user_na = T) %>% # imported calling `haven` for `read_sav` to read an SPSS file and user_na to remove user defined na variables
  haven::zap_missing() %>% # used to convert tagged missingingness (-100, -99, -98, etc which I found running unique(gss_tbl$mosthrs without `zap_missing()`)
  filter(!is.na(mosthrs)) %>% # keeps only the 570 observations in `mosthrs` that are not na; confirmed with length(gss_tbl$mosthrs)
  # rename(`work hours` = mosthrs) %>% #renamed as per 2.5.3
  select(-c(hrs1, hrs2)) %>% #removed as per 2.5.4
  select(which(colMeans(is.na(.)) < 0.75)) %>% #retained only the variables which had < 75% missingness  
  mutate(across(where(haven::is.labelled), as.numeric)) %>% # removed haven labeling
  mutate(mosthrs = as.numeric(mosthrs)) %>% # turned work hours into a numeric call for visualization
  as_tibble() # coerced to tibble as greater compatibility with `caret`

# Visualization
ggplot(gss_tbl, aes(x = mosthrs)) + # used ggplot to display `work hours`, from `gss_tbl`
  geom_histogram(binwidth = 4) # selected histogram for a univariate distribution; I was getting a warning to pick a better binwidth; I chose 4 as it serves as a typical half workday

# Analysis

## Define training and test sets
set.seed(123)
holdout_indices <- createDataPartition(gss_tbl$mosthrs, p = .75, list=F) # establishes 75/25 split of data
gss_training <- gss_tbl[holdout_indices,] 
gss_holdout <- gss_tbl[-holdout_indices,]

# gss_training_pp <- preProcess(gss_training,
#                               method=c("zv","medianImpute","center","scale"))
# gss_training_pp_df <- predict(gss_training_pp, gss_training)

## Define consistent folds 
cv_control <- trainControl(
  method="cv", # cross validation
  number=10, # 10-fold
  search = "random",
  verboseIter = T # will print the training log
)

## Decide on hyperparameters to look at
tune_grid_enet <- expand.grid( # incorporated grid search for plausible range of hyperparamters
  alpha = c(0,1), # this tunes between ridge and lasso to enable elastic net when using `glmnet` as method
  lambda = seq(0.0001, 0.1, length = 10) #penalization/regularization parameter
)

tune_grid_rf <- expand.grid( # incorporated grid search for plausible range of hyperparamters
  mtry = c(23, 178, 267, 535), # sets default of rounded down square root of number of IVs, two intermediate values, and number of IVs; included based on nearZeroVariance(gss_training)
  splitrule = c("variance", "extratrees", "maxstat"),  
  min.node.size = 5 # 5 is the default for regression according to help("ranger")
)

tune_grid_xgb <- expand.grid( # incorporated grid search for plausible range of hyperparamters
  nrounds = c(50, 150, 300), # number of boosting rounds
  alpha = c(0, 1), # this tunes the L1 and L2 regularization parameter
  lambda = seq(0.0001, 0.1, length = 5), # penalization/regularization cost parameter
  eta = c(0.3, 0.5) # default boosters for linear 0.5; 0.3 is default for tree-based booster 
)

## Run k-fold testing hyperparameters on training set

model1 <- train(
  mosthrs ~ ., # `work hours` from all variables in the dataset
  gss_training,
  na.action = na.pass,
  method = "lm",
  preProcess = c("zv", "medianImpute","center","scale"), # there were 35 variables showing near zero variance which challenged the model; added in "zv" to account for these across all models
  trControl=cv_control
  )
model1

model2 <- train(
  mosthrs ~ .,
  gss_training,
  na.action = na.pass,
  method = "glmnet",
  preProcess = c("zv", "medianImpute","center","scale"),
  tuneGrid = tune_grid_enet,
  trControl = cv_control
)
model2

model3 <- train(
  mosthrs ~ .,
  gss_training,
  na.action = na.pass,
  method = "ranger",
  preProcess = c("zv", "medianImpute","center","scale"),
  tuneGrid = tune_grid_rf,
  trControl=cv_control
)
model3

model4 <- train(
  mosthrs ~.,
  gss_training,
  na.action = na.pass,
  method = "xgbLinear",
  preProcess = c("zv", "medianImpute", "center", "scale"),
  tuneGrid = tune_grid_xgb,
  trControl = cv_control
)
model4

## Examine k-fold CV results
summary(resamples(list(model1,model2, model3, model4)))
dotplot(resamples(list(model1,model2, model3, model4)))

## Compare to holdout CV results
pred1 <- predict(model1, newdata = gss_holdout, na.action = na.pass)
pred2 <- predict(model2, newdata = gss_holdout, na.action = na.pass)
pred3 <- predict(model3, newdata = gss_holdout, na.action = na.pass)
pred4 <- predict(model4, newdata = gss_holdout, na.action = na.pass)

# Publication
table1_tbl <- tibble(
  algo = c(model1$method, model2$method, model3$method, model4$method),
  cv_rsq = str_remove(round(c(getTrainPerf(model1)$TrainRsquared, 
             getTrainPerf(model2)$TrainRsquared, 
             getTrainPerf(model3)$TrainRsquared,
             getTrainPerf(model4)$TrainRsquared), 2), "^0"),
  ho_rsq = str_remove(round(c(postResample(pred1, gss_holdout$mosthrs)["Rsquared"], 
             postResample(pred2, gss_holdout$mosthrs)["Rsquared"], 
             postResample(pred3, gss_holdout$mosthrs)["Rsquared"], 
             postResample(pred4, gss_holdout$mosthrs)["Rsquared"]), 2), "^0")
) %>% 
  write_csv(file = "../figs/table1.csv")

# How did your results change between models? Why do you think this happened, specifically?
# How did you results change between k-fold CV and holdout CV? Why do you think this happened, specifically?
# Among the four models, which would you choose for a real-life prediction problem, and why? Are there tradeoffs? Write up to a paragraph.
