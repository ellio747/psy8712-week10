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
  rename(`work hours` = mosthrs) %>% #renamed as per 2.5.3
  select(-c(hrs1, hrs2)) %>% #removed as per 2.5.4
  select(which(colMeans(is.na(.)) < 0.75)) %>% #retained only the variables which had < 75% missingness  
  mutate(across(where(haven::is.labelled), as.numeric)) %>% # removed haven labeling
  mutate(`work hours` = as.numeric(`work hours`)) %>% # turned work hours into a numeric call for visualization
  as_tibble() # coerced to tibble as greater compatibility with `caret`

# Visualization
(ggplot(gss_tbl, aes(x = `work hours`)) + # used ggplot to display `work hours`, from `gss_tbl`
  geom_histogram(binwidth = 4)) %>%  # selected histogram for a univariate distribution; I was getting a warning to pick a better binwidth; I chose 4 as it serves as a typical half workday
  ggsave(filename = "../figs/hours_histogram.png", width = 1920, height = 1080, units = "px", dpi = 300)

# Analysis (I used slide 32 from the Week 10 lecture to subset my steps below)

## Define training and test sets
set.seed(123) # set seed for reproducibility
holdout_indices <- createDataPartition(gss_tbl$`work hours`, p = .75, list=F) # establishes 75/25 split of data
gss_training <- gss_tbl[holdout_indices,] # creates training dataset
gss_holdout <- gss_tbl[-holdout_indices,] # creates holdout dataset

## Define consistent folds 
cv_control <- trainControl(
  method="cv", # cross validation
  number=10, # 10-fold
  search = "random",
  verboseIter = T # will print the training log
)

## Run k-fold testing hyperparameters on training set

model1 <- train(
  `work hours` ~ ., # `work hours` from all variables in the dataset
  gss_training, # training dataset
  na.action = na.pass, # added na.pass to enable skipping the nas in each variable
  method = "lm", #OLS Linear Model as required
  preProcess = c("zv", "medianImpute","center","scale"), # there were 35 variables showing zero variance which challenged the model; added in "zv" to account for these across all models
  trControl=cv_control # Use consistent fold model
  )
model1

model2 <- train(
  `work hours` ~ ., # `work hours` from all variables in the dataset
  gss_training, # training dataset
  na.action = na.pass, # added na.pass to enable skipping the nas in each variable
  method = "glmnet", # elastic net, using glmnet with hyperparamter (alpha = 0 or 1)
  preProcess = c("zv", "medianImpute","center","scale"),
  tuneGrid = expand.grid( # incorporated grid search for plausible range of hyperparamters
    alpha = c(0,1), # this tunes between ridge and lasso to enable elastic net when using `glmnet` as method
    lambda = seq(0.0001, 0.1, length = 10) #penalization/regularization parameter
  ),
  trControl = cv_control # Use consistent fold model
)
model2

model3 <- train(
  `work hours` ~ ., # `work hours` from all variables in the dataset
  gss_training, # training dataset
  na.action = na.pass, # added na.pass to enable skipping the nas in each variable
  method = "ranger", # random forest model
  preProcess = c("zv", "medianImpute","center","scale"),
  tuneGrid = expand.grid( # incorporated grid search for plausible range of hyperparamters
    mtry = c(23, 178, 267, 535), # sets default of rounded down square root of number of IVs, two intermediate values, and number of IVs; included based on nearZeroVariance(gss_training)
    splitrule = c("variance", "extratrees"), #"maxstat"), # used three common split rules for regression according to help("ranger") 
    min.node.size = 5 # 5 is the default for regression according to help("ranger")
  ),
  trControl=cv_control # Use consistent fold model
)
model3

model4 <- train(
  `work hours` ~., # `work hours` from all variables in the dataset
  gss_training, # training dataset
  na.action = na.pass, # added na.pass to enable skipping the nas in each variable
  method = "xgbTree", #eXtreme Gradient Boost = selected Tree as it is another random forest, non-linear model that is more sophisticated than RF; it takes a long time to run with these hyperparamters
  preProcess = c("zv", "medianImpute", "center", "scale"),
  tuneGrid = expand.grid( # incorporated grid search for plausible range of hyperparamters
    nrounds = 300, # number of boosting rounds
    eta = c(0.01, 0.3), # 0.3 is default for tree-based booster; selected those less than, as they will create more robust model 
    max_depth = c(6, 9), # 6 is default for tree depth; selected to account for either side of default
    min_child_weight = c(1, 5), # min observations needed in a leaf; the larger the more conservative and default is 1
    gamma = c(0, 1), # minimum loss reduction with more splits happening results in more conservative model
    colsample_bytree = c(.5, 1), # this is a subsample ration; default is 1 and needed to be values less than 1
    subsample = c(.5, 1) # subsampling the training instance; .5 is default larger prevents overfitting
  ),
  trControl = cv_control # Use consistent fold model
)
model4

## Examine k-fold CV results
summary(resamples(list(model1,model2, model3, model4))) # This summarized all my models
dotplot(resamples(list(model1,model2, model3, model4)), metric = "Rsquared")  # This provided a visualization of all my models

## Compare to holdout CV results
pred1 <- predict(model1, newdata = gss_holdout, na.action = na.pass) # This and the next three lines of code obtain the out of sample prediction
pred2 <- predict(model2, newdata = gss_holdout, na.action = na.pass)
pred3 <- predict(model3, newdata = gss_holdout, na.action = na.pass)
pred4 <- predict(model4, newdata = gss_holdout, na.action = na.pass)

# Publication
table1_tbl <- tibble( # create summary tibble
  algo = c(model1$method, model2$method, model3$method, model4$method), # obtain the model name from the results
  cv_rsq = str_remove(round(c(getTrainPerf(model1)$TrainRsquared, # obtain the R2 from the k-fold CV
             getTrainPerf(model2)$TrainRsquared, 
             getTrainPerf(model3)$TrainRsquared,
             getTrainPerf(model4)$TrainRsquared), 2), "^0"),
  ho_rsq = str_remove(round(c(postResample(pred1, gss_holdout$`work hours`)["Rsquared"], # obtain the R2 from the CV predict
             postResample(pred2, gss_holdout$`work hours`)["Rsquared"], 
             postResample(pred3, gss_holdout$`work hours`)["Rsquared"], 
             postResample(pred4, gss_holdout$`work hours`)["Rsquared"]), 2), "^0")
) %>% 
  write_csv(file = "../figs/table1.csv") # save the table

# Q1: How did your results change between models? Why do you think this happened, specifically?
# A2: The linear models (OLS and elastic net) performed poorer than the nonlinear tree-based models. The eXtreme Gradient Boosting (xgb) Tree performed slightly better in prediction than Random Forest (rf), but at a much greater computational cost.The more complex models have more hyperparamters to tune, allowing for increasing in-sample variance explained. These more flexible models are able to account for more fine-tuned non-linear relationships within the data. 
# Q2: How did you results change between k-fold CV and holdout CV? Why do you think this happened, specifically? 
# A2: The k-fold CV model R2 was slightly higher than the holdout CV R2 across each model. Each of the k-fold CV models is using in-sample data and is slightly more optimistic than can be the case with out of sample performance. Also by tuning hyperparameters, the fit is greater to explain the current dataset, yet may approach overfitting the training data, while being unable to accurately generalize to unseen out of sample data.  
# Q3: Among the four models, which would you choose for a real-life prediction problem, and why? Are there tradeoffs? Write up to a paragraph.
# A3: I would be inclined to use the rf (model3) model in a real-life prediction problem, given some of the computational constraints of my current setup (absent a supercomputer). While in the current case, with the hyperparameters I choose to tune the rf model, performed very well at a much lower cost than an xgb model, but it also had a slightly lower difference between k-fold CV and CV .0273 compared to xgb .0275. This suggests the rf method is a bit more generalizable to out of sample data. While it may be more difficult to interpret the variables that matter in nonlinear rf and xgb models, from a prediction standpoint, rf is a superior predictor to the linear methods at a lower cost. Should I be more interested in interpretability, I may recommend an elastic net model where I could explain the relative importance of predictors. The rf model also outperformed the xgb model for speed, even though while that model ran I was able to serve my kids dinner.  
