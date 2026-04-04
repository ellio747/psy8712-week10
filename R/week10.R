# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(haven) # Read SPSS files
library(tidyverse) # Data analysis tools
library(caret) # Machine learning

# Data Import and Cleaning
gss_tbl <- read_sav("../data/GSS2016.sav", user_na = TRUE) %>% 
  zap_missing() %>% 
  filter(!is.na(mosthrs)) %>% 
  rename(`work hours` = mosthrs) %>% 
  select(-c(hrs1, hrs2)) %>% 
  select(which(colMeans(is.na(.)) < 0.75)) %>% 
  mutate(across(where(is.character), as.factor)) %>% 
  mutate(`work hours` = as.numeric(`work hours`)) %>% 
  as_tibble()

# Visualization
ggplot(gss_tbl, aes(x = `work hours`)) + 
  geom_histogram()

# Analysis

## Define training and test sets
set.seed(123)
holdout_indices <- createDataPartition(gss_tbl$`work hours`, p = .75, list=F)
gss_training <- gss_tbl[-holdout_indices,]
gss_holdout <- gss_tbl[holdout_indices,]

## Define consistent folds 
cv_control <- trainControl(
  method="cv", 
  number=10, 
  verboseIter = T
)

## Decide on hyperparameters to look at
tune_grid <- expand.grid(
  alpha = c(0,1),
  lambda = seq(0.0001, 0.1, length = 10) # ,
  # mtry = c(2, 3, 4, 5, 10, 20)
)

## Which Models Running
mods <- c("lm", "glmnet", "ranger", "xgbLinear")

## Run k-fold testing hyperparameters on training set

# Four model types: #modelLookup(models)
# OLS regression = "lm" no tuning parameters
# elastic net = "glmnet" alpha and lambda (tuning parameters)
# random forest = "ranger" mtry, splitrule, min.nod.size
# “eXtreme Gradient Boosting” = xgbLinear alpha and lambda nrounds

model1 <- train(
  `work hours` ~ .,
  gss_training,
  na.action = na.pass,
  method = "lm",
  preProcess = c("medianImpute","center","scale"),
  trControl=cv_control
  )
model1

model2 <- train(
  `work hours` ~ .,
  gss_training,
  na.action = na.pass,
  method = "glmnet",
  preProcess = c("medianImpute","center","scale"),
  tuneGrid = tune_grid,
  trControl = cv_control
)
model2

model3 <- train(
  `work hours` ~ .,
  gss_training,
  na.action = na.pass,
  method = "ranger",
  preProcess = c("medianImpute","center","scale"),
  tuneGrid = tune_grid,
  trControl=cv_control
)
model3

model4 <- train(
  `work hours` ~ .,
  gss_training,
  na.action = na.pass,
  method = "xgbLinear",
  preProcess = c("medianImpute","center","scale"),
  tuneGrid = tune_grid,
  trControl=cv_control
)
model3

## Examine k-fold CV results

## Compare to holdout CV results

# Publication
table1_tbl <- tibble(
  algo = ,
  cv_rsq = ,
  ho_rsq = 
)