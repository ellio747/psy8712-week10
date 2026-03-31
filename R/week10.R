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
  select(where(function(x) mean(is.na(x)) < 0.75))

# Visualization
ggplot(gss_tbl, aes(x = `work hours`)) + 
  geom_histogram()

# Analysis
model <- train(
  `work hours` ~ .,
  data = gss_tbl,
  method = "rf", 
  preProcess = "medianImpute"
)

# Publication
table1_tbl <- tibble(
  algo = ,
  cv_rsq = ,
  ho_rsq = 
)