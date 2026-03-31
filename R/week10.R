# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(haven) # Read SPSS files

# Data Import and Cleaning
gss_tbl <- read_sav("../data/GSS2016.sav")

# Visualization

# Analysis

# Publication
