# Set working directory  
setwd("Your path")  

# Load required packages
library(gtsummary)
library(tidyverse)
library(flextable)

# Read data
data <- read.csv("Your own clinical information")

# Select all variables except Group for multivariate logistic regression
predictors <- names(data)[!names(data) %in% "Group"]
formula <- as.formula(paste("Group ~", paste(predictors, collapse = " + ")))

# Perform multivariate logistic regression
model <- glm(formula, 
             family = binomial(link = "logit"), 
             data = data)

# Generate results table using gtsummary, showing all variables
tbl <- tbl_regression(model, 
                      exponentiate = TRUE,
                      pvalue_fun = ~style_pvalue(.x, digits = 3)) %>%
  add_global_p() %>%
  bold_p(t = 0.05) %>%  # Bold significant results
  bold_labels()

# Convert to flextable object
ft <- tbl %>% 
  as_flex_table() %>%
  fontsize(size = 10, part = "all") %>%
  theme_box()

# Save as Word document
save_as_docx(ft, path = "multi-logistic_regression_results.docx")


