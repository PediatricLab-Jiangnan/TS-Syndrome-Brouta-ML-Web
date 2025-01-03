# Load required packages
library(gtsummary)    # For creating summary tables
library(tidyverse)    # For data manipulation and visualization
library(flextable)    # For creating flexible tables

# Set working directory
setwd("Your path")

# Read the CSV file containing patient data
data <- read.csv("Your own information")

# Data preprocessing:
# 1. Convert sex and Group to factors with meaningful labels
# 2. Convert all other columns to numeric type
data <- data %>%
  mutate(
    # Convert sex to factor (0=Male, 1=Female)
    sex = factor(sex, levels = c(0, 1), labels = c("Male", "Female")),
    # Convert Group to factor (0=Control, 1=TS)
    Group = factor(Group, levels = c(0, 1), labels = c("Control", "TS disorders")),
    # Convert all other columns to numeric
    across(!c(sex, Group), as.numeric)
  )

# Create baseline characteristics table
tbl <- data %>%
  tbl_summary(
    by = Group,    # Stratify by Group
    statistic = list(
      # Format continuous variables as mean ± SD
      all_continuous() ~ "{mean} ± {sd}",
      # Format categorical variables as n (%)
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = list(all_continuous() ~ 2),    # Set decimal places for continuous variables
    missing = "no" 
  ) %>%
  add_p(
    test = list(
      all_continuous() ~ "t.test",         # Use t-test for continuous variables
      all_categorical() ~ "chisq.test"     # Use chi-square test for categorical variables
    ),
    pvalue_fun = ~ style_pvalue(.x, digits = 3)    # Format p-values with 3 decimal places
  ) %>%
  bold_labels()    # Make variable labels bold

# Display the table
tbl

# Export table to Word document
tbl %>%
  as_flex_table() %>%
  save_as_docx(path = "baseline_characteristics.docx")
