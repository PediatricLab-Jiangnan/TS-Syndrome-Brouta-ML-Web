# Load packages for forest plot
library(forestplot)
library(dplyr)
library(broom)

# Extract data from model
forest_data <- model %>%
  tidy(conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term != "(Intercept)") %>%  
  filter(p.value < 0.05)

# Prepare forest plot data matrix (add NA row for title)
forest_data_matrix <- rbind(
  rep(NA, 3),
  cbind(
    forest_data$estimate,    
    forest_data$conf.low,    
    forest_data$conf.high    
  )
)

# Create text matrix
tabletext <- cbind(
  c("Variable", forest_data$term),  
  c("OR (95% CI)", 
    sprintf("%.2f (%.2f-%.2f)", 
            forest_data$estimate, 
            forest_data$conf.low, 
            forest_data$conf.high)),  
  c("P-value",
    sprintf("%.3f", forest_data$p.value))  
)

# Create forest plot with enhanced styling
forestplot(
  tabletext,  
  forest_data_matrix,  
  new_page = TRUE,
  is.summary = c(TRUE, rep(FALSE, nrow(forest_data))),  
  title = "Forest Plot of Significant Variables (p < 0.05)",
  
  # Color scheme
  col = fpColors(
    box = "#4682B4",        # Steel blue for boxes
    lines = "#4F94CD",      # Coordinated line color
    zero = "#B0C4DE",       # Light blue reference line
    summary = "#2F4F4F"     # Dark slate gray for header
  ),
  
  # Basic settings
  xlab = "Odds Ratio",
  zero = 1,
  boxsize = 0.25,
  lineheight = "auto",
  colgap = unit(8, "mm"),
  
  # CI display settings
  ci.vertices = TRUE,
  ci.vertices.height = 0.1,
  
  # X-axis ticks
  xticks = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10),
  
  # Text styling
  txt_gp = fpTxtGp(
    label = gpar(cex = 0.9, fontfamily = "sans", fontface = "bold"),
    ticks = gpar(cex = 0.8),
    xlab = gpar(cex = 1, fontface = "bold"),
    title = gpar(cex = 1.2, fontface = "bold")
  ),
  
  # Grid and border settings
  box = TRUE,
  boxBorder = "#2F4F4F",
  grid = structure(TRUE, gp = gpar(lty = 2, col = "#E6E6E6")),
  
  # Horizontal lines
  hrzl_lines = list(
    gpar(lwd = 1.5, col = "#2F4F4F"),  # Top border
    gpar(lwd = 1.5, col = "#2F4F4F"),  # Below title
    rep(list(gpar(lwd = 0.5, col = "#B0C4DE")), nrow(forest_data)-1),  # Between data rows
    gpar(lwd = 1.5, col = "#2F4F4F")   # Bottom border
  ) %>% unlist(recursive = FALSE),
  
  # Other settings
  margin = unit(c(2, 2, 2, 2), "cm"),
  xlog = TRUE,
  graphwidth = unit(10, "cm")
)

# Save plot to PDF
pdf("significant_forest_plot.pdf", width = 10, height = max(7, nrow(forest_data) * 0.5))
# Repeat forest plot code here
dev.off()
