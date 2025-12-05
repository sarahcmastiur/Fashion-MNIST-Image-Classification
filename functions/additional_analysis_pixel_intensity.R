# Define the pixel intensity analysis function inline
pixel_intensity_analysis <- function(file_path) {
  
  library(data.table)
  library(ggplot2)
  library(reshape2)
  library(viridis)
  
  # Load and prepare data
  train <- fread(file_path)
  
  # Identify pixel columns (all except first column which is label)
  pixel_cols <- colnames(train)[-1]
  label_col <- colnames(train)[1]
  
  # Rename label column for consistency
  setnames(train, label_col, "label")
  train[, label := as.factor(label)]
  
  # Plot 1: Average Pixel Intensity by Category
  
  # Calculate overall mean intensity per category
  overall_intensity <- train[, .(avg_intensity = rowMeans(.SD, na.rm = TRUE)), 
                             by = label, .SDcols = pixel_cols]
  overall_intensity <- overall_intensity[, .(avg_intensity = mean(avg_intensity)), by = label]
  
  p1 <- ggplot(overall_intensity, aes(x = reorder(label, -avg_intensity), y = avg_intensity, fill = label)) +
    geom_bar(stat = "identity") +
    scale_fill_viridis_d() +
    labs(title = "Average Pixel Intensity by Clothing Category",
         subtitle = "Darker items show higher grayscale intensity (0-255 scale)",
         x = "Category", y = "Average Pixel Intensity") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1))
  print(p1)
  

  # Plot 2: Pixel Intensity Heatmap by Category
  
  # Calculate mean intensity for each pixel by label
  intensity_by_label <- train[, lapply(.SD, mean), by = label, .SDcols = pixel_cols]
  
  # Convert pixel columns to long format with row/col coordinates
  intensity_melted <- melt(intensity_by_label, id.vars = "label",
                           variable.name = "pixel", value.name = "intensity")
  intensity_melted <- as.data.table(intensity_melted)
  
  # Extract pixel number and convert to 7x7 grid coordinates
  intensity_melted[, pixel_num := as.numeric(gsub("pixel", "", pixel))]
  intensity_melted[, row := (pixel_num - 1) %/% 7 + 1]
  intensity_melted[, col := (pixel_num - 1) %% 7 + 1]
  
  # Get top 4 categories by intensity
  top_categories <- head(overall_intensity[order(-avg_intensity)], 4)$label
  
  heatmap_data <- intensity_melted[label %in% top_categories]
  
  p2 <- ggplot(heatmap_data, aes(x = col, y = row, fill = intensity)) +
    geom_tile() +
    facet_wrap(~label, nrow = 2) +
    scale_y_reverse() +
    scale_fill_gradient(low = "white", high = "black", name = "Intensity") +
    labs(title = "Pixel Intensity Heatmaps: Top 4 Most Intense Categories",
         subtitle = "Darker areas = higher pixel intensity values",
         x = "Column", y = "Row") +
    theme_minimal(base_size = 10) +
    theme(aspect.ratio = 1)
  print(p2)
  

  # Plot 3: Feature Variance (Consistency) by Category
  
  # Calculate mean variance per category
  consistency_scores <- train[, .(
    mean_variance = mean(apply(.SD, 1, var, na.rm = TRUE))
  ), by = label, .SDcols = pixel_cols]
  
  p3 <- ggplot(consistency_scores, aes(x = reorder(label, -mean_variance), y = mean_variance, fill = label)) +
    geom_bar(stat = "identity") +
    scale_fill_viridis_d() +
    labs(title = "Feature Variance (Pixel Consistency) by Category",
         subtitle = "Higher variance = more variation in pixel patterns within category",
         x = "Category", y = "Mean Pixel Variance") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1))
  print(p3)
  

  # Plot 4: Intensity Distribution (Box Plot) Comparison

  
  # Calculate average pixel value for each image
  image_intensities <- train[, .(
    avg_intensity = rowMeans(.SD, na.rm = TRUE)
  ), by = label, .SDcols = pixel_cols]
  
  p4 <- ggplot(image_intensities, aes(x = label, y = avg_intensity, fill = label)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.2, size = 0.5) +
    scale_fill_viridis_d() +
    labs(title = "Distribution of Average Pixel Intensity Across Items",
         subtitle = "Each point = one image; shows variation within each category",
         x = "Category", y = "Average Pixel Intensity per Image") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1))
  print(p4)
  

  # Plot 5: Comparison Scatter - Intensity vs Variance
  
  # Merge intensity and variance data
  intensity_variance_compare <- merge(
    overall_intensity,
    consistency_scores,
    by = "label"
  )
  
  p5 <- ggplot(intensity_variance_compare, 
               aes(x = avg_intensity, y = mean_variance, label = label, color = label)) +
    geom_point(size = 5, alpha = 0.7) +
    geom_text(hjust = 0.5, vjust = -1, size = 3) +
    scale_color_viridis_d() +
    labs(title = "Intensity vs. Variance: Product Characterization",
         subtitle = "Top-left: Dark & consistent | Top-right: Dark & varied | Bottom-left: Light & consistent",
         x = "Average Pixel Intensity",
         y = "Mean Pixel Variance") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
  print(p5)

  # Return Summary Statistics
  
  summary_stats <- list(
    overall_intensity = overall_intensity,
    consistency_scores = consistency_scores,
    intensity_variance = intensity_variance_compare
  )
  
  invisible(summary_stats)
}