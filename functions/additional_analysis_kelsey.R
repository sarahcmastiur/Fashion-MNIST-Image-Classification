#Kelsey additional analysis question 1 

visual_similarity_analysis <- function(file_path) {
  
  library(data.table)
  library(ggplot2)
  library(Rtsne)
  library(dplyr)
  library(reshape2)
  library(viridis)
  
  #load and prepare data
  train <- fread(file_path)
  colnames(train)[1] <- "label"
  
  features <- as.matrix(train[, -1])
  labels <- as.factor(train$label)
  
  #Class distribution
  label_counts <- as.data.table(table(labels))
  colnames(label_counts) <- c("Category", "Count")
  
  p1 <- ggplot(label_counts, aes(x = reorder(Category, -Count), y = Count, fill = Category)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_fill_viridis_d() +
    labs(title = "Distribution of Clothing Categories",
         x = "Category", y = "Number of Images") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
  print(p1)
  
  #PCA Visualization
  set.seed(42)
  pca_result <- prcomp(features, scale. = TRUE)
  pca_df <- data.frame(pca_result$x[, 1:2], Label = labels)
  
  p2 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Label)) +
    geom_point(alpha = 0.5, size = 1) +
    scale_color_viridis_d() +
    labs(title = "PCA Projection of Clothing Categories",
         subtitle = "Each point represents a 49-feature image projected into 2D",
         x = "Principal Component 1",
         y = "Principal Component 2") +
    theme_minimal(base_size = 12)
  print(p2)
  
  # t-SNE Visualization
  subset_index <- sample(1:nrow(features), 2000)
  tsne_input <- features[subset_index, ]
  tsne_labels <- labels[subset_index]
  
  tsne_result <- Rtsne(tsne_input, perplexity = 30, verbose = TRUE, check_duplicates = FALSE)
  tsne_df <- data.frame(X1 = tsne_result$Y[, 1],
                        X2 = tsne_result$Y[, 2],
                        Label = tsne_labels)
  
  p3 <- ggplot(tsne_df, aes(x = X1, y = X2, color = Label)) +
    geom_point(alpha = 0.7, size = 1.5) +
    scale_color_viridis_d() +
    labs(title = "t-SNE Visualization of Clothing Category Similarities",
         subtitle = "Overlapping clusters suggest visually similar categories",
         x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
    theme_minimal(base_size = 12)
  print(p3)
  
  #Class Centroid Similarity
  centroids <- train %>%
    group_by(label) %>%
    summarise(across(everything(), mean))
  
  dist_matrix <- as.matrix(dist(centroids[, -1]))
  rownames(dist_matrix) <- centroids$label
  colnames(dist_matrix) <- centroids$label
  
  dist_df <- melt(dist_matrix)
  colnames(dist_df) <- c("Class1", "Class2", "Distance")
  
  p4 <- ggplot(dist_df, aes(x = Class1, y = Class2, fill = Distance)) +
    geom_tile(color = "white") +
    scale_fill_viridis(option = "plasma", direction = -1) +
    labs(title = "Visual Similarity Between Clothing Categories (Euclidean Distance)",
         subtitle = "Darker = more visually similar",
         x = "Category", y = "Category") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(p4)
  
  #return all plots as a list
  invisible(list(Distribution = p1, PCA = p2, tSNE = p3, Heatmap = p4))
}
