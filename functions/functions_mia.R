
# model 8: SVM
evaluate_svm <- function(sample_sizes = c(500, 1000, 2000),
                         data_path = "Data",
                         test_data,
                         total_train_rows = 60000) {
  
  results <- data.table()
  
  # pre_process datasets (standardize datasets) before training neural networks 
  test_data$label <- as.factor(test_data$label)
  test_x <- test_data[, !("label"), with = FALSE]
  # normalize to [0,1]
  test_x <- test_x[, lapply(.SD, function(x) as.numeric(x) / 255)]
  test_y <- as.factor(test_data$label)
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      
      train_y <- as.factor(training_dat$label)
      #normalize to [0,1]
      train_x <- training_dat[, (names(training_dat)[names(training_dat) != "label"]) :=
                     lapply(.SD, function(x) as.numeric(x) / 255),
                   .SDcols = !("label")]
      
      train_x <- train_x[, names(test_x), with = FALSE]
      
      cat("\nRunning SVM for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      model <- svm(train_x, train_y, kernel = "linear", cost = 1, scale = FALSE) # model parameters
      pred <- predict(model, test_x)
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "svm",
        sample_size = n,
        iteration = i,
        runtime_sec = runtime,
        A = A,
        B = B,
        C = C,
        Points = Points,
        true_label = test_y,
        predicted_label = pred
      )
      
      results <- rbind(results, res)
      cat("SVM Done:", n, "x", i, " | Points =", round(Points, 4), "\n")
    }
  }
  
  return(results)
}


# model 9: Neural Networks
evaluate_neural_networks <- function(sample_sizes = c(500, 1000, 2000),
                         data_path = "Data",
                         test_data,
                         total_train_rows = 60000) {
  
  results <- data.table()
  
  # pre_process datasets (standardize datasets) before training neural networks 
  test_data$label <- as.factor(test_data$label)
  test_x <- test_data[, !("label"), with = FALSE]
  # normalize to [0,1]
  test_x <- test_x[, lapply(.SD, function(x) as.numeric(x) / 255)]
  test_y <- as.factor(test_data$label)
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      
      training_dat$label <- as.factor(training_dat$label)
      #normalize to [0,1]
      training_dat[, (names(training_dat)[names(training_dat) != "label"]) :=
                     lapply(.SD, function(x) as.numeric(x) / 255),
                   .SDcols = !("label")]
      
      cat("\nRunning Neural Networks for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      model <- nnet(label ~ ., data =training_dat, size = 10, maxit = 200, trace = FALSE) # model parameters
      pred <- predict(model, test_x,type = "class")
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "Neural Networks",
        sample_size = n,
        iteration = i,
        runtime_sec = runtime,
        A = A,
        B = B,
        C = C,
        Points = Points,
        true_label = test_y,
        predicted_label = pred
      )
      
      results <- rbind(results, res)
      cat("Neural Networks Done:", n, "x", i, " | Points =", round(Points, 4), "\n")
    }
  }
  
  return(results)
}



# model 10: K-Nearest Neighbors
evaluate_knn <- function(sample_sizes = c(500, 1000, 2000),
                                     data_path = "Data",
                                     test_data,
                                     total_train_rows = 60000) {
  
  results <- data.table()
  
  # pre_process datasets (standardize datasets) before training neural networks 
  test_data$label <- as.factor(test_data$label)
  test_x <- test_data[, !("label"), with = FALSE]
  # normalize to [0,1]
  test_x <- test_x[, lapply(.SD, function(x) as.numeric(x) / 255)]
  test_y <- as.factor(test_data$label)
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      
      train_y <- as.factor(training_dat$label)
      #normalize to [0,1]
      train_x <- training_dat[, (names(training_dat)[names(training_dat) != "label"]) :=
                     lapply(.SD, function(x) as.numeric(x) / 255),
                   .SDcols = !("label")]
      train_x <- train_x[, names(test_x), with = FALSE]
      cat("\nRunning K-Nearest Neighbors for sample", n, "iteration", i, "...\n")
     
      start_time <- Sys.time()
      pred <- knn(train = train_x, test = test_x, cl = train_y, k = 5)# model parameters
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "K-Nearest Neighbors",
        sample_size = n,
        iteration = i,
        runtime_sec = runtime,
        A = A,
        B = B,
        C = C,
        Points = Points,
        true_label = test_y,
        predicted_label = pred
      )
      
      results <- rbind(results, res)
      cat("K-Nearest Neighbors Done:", n, "x", i, " | Points =", round(Points, 4), "\n")
    }
  }
  
  return(results)
}




# scoreboard
generate_scoreboard <- function(results_svm, results_knn, results_nn) {
  library(data.table)
  
  # for each model, only choose a,b,c, and points one time
  squeeze <- function(dt) {
    dt <- as.data.table(dt)
    dt[, .(
      sample_size = unique(sample_size),
      A = round(unique(A), 4),
      B = round(unique(B), 4),
      C = round(unique(C), 4),
      Points = round(unique(Points), 4),
      iteration = unique(iteration)
    ), by = .(model, sample_size, iteration)]
  }
  
  # merge squeeze results to model results
  svmt <- squeeze(results_svm)[, model_type := "svm"]
  knnt <- squeeze(results_knn)[, model_type := "knn"]
  nnt  <- squeeze(results_nn )[, model_type := "nn" ]
  
  # merge corresponding sample data
  add_data <- function(dt) {
    dt[, Data := paste0("dat_", sample_size, "_", iteration)]
    dt
  }
  svmt <- add_data(svmt)
  knnt <- add_data(knnt)
  nnt  <- add_data(nnt)
  
  # merge three model results 
  all_results <- rbindlist(list(svmt, knnt, nnt), use.names = TRUE, fill = TRUE)
  setorder(all_results, model_type, sample_size, iteration)
  
  # name each model to be svm1..svm9 / knn1..knn9 / nn1..nn9 for future differentiation and comparison
  all_results[, model_index := seq_len(.N), by = model_type]
  all_results[, model := paste0(model_type, model_index)]
  
  # keep selected columns 
  final_table <- all_results[, .(model, sample_size, Data, A, B, C, Points)]
  final_table[, Row := seq_len(.N)]
  setcolorder(final_table, c("Row", "model", "sample_size", "Data", "A", "B", "C", "Points"))
  
 
  return(final_table[])
}



# Predictive Accuracy by Product
calculate_product_accuracy <- function(data, products) {
  library(data.table)
  dt <- as.data.table(data)
  
  # check if selected variables exists ("true_label", "predicted_label")
  stopifnot(all(c("true_label", "predicted_label") %in% names(dt)))
  dt <- dt[true_label %in% products]
  
  # group by product, calculate accuracy rate = correct_count / total_count 
  agg <- dt[, .(
    total_count   = .N,
    correct_count = sum(true_label == predicted_label),
    accuracy      = paste0(round(100 * mean(true_label==predicted_label),4),"%"))   
    , by = .(product = true_label)]
  
  #create target data table, merge full and agg 
  product_table <- data.table(product = products)
  full <- product_table[agg, on = "product"]
  
  # if certain product's sum_count = 0, set value to be 0 and NA
  full[is.na(total_count), `:=`(total_count = 0L, correct_count = 0L, accuracy = NA_real_)]
  
  # display data table by products
  full[, product := factor(product, levels = products)]
  setorder(full, product)
  
  return(full[])
}



#calculate_misclassification
calculate_misclassification <- function(data, products = NULL) {
  library(data.table)
  dt <- as.data.table(data)
  
  stopifnot(all(c("true_label", "predicted_label") %in% names(dt)))
  
  #product-specific filtering 
  if (!is.null(products)) {
    dt <- dt[true_label %in% products]
  }
  
  # filter prediction errors 
  dt_mis <- dt[true_label != predicted_label]
  
  # summarize misclassification details by products and by false-classified product pairs 
  result <- dt_mis[, .(
    misclassified_count = .N
  ), by = .(true_label, predicted_label)]
  
  # calculate total misclassifications number by specific product 
  totals <- result[, .(total_mis = sum(misclassified_count)), by = true_label]
  
  result <- merge(result, totals, by = "true_label", all.x = TRUE)
  
  # calculate misclassification percentage
  result[, percentage := paste0(round(100 * misclassified_count / total_mis, 4), "%")]
  # select needed columns 
  result<-result[,.(true_label,predicted_label,misclassified_count,percentage)]
  
  # descending order of misclassification percentage 
  setorder(result, -misclassified_count)
  
  return(result[])}



library(ggplot2)
library(reshape2)
library(patchwork)
library(data.table)
library(DT)


# convert to matrix using grayscale data
make_matrix <- function(row) {
  matrix(
    as.numeric(as.vector(as.matrix(row))),
    nrow = 7, ncol = 7, byrow = TRUE
  )
}


# black and white heatmap 
plot_heatmap <- function(mat, title) {
  df <- reshape2::melt(mat)
  ggplot(df, aes(Var2, Var1, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black", limits = c(0, 255)) +
    ggtitle(title) +
    theme_minimal() +
    coord_fixed()
}
