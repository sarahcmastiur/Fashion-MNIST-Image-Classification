
# Model 4: Random Forest
evaluate_rf <- function(sample_sizes = c(500, 1000, 2000),
                        data_path = "Data",
                        test_data,
                        total_train_rows = 60000) {
  
  results <- data.table()
  
  # Preprocess test data
  test_data$label <- as.factor(test_data$label)
  test_y <- test_data$label
  test_x <- test_data[, !("label"), with = FALSE]
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      
      training_dat$label <- as.factor(training_dat$label)
      
      cat("\nRunning Random Forest for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      set.seed(1031)
      formula <- as.formula("label ~ .")
      model <- randomForest(formula, data = training_dat, ntree = 1000)
      pred <- predict(model, test_x)
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "Random Forest",
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
      cat("Random Forest Done:", n, "x", i, " | Points =", round(Points, 4), "\n")
    }
  }
  
  return(results)
}


# Model 5: Tuned Random Forest (with cross-validation tuning)
evaluate_rf_tuned <- function(sample_sizes = c(500, 1000, 2000),
                              data_path = "Data",
                              test_data,
                              total_train_rows = 60000) {
  
  library(caret)
  results <- data.table()
  
  # Preprocess test data
  test_data$label <- as.factor(test_data$label)
  test_y <- test_data$label
  test_x <- test_data[, !("label"), with = FALSE]
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      
      training_dat$label <- as.factor(training_dat$label)
      
      cat("\nRunning Tuned Random Forest for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      set.seed(1031)
      
      # Tune mtry with cross-validation
      trControl <- trainControl(method = "cv", number = 5)
      tuneGrid <- expand.grid(mtry = 1:(ncol(training_dat)-1))
      
      cvModel <- train(label ~ ., data = training_dat, method = "rf",
                       trControl = trControl, tuneGrid = tuneGrid, ntree = 1000, verbose = FALSE)
      
      # Build final model with best mtry
      best_mtry <- cvModel$bestTune$mtry
      model <- randomForest(label ~ ., data = training_dat, ntree = 1000, mtry = best_mtry)
      pred <- predict(model, test_x)
      
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "Tuned Random Forest",
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
      cat("Tuned Random Forest Done:", n, "x", i, " | Points =", round(Points, 4), "\n")
    }
  }
  
  return(results)
}


# Model 6: Gradient Boosting Machine
evaluate_gbm <- function(sample_sizes = c(500, 1000, 2000),
                         data_path = "Data",
                         test_data,
                         total_train_rows = 60000) {
  
  results <- data.table()
  
  # Preprocess test data
  test_data$label <- as.factor(test_data$label)
  test_y <- test_data$label
  test_x <- test_data[, !("label"), with = FALSE]
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      
      training_dat$label <- as.factor(training_dat$label)
      
      cat("\nRunning GBM for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      set.seed(1031)
      formula <- as.formula("label ~ .")
      model <- gbm(formula, data = training_dat, distribution = "multinomial",
                   n.trees = 500, interaction.depth = 2, shrinkage = 0.01, verbose = FALSE)
      
      # Get predictions
      pred_probs <- predict(model, test_x, n.trees = 500, type = "response")
      pred <- colnames(pred_probs)[apply(pred_probs, 1, which.max)]
      pred <- as.factor(pred)
      
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "GBM",
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
      cat("GBM Done:", n, "x", i, " | Points =", round(Points, 4), "\n")
    }
  }
  
  return(results)
}


# Model 7: Tuned GBM (with parameter tuning)
evaluate_gbm_tuned <- function(sample_sizes = c(500, 1000, 2000),
                               data_path = "Data",
                               test_data,
                               total_train_rows = 60000) {
  
  results <- data.table()
  
  # Preprocess test data
  test_data$label <- as.factor(test_data$label)
  test_y <- test_data$label
  test_x <- test_data[, !("label"), with = FALSE]
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      
      training_dat$label <- as.factor(training_dat$label)
      
      cat("\nRunning Tuned GBM for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      set.seed(1031)
      
      # Simpler tuning grid to avoid computational issues
      trControl <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
      tuneGrid <- expand.grid(
        n.trees = 500,
        interaction.depth = c(2, 3),
        shrinkage = c(0.01, 0.05),
        n.minobsinnode = c(10)
      )
      
      # Use try() to handle potential errors
      cvModel <- tryCatch({
        train(label ~ ., data = training_dat, method = "gbm",
              trControl = trControl, tuneGrid = tuneGrid, verbose = FALSE)
      }, error = function(e) {
        cat("Tuning failed, using default parameters\n")
        NULL
      })
      
      # If tuning failed or succeeded, build final model
      if (!is.null(cvModel)) {
        best_depth <- cvModel$bestTune$interaction.depth
        best_shrinkage <- cvModel$bestTune$shrinkage
      } else {
        best_depth <- 2
        best_shrinkage <- 0.01
      }
      
      model <- gbm(label ~ ., data = training_dat, distribution = "multinomial",
                   n.trees = 500,
                   interaction.depth = best_depth,
                   shrinkage = best_shrinkage,
                   n.minobsinnode = 10,
                   verbose = FALSE)
      
      # Get predictions
      pred_probs <- predict(model, test_x, n.trees = 500, type = "response")
      pred <- colnames(pred_probs)[apply(pred_probs, 1, which.max)]
      pred <- as.factor(pred)
      
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "Tuned GBM",
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
      cat("Tuned GBM Done:", n, "x", i, " | Points =", round(Points, 4), "\n")
    }
  }
  
  return(results)
}



generate_scoreboard <- function(results_rf, results_rf_tuned, results_gbm, results_gbm_tuned) {
  
  # Function to squeeze results (one row per model-sample-iteration combination)
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
  
  # Squeeze each model's results
  rft <- squeeze(results_rf)[, model_type := "rf"]
  rftt <- squeeze(results_rf_tuned)[, model_type := "rf_tuned"]
  gbmt <- squeeze(results_gbm)[, model_type := "gbm"]
  gbmtt <- squeeze(results_gbm_tuned)[, model_type := "gbm_tuned"]
  
  # Add data column showing which dataset was used
  add_data <- function(dt) {
    dt[, Data := paste0("dat_", sample_size, "_", iteration)]
    dt
  }
  
  rft <- add_data(rft)
  rftt <- add_data(rftt)
  gbmt <- add_data(gbmt)
  gbmtt <- add_data(gbmtt)
  
  # Combine all results
  all_results <- rbindlist(list(rft, rftt, gbmt, gbmtt), use.names = TRUE, fill = TRUE)
  setorder(all_results, model_type, sample_size, iteration)
  
  # Create model names (rf1-rf9, rf_tuned1-rf_tuned9, etc.)
  all_results[, model_index := seq_len(.N), by = model_type]
  all_results[, model := paste0(model_type, model_index)]
  
  # Keep selected columns
  final_table <- all_results[, .(model, sample_size, Data, A, B, C, Points)]
  final_table[, Row := seq_len(.N)]
  setcolorder(final_table, c("Row", "model", "sample_size", "Data", "A", "B", "C", "Points"))
  
  return(final_table[])
}
