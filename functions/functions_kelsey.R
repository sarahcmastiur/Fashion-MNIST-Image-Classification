
# Models: Multinomial Logistic Regression, Ridge Regression, Lasso Regression



#  Model 1: Multinomial Logistic Regression
evaluate_multinom <- function(sample_sizes = c(500, 1000, 2000),
                              data_path = "Data",
                              test_data,
                              total_train_rows = 60000) {
  
  results <- data.table()
  
  test_data$label <- as.factor(test_data$label)
  test_y <- test_data$label
  test_x <- as.matrix(test_data[, !("label"), with = FALSE])
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      training_dat$label <- as.factor(training_dat$label)
      
      cat("\nRunning Multinomial Logistic Regression for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      set.seed(1031)
      model <- multinom(label ~ ., data = training_dat, trace = FALSE)
      pred <- predict(model, newdata = as.data.frame(test_x), type = "class")
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "Multinomial Logistic Regression",
        sample_size = n,
        iteration = i,
        runtime_sec = runtime,
        A = A, B = B, C = C, Points = Points,
        true_label = test_y,
        predicted_label = pred
      )
      
      results <- rbind(results, res)
      cat("Multinomial Logistic Regression Done:", n, "x", i, "| Points =", round(Points, 4), "\n")
    }
  }
  return(results)
}

#  Model 2: Ridge Regression
evaluate_ridge <- function(sample_sizes = c(500, 1000, 2000),
                           data_path = "Data",
                           test_data,
                           total_train_rows = 60000) {
  
  results <- data.table()
  test_data$label <- as.factor(test_data$label)
  test_y <- test_data$label
  test_x <- as.matrix(test_data[, !("label"), with = FALSE])
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      training_dat$label <- as.factor(training_dat$label)
      
      cat("\nRunning Ridge Regression for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      set.seed(1031)
      x <- as.matrix(training_dat[, !("label"), with = FALSE])
      y <- training_dat$label
      cv_fit <- cv.glmnet(x, y, family = "multinomial", alpha = 0)
      pred <- predict(cv_fit, newx = test_x, s = "lambda.min", type = "class")
      pred <- as.factor(pred)
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "Ridge Regression",
        sample_size = n,
        iteration = i,
        runtime_sec = runtime,
        A = A, B = B, C = C, Points = Points,
        true_label = test_y,
        predicted_label = pred
      )
      
      results <- rbind(results, res)
      cat("Ridge Regression Done:", n, "x", i, "| Points =", round(Points, 4), "\n")
    }
  }
  return(results)
}

#  Model 3: Lasso Regression
evaluate_lasso <- function(sample_sizes = c(500, 1000, 2000),
                           data_path = "Data",
                           test_data,
                           total_train_rows = 60000) {
  
  results <- data.table()
  test_data$label <- as.factor(test_data$label)
  test_y <- test_data$label
  test_x <- as.matrix(test_data[, !("label"), with = FALSE])
  
  for (n in sample_sizes) {
    for (i in 1:3) {
      file_path <- file.path(data_path, paste0("dat_", n, "_", i, ".csv"))
      training_dat <- fread(file_path)
      training_dat$label <- as.factor(training_dat$label)
      
      cat("\nRunning Lasso Regression for sample", n, "iteration", i, "...\n")
      
      start_time <- Sys.time()
      set.seed(1031)
      x <- as.matrix(training_dat[, !("label"), with = FALSE])
      y <- training_dat$label
      cv_fit <- cv.glmnet(x, y, family = "multinomial", alpha = 1)
      pred <- predict(cv_fit, newx = test_x, s = "lambda.min", type = "class")
      pred <- as.factor(pred)
      end_time <- Sys.time()
      
      runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
      A <- n / total_train_rows
      B <- min(1, runtime / 60)
      C <- mean(pred != test_y)
      Points <- 0.15 * A + 0.1 * B + 0.75 * C
      
      res <- data.table(
        model = "Lasso Regression",
        sample_size = n,
        iteration = i,
        runtime_sec = runtime,
        A = A, B = B, C = C, Points = Points,
        true_label = test_y,
        predicted_label = pred
      )
      
      results <- rbind(results, res)
      cat("Lasso Regression Done:", n, "x", i, "| Points =", round(Points, 4), "\n")
    }
  }
  return(results)
}


#  SCOREBOARD FUNCTION
generate_scoreboard <- function(results_multinom, results_ridge, results_lasso) {
  
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
  
  multi_t <- squeeze(results_multinom)[, model_type := "multinom"]
  ridge_t <- squeeze(results_ridge)[, model_type := "ridge"]
  lasso_t <- squeeze(results_lasso)[, model_type := "lasso"]
  
  add_data <- function(dt) {
    dt[, Data := paste0("dat_", sample_size, "_", iteration)]
    dt
  }
  multi_t <- add_data(multi_t)
  ridge_t <- add_data(ridge_t)
  lasso_t <- add_data(lasso_t)
  
  all_results <- rbindlist(list(multi_t, ridge_t, lasso_t), use.names = TRUE, fill = TRUE)
  setorder(all_results, model_type, sample_size, iteration)
  all_results[, model_index := seq_len(.N), by = model_type]
  all_results[, model := paste0(model_type, model_index)]
  
  final_table <- all_results[, .(model, sample_size, Data, A, B, C, Points)]
  final_table[, Row := seq_len(.N)]
  setcolorder(final_table, c("Row", "model", "sample_size", "Data", "A", "B", "C", "Points"))
  
  return(final_table[])
}

