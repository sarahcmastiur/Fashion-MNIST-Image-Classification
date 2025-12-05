setwd("/Users/kelsey/Desktop/APAN5902GP5 copy") #change it every time 
library(data.table)


dat_train <- fread("Data/MNIST-fashion training set-49.csv")
head(dat_train)
nrow(dat_train)

sample_sizes <- c(500, 1000, 2000)
n_iter <- 3

for (size in sample_sizes) {
  for (i in 1:n_iter) {
    
    set.seed(100 + size + i)  

        
    # sample() function 
    sampled_rows <- dat_train[sample(1:.N, size, replace = TRUE)]# with replacement 
        
 
    obj_name <- sprintf("dat_%d_%d", size, i)
    assign(obj_name, sampled_rows)
    
    file_name <- sprintf("dat_%d_%d.csv", size, i)
    fwrite(sampled_rows, file_name)
    
    cat(sprintf("saved %s (%d rows)\n", file_name, size))
  }
}

cat("9 files are successfully created. \n")



