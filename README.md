# APAN5902GP5: Fashion-MNIST Image Classification Project

A comprehensive machine learning project comparing 10 different classification algorithms on the Fashion-MNIST dataset. This project evaluates model performance across different training sample sizes (500, 1000, 2000 rows) using a custom scoring system that balances accuracy, execution time, and computational efficiency.

## Project Overview

**Authors:** Kelsey Chen, Sarah Mastiur, Mia Xue  
**Course:** APAN5902 - Data Science Consulting  
**Date:** November 2025

### Objective

Build and compare machine learning models for recognizing clothing items from the Fashion-MNIST dataset. Each image represents one of 10 clothing categories, and models must predict the correct category based on 49 pixel values (7×7 downsampled images).

### Models Evaluated

1. **Multinomial Logistic Regression** (Kelsey) - Baseline linear classifier
2. **Ridge Regression** (Kelsey) - L2-regularized for feature correlation handling
3. **Lasso Regression** (Kelsey) - L1-regularized for feature selection
4. **Random Forest** (Sarah) - 1000 trees baseline ensemble
5. **Random Forest Tuned** (Sarah) - Optimized mtry via 5-fold cross-validation
6. **Gradient Boosted Machine** (Sarah) - GBM baseline with 500 trees
7. **Gradient Boosted Machine Tuned** (Sarah) - Optimized hyperparameters via 3-fold CV
8. **Support Vector Machine** (Mia) - Linear kernel with cost = 1
9. **Neural Network** (Mia) - 10 hidden neurons, 200 iterations
10. **K-Nearest Neighbors** (Mia) - k=5, Euclidean distance

## Repository Structure

```
.
├── README.md                          # This file
├── .gitignore                         # Git ignore patterns
├── requirements.txt                   # R dependencies and versions
│
├── data/
│   ├── raw/
│   │   ├── MNIST-fashion_training_set-49.csv      # Full training set (60,000 samples)
│   │   ├── MNIST-fashion_testing_set-49.csv       # Test set (10,000 samples)
│   │   └── README_DATA.md                         # Data documentation
│   └── processed/
│       ├── dat_500_1.csv through dat_500_3.csv    # 500-row training samples
│       ├── dat_1000_1.csv through dat_1000_3.csv  # 1000-row training samples
│       └── dat_2000_1.csv through dat_2000_3.csv  # 2000-row training samples
│
├── src/
│   ├── functions_kelsey.R             # Multinomial, Ridge, Lasso evaluation
│   ├── functions_sarah.R              # Random Forest and GBM evaluation
│   ├── functions_mia.R                # SVM, Neural Network, KNN evaluation
│   ├── additional_analysis_pixel_intensity.R   # Pixel intensity analysis
│   └── additional_analysis_kelsey.R            # Additional analyses
│
├── analysis/
│   ├── 01_main_analysis.Rmd           # Main analysis and model evaluation
│   └── 02_additional_analyses.Rmd     # Supplementary analyses
│
├── results/
│   ├── scoreboards/
│   │   ├── scoreboard_regression.csv  # Kelsey's model results
│   │   ├── scoreboard_sarah.csv       # Sarah's model results
│   │   ├── scoreboard_mia.csv         # Mia's model results
│   │   ├── scoreboard_all.csv         # Combined results (all models, all runs)
│   │   └── scoreboard_final.csv       # Final aggregated results
│   ├── models/
│   │   ├── results_multinom.rds
│   │   ├── results_ridge.rds
│   │   ├── results_lasso.rds
│   │   ├── results_rf.rds
│   │   ├── results_rf_tuned.rds
│   │   ├── results_gbm.rds
│   │   ├── results_gbm_tuned.rds
│   │   ├── results_svm.rds
│   │   ├── results_neural_networks.rds
│   │   └── results_k_nearest_neighbors.rds
│   └── reports/
│       ├── Image_Processing_Part_1.html
│       └── Additional_Analyses.html
│
├── scripts/
│   └── sampling.R                     # Data sampling utilities
│
└── docs/
    ├── METHODOLOGY.md                 # Detailed methodology and evaluation metrics
    └── SETUP.md                       # Environment setup instructions
```

## Key Findings

**Best Overall Models:**
- **Random Forest** and **SVM** achieved the lowest total points (~0.16), indicating the best balance of accuracy and efficiency
- Strong performance across all sample sizes

**Model Performance Tiers:**

| Tier | Models | Points | Characteristics |
|------|--------|--------|-----------------|
| Tier 1 | Random Forest, SVM | ~0.16 | High accuracy, moderate runtime |
| Tier 2 | GBM, Neural Network | ~0.17-0.18 | Good accuracy, longer runtime |
| Tier 3 | Ridge, Lasso, Multinomial LR | ~0.18-0.19 | Fast training, lower accuracy |
| Tier 4 | K-Nearest Neighbors | ~0.19-0.20 | Slow predictions, moderate accuracy |

**Impact of Sample Size:**
- Larger sample sizes (2000 rows) improved accuracy across all models
- Runtime increased with sample size, but accuracy gains justified the tradeoff
- Models showed consistent performance patterns across different sample sizes

## Scoring Methodology

Each model is evaluated on three metrics, scaled and aggregated:

- **A:** Accuracy metric (lower misclassification = lower score, better)
- **B:** Runtime metric (faster execution = lower score, better)  
- **C:** Efficiency ratio (computational cost relative to accuracy)
- **Points = (A + B + C) / 3**

Lower total points indicate better overall model performance.

## Getting Started

### Prerequisites

- R 4.0 or higher
- Required packages (see `requirements.txt`)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/APAN5902GP5.git
cd APAN5902GP5
```

2. Install R dependencies:
```r
# Install from requirements.txt or use:
packages <- c("data.table", "DT", "nnet", "glmnet", "caret", 
              "randomForest", "gbm", "e1071", "class", "h2o")
install.packages(packages)
```

3. Run the main analysis:
```r
# Open and knit the main Rmd file in RStudio
rmarkdown::render("analysis/01_main_analysis.Rmd")
```

### File Paths

The analysis expects the following directory structure from the working directory:
- `data/raw/` - Raw data files
- `data/processed/` - Sample data for training
- `src/` - R function files
- `results/` - Output directory for model results

## Usage Examples

### Running a Single Model

```r
# Source the functions
source("src/functions_sarah.R")

# Load test data
library(data.table)
test_set <- fread("data/raw/MNIST-fashion_testing_set-49.csv")

# Evaluate Random Forest across all sample sizes
results <- evaluate_rf(test_data = test_set)
```

### Accessing Results

```r
# Load saved model results
results_rf <- readRDS("results/models/results_rf.rds")

# View final scoreboard
scoreboard_final <- fread("results/scoreboards/scoreboard_final.csv")
print(scoreboard_final)
```

## Project Timeline

- **Model Development:** Each team member developed and evaluated their assigned models
- **Evaluation:** Models tested on 9 different training sets (3 samples × 3 sizes)
- **Analysis:** Results aggregated and compared using standardized scoring
- **Documentation:** Comprehensive reporting in R Markdown format

## Team Responsibilities

| Team Member | Models | Count |
|-------------|--------|-------|
| Kelsey Chen | Multinomial LR, Ridge, Lasso | 3 |
| Sarah Mastiur | Random Forest (2 versions), GBM (2 versions) | 4 |
| Mia Xue | SVM, Neural Networks, KNN | 3 |

## Documentation

- **METHODOLOGY.md** - Detailed explanation of each model, hyperparameters, and evaluation metrics
- **SETUP.md** - Environment configuration and dependency management
- **analysis/** - Full R Markdown reports with code and visualizations

## Output Files

### Main Deliverables

1. **scoreboard_final.csv** - Aggregated performance metrics for all models
2. **Image_Processing_Part_1.html** - Complete analysis report with model descriptions
3. **Additional_Analyses.html** - Supplementary analyses and insights

### Model Objects

All trained models and results are saved as RDS files in `results/models/` for reproducibility.

## Future Improvements

- Implement cross-validation for hyperparameter tuning
- Explore ensemble methods combining top models
- Test on higher resolution images (full 28×28 pixels)
- Investigate feature importance across models
- Deploy best model as interactive web application

## References

- Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
- R Package Documentation: CRAN
- ML Algorithms: Standard textbook implementations

## License

This project is part of coursework for Columbia University APAN5902.

## Contact

For questions or issues, please contact the project team or open an issue in the repository.

---

**Project Status:** Complete ✓  
**Last Updated:** November 2025
