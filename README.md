# Steam Game Success Prediction: A Multi-Model Machine Learning Approach

## Project Overview

This project presents a comprehensive machine learning analysis of Steam game data to predict game popularity and success metrics. The project employs multiple modeling approaches, including classification and regression tasks, to understand the factors that drive game success on the Steam platform.

The dataset consists of 72,563 Steam games with features including pricing, platform support, user reviews, categories, genres, and various metadata attributes. The project is organized into distinct components: data preprocessing, classification models for popularity prediction, and regression models for concurrent user prediction.

---

## Project Structure

```
GamePredictionMLProject/
├── data/
│   ├── processed/
│   │   ├── games_march2025_cleaned.csv      # Cleaned dataset (72,563 games, 22 features)
│   │   └── games_march2025_cleaned.7z      # Compressed archive
│   └── preprocessing/
│       └── 01_data_cleaning.ipynb          # Data preprocessing pipeline
│
├── notebooks/
│   ├── classification/
│   │   ├── 01_softmax_regression_baseline.ipynb
│   │   ├── 02_neural_network_pytorch.ipynb
│   │   └── 03_pre_release_prediction_rf_xgboost.ipynb
│   └── regression/
│       └── 04_peak_concurrent_users_keras.ipynb
│
└── reports/
    └── steam_success_predictor_summary.pdf
```

---

## Data Preprocessing

### File: `data/preprocessing/01_data_cleaning.ipynb`

**Purpose**: Clean and prepare raw Steam game data for machine learning analysis.

**Process**:
- Removed non-predictive columns (descriptions, images, URLs, temporal features)
- Converted boolean platform columns (windows, mac, linux) to binary format (0/1)
- Processed `estimated_owners` field by converting range strings (e.g., "100,000 - 200,000") to numeric midpoints
- Removed games with zero reviews to ensure data quality
- Limited tags to top 5 per game to reduce dimensionality
- Dropped constant columns (columns with only one unique value)

**Results**:
- Initial dataset: 89,618 games with 47 columns
- After cleaning: 72,563 games with 22 columns
- Data reduction: 19% reduction in rows, 53% reduction in columns
- Output: `games_march2025_cleaned.csv`

**Key Transformations**:
- Owner range conversion: "A - B" format → (A + B) / 2
- Platform encoding: True/False → 1/0
- Missing value handling: Removed rows with invalid owner data or zero reviews

---

## Classification Models

### File: `notebooks/classification/01_softmax_regression_baseline.ipynb`

**Objective**: Establish a baseline model for 3-class popularity classification using softmax regression (multiclass logistic regression) implemented from scratch.

**Model Type**: Softmax Regression (Linear Classifier)

**Implementation Details**:
- Custom implementation using NumPy (no scikit-learn)
- Gradient descent optimization with learning rate 0.01
- 1,000 iterations of training
- Manual gradient computation and loss calculation

**Target Variable**:
- Class 0 (Low Popularity): Bottom 25% by estimated owners (18,140 games)
- Class 1 (Medium Popularity): Middle 50% (36,282 games)
- Class 2 (High Popularity): Top 25% (18,141 games)

**Features**:
- 54 total features (including bias term)
- 13 numeric features: price, dlc_count, achievements, recommendations, user_score, positive, negative, peak_ccu, num_reviews_total, required_age, windows, mac, linux
- 40 categorical features: One-hot encoded top 100 game categories (e.g., Single-player, Multi-player, Steam Achievements)

**Preprocessing**:
- StandardScaler for feature normalization
- Added bias term to feature matrix
- Train/test split: 80/20 with stratification

**Results**:
- Training Accuracy: 52.57%
- Test Accuracy: 51.92%
- F1-Macro Score: 0.391
- Improvement over majority baseline: +10.74%

**Per-Class Performance**:
- Class 0 (Low): Precision 0.487, Recall 0.822, F1 0.611
- Class 1 (Medium): Precision 0.500, Recall 0.0005, F1 0.001
- Class 2 (High): Precision 0.594, Recall 0.531, F1 0.560

**Key Findings**:
- Model successfully beats the majority class baseline (41.18%)
- Severe class imbalance issue: Class 1 (Medium) has near-zero recall (0.0005), indicating the model struggles to identify medium-popularity games
- The model tends to predict either Low or High, effectively collapsing the Medium class
- Demonstrates that a simple linear model can capture some signal but requires class balancing techniques

**Training Details**:
- Final training loss: 0.988 (cross-entropy)
- Loss converged smoothly over 1,000 iterations
- No overfitting observed (train and test accuracy are similar)

---

### File: `notebooks/classification/02_neural_network_pytorch.ipynb`

**Objective**: Improve classification performance using a neural network to capture non-linear relationships between features and game popularity.

**Model Type**: Feed-Forward Neural Network (PyTorch)

**Architecture**:
- Input Layer: 464 features
- Hidden Layer 1: 64 neurons with Sigmoid activation
- Hidden Layer 2: 32 neurons with Sigmoid activation
- Output Layer: 3 classes (Low/Medium/High)
- Total Parameters: ~30,000 trainable parameters

**Training Configuration**:
- Framework: PyTorch
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Epochs: 50
- Train/Test Split: 80/20 with stratification

**Features**:
- 464 total features
- 4 numeric features: price, dlc_count, achievements, required_age
- 460 categorical features: MultiLabelBinarizer encoding of supported_languages, categories, and genres

**Preprocessing**:
- StandardScaler for numeric features
- MultiLabelBinarizer for categorical features (languages, categories, genres)
- Combined into single feature matrix

**Results**:
- Test Accuracy: 60.46%
- Final Training Loss: 0.8471
- Improvement over softmax baseline: +8.54 percentage points

**Training Progression**:
- Epoch 5: Accuracy 59.92%, Loss 0.9001
- Epoch 10: Accuracy 59.94%, Loss 0.8923
- Epoch 20: Accuracy 60.23%, Loss 0.8809
- Epoch 30: Accuracy 60.21%, Loss 0.8695
- Epoch 50: Accuracy 60.46%, Loss 0.8471

**Key Findings**:
- Neural network significantly outperforms the linear softmax regression model
- Larger feature space (464 vs 54 features) captures more nuanced patterns
- Model shows stable training with consistent accuracy improvement
- Still exhibits class imbalance issues (not explicitly addressed in this model)
- Demonstrates the value of non-linear models for complex feature interactions

**Advantages**:
- Captures non-linear relationships between features
- Handles high-dimensional categorical data effectively
- Better generalization than linear baseline

---

### File: `notebooks/classification/03_pre_release_prediction_rf_xgboost.ipynb`

**Objective**: Predict game success using only pre-release features, ensuring zero data leakage from post-launch metrics. This addresses a critical limitation of previous models that used post-release data.

**Model Types**: 
- Random Forest (with hyperparameter tuning)
- XGBoost (Gradient Boosting)

**Key Methodology**:
- **Data Leakage Prevention**: Strictly excluded all post-release metrics:
  - Reviews (positive, negative, num_reviews_total)
  - Peak concurrent users (peak_ccu)
  - User scores
  - Recommendations
  - Playtime metrics
- **Temporal Leakage Prevention**: Excluded features that accumulate over time:
  - DLC count (DLCs released after launch)
  - Steam Trading Cards (often gated until success is proven)
- **Balanced Classes**: Used 33%/33%/33% quantile splits (instead of 25%/50%/25%) to maximize signal separation

**Target Variable**:
- Class 0 (Low): Bottom 33% by estimated owners
- Class 1 (Medium): Middle 33%
- Class 2 (High): Top 33%

**Features** (49 pre-release features):
- Numeric: price, required_age, achievements, windows, mac, linux
- Engineered Budget Proxies:
  - `num_supported_languages`: Localization scope (proxy for budget)
  - `num_audio_languages`: Full audio localization count
  - `dev_team_size`: Number of developers
  - `publisher_count`: Number of publishers
- Categorical: Top 100 game categories (one-hot encoded)

**Random Forest Configuration**:
- Hyperparameter Tuning: RandomizedSearchCV with 20 iterations, 3-fold CV
- Best Parameters: n_estimators=100, max_depth=30, min_samples_split=10, min_samples_leaf=2, bootstrap=True
- Scoring Metric: F1-Macro (optimized for imbalanced classes)

**XGBoost Configuration**:
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 6
- eval_metric: mlogloss

**Preprocessing**:
- StandardScaler for all features
- Train/test split: 80/20 with stratification
- 5-fold stratified cross-validation for robust evaluation

**Results**:

**Random Forest**:
- Test Accuracy: 51.80%
- F1-Macro: 0.5122
- 5-Fold CV F1-Macro: 0.5096 (±0.0084)

**Per-Class Performance (Random Forest)**:
- Low: Precision 0.46, Recall 0.44, F1 0.45
- Medium: Precision 0.52, Recall 0.69, F1 0.59
- High: Precision 0.59, Recall 0.43, F1 0.50

**XGBoost**:
- Test Accuracy: 51.48%
- F1-Macro: 0.5070
- 5-Fold CV F1-Macro: 0.5053 (±0.0084)

**Per-Class Performance (XGBoost)**:
- Low: Precision 0.46, Recall 0.45, F1 0.46
- Medium: Precision 0.51, Recall 0.71, F1 0.59
- High: Precision 0.63, Recall 0.38, F1 0.47

**Feature Importance Analysis** (Top 5):
1. achievements
2. cat_Remote_Play_on_TV
3. cat_Family_Sharing
4. cat_Remote_Play_on_Tablet
5. price

**Additional Experiments**:

**Binary Classification (High vs Low)**:
- Accuracy: 61.74%
- F1 Score: 0.5766
- Interpretation: Removing the ambiguous Medium class significantly improves performance, validating that the core signal exists but is obscured by the middle tier

**Price-Only Baseline**:
- Logistic Regression (price only): 38.21% accuracy
- Complex Model Lift: +13.59 percentage points
- Interpretation: Feature engineering provides substantial value beyond simple price heuristics

**Key Findings**:
- Pre-release metadata contains predictive signal (13.6% lift over price-only baseline)
- Budget proxies (localization scope, team size) are strong predictors of success
- Medium class remains challenging to distinguish from High success games
- Model achieves 61.7% accuracy in binary High vs Low classification, demonstrating clear signal separation
- Production value indicators (achievements, platform features) are more predictive than genre alone
- Marketing and community sentiment (not captured in metadata) likely drive breakout hits

**Limitations**:
- Real-world game success follows a Power Law distribution (not balanced 33% splits)
- Missing context: No unstructured data (descriptions, screenshots) or marketing signals
- Medium class ambiguity: Metadata can identify professional products but not predict marketing success

---

## Regression Models

### File: `notebooks/regression/04_peak_concurrent_users_keras.ipynb`

**Objective**: Predict the peak number of concurrent users (peak_ccu) for Steam games using structured metadata and community engagement signals. This is a regression task focusing on player concurrency rather than overall popularity.

**Model Type**: Feed-Forward Neural Network (TensorFlow/Keras)

**Architecture**:
- Input Layer: 18 engineered features
- Hidden Layer 1: 128 neurons, ReLU activation, Dropout 0.3
- Hidden Layer 2: 64 neurons, ReLU activation, Dropout 0.2
- Hidden Layer 3: 32 neurons, ReLU activation
- Output Layer: 1 neuron (regression)

**Training Configuration**:
- Framework: TensorFlow/Keras
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)
- Batch Size: 16
- Epochs: 60
- Validation Split: 20%

**Target Variable**:
- `peak_ccu`: Peak concurrent users (log-transformed)
- Transformation: `y = log1p(peak_ccu)` to handle extreme skew
- Reason: Peak CCU follows a highly skewed distribution with a few games reaching millions of concurrent players

**Feature Engineering** (18 features):

**Engagement Ratios**:
- `positive_ratio`: positive / (positive + negative + 1)
- `reviews_to_owners`: num_reviews_total / (estimated_owners + 1)
- `recommendations_per_review`: recommendations / (num_reviews_total + 1)

**Content Richness Counts**:
- `supported_languages_count`: Number of supported languages
- `full_audio_languages_count`: Number of full audio languages
- `categories_count`: Number of categories
- `genres_count`: Number of genres
- `tags_count`: Number of tags

**Platform and Monetization**:
- `platform_count`: windows + mac + linux
- `is_free`: Binary indicator (price == 0)
- `price_log`: log1p(price)
- `dlc_per_owner`: dlc_count / (estimated_owners + 1)

**Other Features**:
- required_age, dlc_count, achievements, recommendations, positive, negative

**Preprocessing**:
- StandardScaler for all features
- Train/test split: 80/20
- Target log transformation: log1p(peak_ccu)
- Handled infinite values and missing data

**Results**:

**Log Scale Performance**:
- Test MAE (log scale): 0.440
- Final Training Loss: ~0.50 (MSE)

**Real Scale Performance** (after inverse transformation):
- MAE (all games): Numerical overflow (due to extreme outliers)
- MAE (games with peak_ccu < 100,000, predictions clipped to 1M): **10,342 users**

**Training Progression**:
- Epoch 1: Loss 1.227, MAE 0.569
- Epoch 10: Loss 0.562, MAE 0.426
- Epoch 30: Loss 0.514, MAE 0.404
- Epoch 60: Loss ~0.50, MAE ~0.40

**Key Findings**:
- Model successfully captures general popularity trends rather than precise concurrency spikes
- Log transformation is essential for training stability on highly skewed data
- Mean absolute error of ~10,000 users is reasonable given the wide range of peak CCU values
- Model is effective for games with moderate popularity (< 100,000 peak CCU)
- Extreme outliers (games with millions of concurrent users) are difficult to predict accurately
- Useful for server capacity planning and multiplayer viability assessment

**Practical Applications**:
- Estimate server requirements before game launch
- Assess multiplayer game viability
- Prioritize promotional efforts based on predicted engagement
- Understand which features correlate with higher player concurrency

**Limitations**:
- Cannot predict exact player spikes (requires temporal/marketing data)
- Struggles with extreme outliers (viral games)
- Missing temporal features (release timing, marketing campaigns)
- No access to external factors (streaming, social media buzz)

---

## Comparative Analysis

### Model Performance Summary

| Model | Task | Accuracy/F1 | Key Strength | Key Limitation |
|-------|------|--------------|--------------|----------------|
| Softmax Regression | 3-Class Classification | 51.92% / F1: 0.391 | Simple, interpretable baseline | Severe class imbalance, near-zero Medium recall |
| Neural Network (PyTorch) | 3-Class Classification | 60.46% | Captures non-linear patterns, best classification accuracy | Still has class imbalance issues |
| Random Forest (Pre-Release) | 3-Class Classification | 51.80% / F1: 0.512 | No data leakage, practical for real-world use | Lower accuracy due to limited features |
| XGBoost (Pre-Release) | 3-Class Classification | 51.48% / F1: 0.507 | No data leakage, robust to overfitting | Similar limitations to Random Forest |
| Neural Network (Keras) | Regression (Peak CCU) | MAE: 10,342 users | Captures popularity trends | Cannot predict exact spikes, struggles with outliers |

### Key Insights Across All Models

1. **Class Imbalance is Critical**: All classification models struggle with the Medium popularity class, which represents 50% of the data in the original split but is poorly predicted. Binary classification (High vs Low) achieves 61.7% accuracy, demonstrating that the signal exists but is obscured by the ambiguous middle tier.

2. **Feature Engineering Matters**: The pre-release model achieves a 13.6% lift over a simple price-only baseline, proving that engineered features (budget proxies, platform support) capture meaningful signal beyond basic pricing.

3. **Non-Linear Models Outperform**: The PyTorch neural network (60.5% accuracy) significantly outperforms the linear softmax regression (51.9% accuracy), indicating that game success depends on complex feature interactions.

4. **Data Leakage is a Real Problem**: Models using post-release metrics (reviews, peak_ccu) achieve higher accuracy but are not practical for pre-launch prediction. The pre-release models (51-52% accuracy) represent realistic performance expectations.

5. **Log Transformation is Essential**: For regression on highly skewed targets like peak_ccu, log transformation is necessary for training stability and meaningful predictions.

6. **Budget Proxies are Strong Predictors**: Features like number of supported languages, achievements, and platform features (Remote Play, Family Sharing) act as proxies for production budget and are among the top predictors of success.

---

## Technical Requirements

### Python Packages
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch (PyTorch)
- tensorflow/keras
- xgboost
- imbalanced-learn (for SMOTE, if used)

### Data
- Primary dataset: `data/processed/games_march2025_cleaned.csv`
- Original raw data: Available in compressed format (`games_march2025_cleaned.7z`)

---

## Usage Instructions

1. **Data Preparation**: Extract `games_march2025_cleaned.csv` from the `.7z` archive if needed, or run the data cleaning notebook to generate it from raw data.

2. **Running Notebooks**: 
   - For local execution: Ensure the dataset is in `data/processed/` directory
   - For Google Colab: Upload the dataset when prompted
   - Notebooks are designed to work in both environments

3. **Execution Order** (for classification models):
   - Start with `01_softmax_regression_baseline.ipynb` for baseline comparison
   - Proceed to `02_neural_network_pytorch.ipynb` for improved performance
   - Use `03_pre_release_prediction_rf_xgboost.ipynb` for practical pre-launch prediction

4. **Regression Model**: The peak CCU prediction model (`04_peak_concurrent_users_keras.ipynb`) is independent and can be run separately.

---

## Future Improvements

1. **Address Class Imbalance**: Implement SMOTE oversampling, class weights, or focal loss to improve Medium class prediction
2. **Feature Engineering**: Add interaction features, polynomial features, and text-based features from game descriptions
3. **External Data**: Incorporate marketing spend, social media metrics, streaming data, and review sentiment analysis
4. **Ensemble Methods**: Combine multiple models (voting, stacking) for improved robustness
5. **Hyperparameter Tuning**: Systematic optimization for all models using grid search or Bayesian optimization
6. **Cross-Validation**: Implement k-fold cross-validation for all models to ensure robust performance estimates

---

## Conclusion

This project demonstrates that machine learning models can extract meaningful signals from Steam game metadata to predict popularity and engagement. While no single model achieves perfect accuracy, the combination of approaches provides valuable insights:

- Linear models establish baselines and demonstrate basic signal
- Non-linear models capture complex feature interactions
- Pre-release models offer practical utility for game developers
- Regression models enable infrastructure planning

The consistent challenge across all models is the ambiguous "Medium" popularity class, suggesting that game success depends on factors beyond metadata—particularly marketing, community building, and timing—that are not captured in structured data alone.

---

## Additional Resources

For a comprehensive analysis of methodology, detailed results, and conclusions, please refer to:
- Full project report: `reports/steam_success_predictor_summary.pdf`
- Presentation Deck: [Presentation Deck](https://docs.google.com/document/d/1-Yo-nN80w7fhqkkA31zVS5tt6KpWvVwQqtO-80TFUDw/edit?usp=sharing)

---

## Authors

This project was developed as part of a 2025 Machine Learning course, with different team members contributing different modeling approaches.

---

## License

This project is for educational purposes.
