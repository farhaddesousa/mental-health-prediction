# Load necessary libraries
library(ISLR)
library(tree)
library(randomForest)
library(gbm)
library(e1071)
library(plot3D)
library(caret)
library(keras)
library(tensorflow)


#load data
load("mhcld_puf_2019.RData")

# Convert 'df' to a data frame called 'data'
data <- data.frame(df)

# Select all columns except 'YEAR', 'CASEID', and 'STATEFIP'
selected_data <- subset(data, select = -c(YEAR, CASEID, STATEFIP))

# Remove rows with missing 'SMISED' values (-9 indicates missing)
cleaned_data <- subset(selected_data, SMISED != -9)

# Convert all columns to factors
col_names <- names(cleaned_data)
cleaned_data[, col_names] <- lapply(cleaned_data[, col_names], factor)

# Set seed for reproducibility
set.seed(123)

# Split data into training (70%) and test (30%) sets
sample_indices <- sample(c(TRUE, FALSE), nrow(cleaned_data), replace = TRUE, prob = c(0.7, 0.3))
train <- cleaned_data[sample_indices, ]
test <- cleaned_data[!sample_indices, ]

# Create smaller subsets for computational feasibility
n <- min(5000, nrow(train))
small_train <- train[1:n, ]
m <- min(5000, nrow(test))
small_test <- test[1:m, ]

# ---- Simple Classification Tree ----
# Fit a classification tree model
tree_SMISED <- tree(SMISED ~ ., data = train)
summary(tree_SMISED)

# Plot the tree
plot(tree_SMISED)
text(tree_SMISED, pretty = 0)

# Predict on test data
tree_pred <- predict(tree_SMISED, test, type = "class")

# Confusion matrix
table(tree_pred, test$SMISED)

# ---- Random Forest ----
# Fit a random forest model
forest_SMISED <- randomForest(SMISED ~ ., data = small_train, mtry = 4, ntree = 50, importance = TRUE)
print(forest_SMISED)

# Predict on test data
forest_pred <- predict(forest_SMISED, newdata = small_test)

# Calculate accuracy
test_accuracy <- sum(forest_pred == small_test$SMISED) / length(forest_pred)
train_accuracy <- sum(predict(forest_SMISED, newdata = small_train) == small_train$SMISED) / nrow(small_train)

# Variable importance
importance(forest_SMISED)

# ---- Boosting ----
# Fit a boosting model
boost_SMISED <- gbm(SMISED ~ ., data = small_train, distribution = "multinomial",
                    n.trees = 50, interaction.depth = 1)
summary(boost_SMISED)

# Predict on test data
boost_pred <- predict(boost_SMISED, newdata = small_test, n.trees = 50, type = "response")

# Convert probabilities to class labels
boost_pred_class <- apply(boost_pred, 1, which.max)
boost_test_accuracy <- sum(boost_pred_class == as.numeric(small_test$SMISED)) / nrow(small_test)

# ---- Logistic Regression ----
# Fit a logistic regression model
logreg_SMISED <- glm(SMISED ~ ., data = small_train, family = binomial)
summary(logreg_SMISED)

# ---- Support Vector Machine (SVM) ----
# Fit an SVM model with polynomial kernel
svm_SMISED <- svm(SMISED ~ ., data = small_train, kernel = "polynomial", degree = 2, cost = 10)

# Predict on training data
train_pred <- predict(svm_SMISED, small_train)
table(predict = train_pred, truth = small_train$SMISED)

# Predict on test data
test_pred <- predict(svm_SMISED, small_test)
table(predict = test_pred, truth = small_test$SMISED)

# Calculate test accuracy
test_accuracy <- sum(test_pred == small_test$SMISED) / length(test_pred)

# ---- Cross-Validation for SVM ----
# Perform cross-validation to find optimal 'degree' and 'cost'
cv_error <- matrix(nrow = 10, ncol = 10)
for (i in 1:10) {
  for (j in 1:10) {
    svm_model <- svm(SMISED ~ ., data = small_train, kernel = "polynomial", degree = i, cost = j)
    svm_pred <- predict(svm_model, small_test)
    cv_error[i, j] <- 1 - sum(svm_pred == small_test$SMISED) / length(svm_pred)
  }
}

# Plot cross-validation error
x <- 1:10
y <- 1:10
z <- cv_error
hist3D(x = x, y = y, z = z, zlim = c(0, 0.5), theta = 300, phi = 40, axes = TRUE, label = TRUE,
       xlab = "Polynomial degree", ylab = "Cost", zlab = "Error", nticks = 3, ticktype = "detailed",
       space = 0.5, lighting = FALSE, light = "diffuse", shade = 0.5)

# Find optimal parameters
opt_indices <- which(cv_error == min(cv_error), arr.ind = TRUE)
opt_degree <- opt_indices[1, 1]
opt_cost <- opt_indices[1, 2]

# Fit optimal SVM model
svm_SMISED_opt <- svm(SMISED ~ ., data = small_train, kernel = "polynomial", degree = opt_degree, cost = opt_cost)
svm_pred_opt <- predict(svm_SMISED_opt, small_test)
svm_test_accuracy <- sum(svm_pred_opt == small_test$SMISED) / length(svm_pred_opt)

# ---- Neural Network ----
# Preprocess data for neural network
# One-hot encode categorical variables
dummy_train <- dummyVars(" ~ .", data = small_train)
x_train <- data.frame(predict(dummy_train, newdata = small_train))
y_train <- as.numeric(small_train$SMISED) - 1  # Adjusting factor levels to start at 0

dummy_test <- dummyVars(" ~ .", data = small_test)
x_test <- data.frame(predict(dummy_test, newdata = small_test))
y_test <- as.numeric(small_test$SMISED) - 1

# Convert to matrices
x_train <- as.matrix(x_train)
x_test <- as.matrix(x_test)

# Define the neural network model
num_classes <- length(unique(y_train))
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2
)

# Evaluate the model on test data
score <- model %>% evaluate(x_test, y_test)
cat('Test loss:', score$loss, '\n')
cat('Test accuracy:', score$accuracy, '\n')
