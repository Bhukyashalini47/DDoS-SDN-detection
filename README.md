#Step 1: Import all the libraries
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") 
#for classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv('SDN_DDOS_dataset.csv')
df.head()
df.shape
df.info()
df.dtypes
df['label'] = df['label'].astype('category')
df['time'] = pd.to_datetime(df['time'])
df['switch_id'] = df['switch_id'].astype('object')
#Check the data types again
df.dtypes
df.head()
df.describe(exclude=[np.number])
#Checking for missing values
df.isnull().sum()
df.head()
df.drop(columns={df.columns[0],df.columns[1]},inplace=True)
df.describe()
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
df = scaler.fit_transform(df)

# Print the standardized data
print("Standardized Data:")
print(df)

df=pd.DataFrame(df)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Assuming 'df' is your pandas DataFrame containing the dataset
# Normalize the entire DataFrame
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Now 'df_normalized' contains the normalized dataset
df

df.head()


# Assuming X contains your features and y contains your target variable
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
#checking for missing values in X
X.isna().sum()
X.fillna(df.mean(), inplace=True)
X.isna().sum()
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = svm_classifier.predict(X_test)


# generating confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

# generating confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



# Initialize SVM classifier
# svm_classifier = SVC(kernel='linear', probability=True)  # You can choose different kernels based on your data
# svm_classifier.fit(X_train, y_train)

# Predict probabilities for the positive class
y_score = svm_classifier.decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

svm_classifier = SVC(kernel='linear', probability=True)  # You can choose different kernels based on your data
svm_classifier.fit(X_train, y_train)
#
# Predict probabilities for the positive class
y_score = svm_classifier.decision_function(X_test)

# Compute precision-recall curve and area under the curve
precision, recall, _ = precision_recall_curve(y_test, y_score)
average_precision = average_precision_score(y_test, y_score)

# Plot precision-recall curve
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.show()
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier with regularization (adjust C parameter for regularization)
# You can also adjust other parameters such as kernel, degree, etc., based on your dataset characteristics
# svm_classifier = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the SVM model
svm_classifier.fit(X_train, y_train)

# Predict labels for test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import train_test_split, cross_val_score

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Perform cross-validation to evaluate the model
# Adjust cv parameter for the number of folds in cross-validation
# Higher cv values can provide more reliable estimates but require more computational resources
cv_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)

# Calculate the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("Mean Cross-Validation Score:", mean_cv_score)

# Train the SVM model on the entire training set
svm_classifier.fit(X_train, y_train)

# Evaluate the model on the test set
test_accuracy = svm_classifier.score(X_test, y_test)
print("Test Set Accuracy:", test_accuracy)

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Initialize stratified k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(svm_classifier, X, y, cv=kfold)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Calculate and print the mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print("Mean Cross-Validation Score:", mean_cv_score)

from sklearn.metrics import confusion_matrix

# generating confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have initialized and fitted your SVM classifier already
svm_classifier = SVC(kernel='linear', probability=True)  # You can choose different kernels based on your data
svm_classifier.fit(X_train, y_train)

# Predict probabilities for the positive class
y_score = svm_classifier.decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score

svm_classifier = SVC(kernel='linear', probability=True)  # You can choose different kernels based on your data
svm_classifier.fit(X_train, y_train)
#
# Predict probabilities for the positive class
y_score = svm_classifier.decision_function(X_test)

# Compute precision-recall curve and area under the curve
precision, recall, _ = precision_recall_curve(y_test, y_score)
average_precision = average_precision_score(y_test, y_score)

# Plot precision-recall curve
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# Initialize and train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

# Get decision function values (raw scores)
decision_values = svm_classifier.decision_function(X)

# Compute hinge loss
hinge_loss = np.maximum(0, 1 - y * decision_values).mean()

print("Hinge Loss:", hinge_loss)

!pip  install tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset

# Perform preprocessing
# For example, you can scale features
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Split the data into features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Define CNN architecture
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape data for CNN input (add one dimension for channel)
X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train the model
model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_cnn, y_test)
print("Test Accuracy:", accuracy)

model.summary()

import numpy as np
from sklearn.metrics import confusion_matrix

# Make predictions on test data
y_pred_prob = model.predict([X_test_cnn, X_test_cnn])
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Make predictions on test data
y_pred_prob = model.predict([X_test_cnn, X_test_cnn])


# Assuming y_test and y_pred_prob are already defined appropriately

# Flatten the y_pred_prob if needed
y_pred_prob_flat = y_pred_prob.flatten()

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob_flat)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

from tensorflow.keras.layers import LSTM

# Define RNN architecture
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_cnn, y_test)
print("Test Accuracy:", accuracy)

# Make predictions on test data
y_pred_probs = model.predict(X_test_cnn)
y_pred_binary = (y_pred_probs > 0.5).astype(int)  # Threshold probabilities to get binary predictions

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute fpr, tpr, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Compute ROC area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Compute precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, Concatenate

# Define CNN part of the hybrid model
cnn_input = Input(shape=(X_train.shape[1], 1))
conv1 = Conv1D(32, 3, activation='relu')(cnn_input)
maxpool1 = MaxPooling1D(2)(conv1)
flatten1 = Flatten()(maxpool1)

# Define RNN part of the hybrid model
rnn_input = Input(shape=(X_train.shape[1], 1))
lstm1 = LSTM(64)(rnn_input)

# Concatenate the outputs of CNN and RNN
merged = Concatenate()([flatten1, lstm1])

# Dense layers for classification
dense1 = Dense(64, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense1)

# Define the model
model = Model(inputs=[cnn_input, rnn_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train_cnn, X_train_cnn], y_train, epochs=10, batch_size=32, validation_data=([X_test_cnn, X_test_cnn], y_test))

# Make predictions on test data
y_pred_probs = model.predict([X_test_cnn, X_test_cnn])

# Convert predicted probabilities to binary predictions
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# Compute confusion matrix
confusionmatrix = confusion_matrix(y_test, y_pred_binary)

# Print confusion matrix
print("Confusion Matrix:")
print(confusionmatrix)
plt.figure(figsize=(8, 6))
sns.heatmap(confusionmatrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Make predictions on test data
y_pred_probs = model.predict([X_test_cnn, X_test_cnn])

# Compute fpr, tpr, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Compute ROC area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Make predictions on test data
y_pred_probs = model.predict([X_test_cnn, X_test_cnn])

# Compute precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()







