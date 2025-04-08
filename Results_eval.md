
# Random Forest with Biomechanically handcrafter features, original data

```sh
              precision    recall  f1-score   support

           0       0.92      0.88      0.90        26
           1       0.88      0.92      0.90        25

    accuracy                           0.90        51
   macro avg       0.90      0.90      0.90        51
weighted avg       0.90      0.90      0.90        51

Confusion Matrix:
 [[23  3]
 [ 2 23]]
🏆 ROC-AUC Score: 0.9708
```

# Random Forest with Biomechanically handcrafter features, training set augmented

```sh
              precision    recall  f1-score   support

           0       0.93      0.95      0.94        39
           1       0.94      0.92      0.93        37

    accuracy                           0.93        76
   macro avg       0.93      0.93      0.93        76
weighted avg       0.93      0.93      0.93        76

Confusion Matrix:
 [[37  2]
 [ 3 34]]
ROC-AUC Score: 0.9913

```

# Random Forest with raw features (3D joint coordinates, joint angles), original set

```sh
              precision    recall  f1-score   support

           0       0.96      0.88      0.92        26
           1       0.89      0.96      0.92        25

    accuracy                           0.92        51
   macro avg       0.92      0.92      0.92        51
weighted avg       0.92      0.92      0.92        51

Confusion Matrix:
 [[23  3]
 [ 1 24]]
 ROC-AUC Score: 0.9838
```

# Random Forest with raw features (3D joint coordinates, joint angles), training set augmented

```sh
              precision    recall  f1-score   support

           0       0.97      0.97      0.97        39
           1       0.97      0.97      0.97        37

    accuracy                           0.97        76
   macro avg       0.97      0.97      0.97        76
weighted avg       0.97      0.97      0.97        76

Confusion Matrix:
 [[38  1]
 [ 1 36]]
ROC-AUC Score: 0.9972

```

# BiLSTM with raw features, normalized and padded, original data set 

```sh
              precision    recall  f1-score   support

           0       1.00      0.85      0.92        39
           1       0.86      1.00      0.92        37

    accuracy                           0.92        76
   macro avg       0.93      0.92      0.92        76
weighted avg       0.93      0.92      0.92        76


 Confusion Matrix:
[[33  6]
 [ 0 37]]

 ROC-AUC Score: 0.9910
```

# BiLSTM with Bio features, normalized and padded, original data set 
```sh
📊 Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.90      0.93        39
           1       0.90      0.97      0.94        37

    accuracy                           0.93        76
   macro avg       0.94      0.94      0.93        76
weighted avg       0.94      0.93      0.93        76


🧩 Confusion Matrix:
[[35  4]
 [ 1 36]]

🏆 ROC-AUC Score: 0.9771

```

# BiLSTM with Bio features, normalized and padded, augmented training set
```sh
 Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96        39
           1       0.97      0.95      0.96        37

    accuracy                           0.96        76
   macro avg       0.96      0.96      0.96        76
weighted avg       0.96      0.96      0.96        76


 Confusion Matrix:
[[38  1]
 [ 2 35]]

 ROC-AUC Score: 0.9854
 ```

 