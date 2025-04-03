
# Random Forest with Biomechanically handcrafter features, original data

```sh
              precision    recall  f1-score   support

           0       0.92      0.90      0.91        39
           1       0.89      0.92      0.91        37

    accuracy                           0.91        76
   macro avg       0.91      0.91      0.91        76
weighted avg       0.91      0.91      0.91        76

Confusion Matrix:
 [[35  4]
 [ 3 34]]
ROC-AUC Score: 0.9730
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

           0       0.92      0.90      0.91        39
           1       0.89      0.92      0.91        37

    accuracy                           0.91        76
   macro avg       0.91      0.91      0.91        76
weighted avg       0.91      0.91      0.91        76

Confusion Matrix:
 [[35  4]
 [ 3 34]]
ROC-AUC Score: 0.9730
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