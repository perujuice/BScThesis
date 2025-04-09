
# BiLSTM with bio features, normalized and padded

## The first run was done using original data and this is the Cross Validation result

```sh
 Best Parameters: {'epochs': 100, 'model__dropout_rate': 0.3, 'model__learning_rate': 0.0003, 'model__lstm_units': 96}
 Best Cross-Validated Accuracy: 0.9450
    mean_test_score  std_test_score  \
25         0.945000        0.050990   
50         0.915244        0.058458   
52         0.915122        0.064534   
40         0.915000        0.048990   
43         0.905122        0.104246   

                                               params  
25  {'epochs': 100, 'model__dropout_rate': 0.3, 'm...  
50  {'epochs': 150, 'model__dropout_rate': 0.4, 'm...  
52  {'epochs': 150, 'model__dropout_rate': 0.4, 'm...  
40  {'epochs': 150, 'model__dropout_rate': 0.2, 'm...  
43  {'epochs': 150, 'model__dropout_rate': 0.3, 'm... 
```

## This is the evaluation using all 20 seeds to split the test set 20 times in various ways

```sh
Final evaluation over multiple test splits:
Accuracy  : 0.9588 ± 0.0316
Precision : 0.9280 ± 0.0488
Recall    : 0.9960 ± 0.0120
F1        : 0.9602 ± 0.0294
Roc_auc   : 0.9844 ± 0.0171

Mean Confusion Matrix (rounded):
[[24  2]
 [ 0 25]]
```

![alt text](figs/fig1.png)

## Finally I included a representation of what features hurt the outcome most by using a technique callsed Permutation Feature Importance for the BiLSTM

![alt text](figs/fig2.png)


# BiLSTM with raw features, normalized and padded

## The first run was done using original data and this is the Cross Validation result

```sh
 Best Parameters: {'epochs': 100, 'model__dropout_rate': 0.2, 'model__learning_rate': 0.0005, 'model__lstm_units': 64}
 Best Cross-Validated Accuracy: 0.9554
    mean_test_score  std_test_score  \
21         0.955366        0.028785   
23         0.950366        0.031341   
22         0.950244        0.027390   
14         0.950244        0.027390   
25         0.950122        0.031720   

                                               params  
21  {'epochs': 100, 'model__dropout_rate': 0.2, 'm...  
23  {'epochs': 100, 'model__dropout_rate': 0.2, 'm...  
22  {'epochs': 100, 'model__dropout_rate': 0.2, 'm...  
14  {'epochs': 50, 'model__dropout_rate': 0.4, 'mo...  
25  {'epochs': 100, 'model__dropout_rate': 0.3, 'm...  
```

## This is the evaluation using all 20 seeds to split the test set 20 times in various ways

```sh
 Final evaluation over multiple test splits:
Accuracy  : 0.9951 ± 0.0085
Precision : 1.0000 ± 0.0000
Recall    : 0.9900 ± 0.0173
F1        : 0.9949 ± 0.0088
Roc_auc   : 0.9986 ± 0.0027

🧩 Mean Confusion Matrix (rounded):
[[26  0]
 [ 0 25]]
 ```

 ![alt text](figs/fig3.png)

 ## Finally I included a representation of what features hurt the outcome most by using a technique callsed Permutation Feature Importance for the BiLSTM
- Clearly less interpretable, the features dont tell us much about what went wrong but the model performs extremely well and consistently!

 ![alt text](figs/fig4.png)