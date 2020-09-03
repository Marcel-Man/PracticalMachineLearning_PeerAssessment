---
title: Weighting Lifting Exercise prediction
  analysis
output:
  html_document:
    keep_md: yes
---


```r
library(caret)
library(gbm)
library(dplyr)
library(tidyverse)
```

## Data Loading
The data in this report comes from http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. We start off by downloading the training and testing datasets into R.


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="pml-testing.csv")
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

## Data Cleanup
The original data is organized in a way that there are some measurement lines, followed by a summary line. The summary line contains a lot of NA / DIV!0 values with the measurements, and is denoted in the data by the new_window = 'yes' variable. Here we filter them out as summary is simply transformation of measurement statistics, and would be expected to be highly-correlated with them. Besides, we take out the independent variables used in our model.


```r
set.seed(12345)
training_clean <- training %>% filter(new_window == 'no') %>% select(starts_with("gyros"), starts_with("accel"), starts_with("magnet"), starts_with("classe")) 
```

## Create data subsets for cross-validation
We partition the training set into training and validation subsets for cross-validation purpose, assuming a 70-30 split.


```r
training_subIndex <- createDataPartition(y=training_clean$classe, p=0.7, list=FALSE)
training_sub <- training_clean[training_subIndex, ] 
validation_sub <- training_clean[-training_subIndex, ]
```

## Model Fitting and Validation
We will begin by fitting the training subset to 3 models, random forest, boosting and linear discriminant analysis. Then we validate the performance of our models by passing the validation subset to them. As can be seen below, random forest attained the highest level of accuracy from the validation subset, and better out-of-sample error than boosting and linear discriminant analysis models. 
As a reference, we also printed out the times so you may compare the relative time taken to build to models.


```r
Sys.time()
```

```
## [1] "2020-09-03 19:38:12 CST"
```

```r
mod1 <- train(classe ~ ., method="rf", data=training_sub)
Sys.time()
```

```
## [1] "2020-09-03 20:27:47 CST"
```

```r
mod2 <- train(classe ~ ., method="gbm", data=training_sub, verbose=FALSE)
Sys.time()
```

```
## [1] "2020-09-03 21:35:57 CST"
```

```r
mod3 <- train(classe ~ ., method="lda", data=training_sub)
Sys.time()
```

```
## [1] "2020-09-03 21:36:05 CST"
```

```r
pred1 <- predict(mod1, validation_sub)
pred2 <- predict(mod2, validation_sub)
pred3 <- predict(mod3, validation_sub)

confusionMatrix(as.factor(validation_sub$classe), pred1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1640    1    0    0    0
##          B   16 1084   15    0    0
##          C    1   12  992    0    0
##          D    1    0   23  918    2
##          E    0    0    3    7 1048
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9859          
##                  95% CI : (0.9826, 0.9888)
##     No Information Rate : 0.2877          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9822          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9891   0.9881   0.9603   0.9924   0.9981
## Specificity            0.9998   0.9934   0.9973   0.9946   0.9979
## Pos Pred Value         0.9994   0.9722   0.9871   0.9725   0.9905
## Neg Pred Value         0.9956   0.9972   0.9914   0.9985   0.9996
## Prevalence             0.2877   0.1904   0.1792   0.1605   0.1822
## Detection Rate         0.2846   0.1881   0.1721   0.1593   0.1818
## Detection Prevalence   0.2847   0.1935   0.1744   0.1638   0.1836
## Balanced Accuracy      0.9944   0.9908   0.9788   0.9935   0.9980
```

```r
confusionMatrix(as.factor(validation_sub$classe), pred2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1578   18   14   29    2
##          B   81  955   61   10    8
##          C   27   54  905   14    5
##          D   11   11   71  837   14
##          E   10   29   25   36  958
## 
## Overall Statistics
##                                           
##                Accuracy : 0.908           
##                  95% CI : (0.9003, 0.9154)
##     No Information Rate : 0.2962          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8835          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9244   0.8950   0.8411   0.9039   0.9706
## Specificity            0.9845   0.9659   0.9787   0.9779   0.9791
## Pos Pred Value         0.9616   0.8565   0.9005   0.8867   0.9055
## Neg Pred Value         0.9687   0.9759   0.9641   0.9815   0.9938
## Prevalence             0.2962   0.1851   0.1867   0.1607   0.1713
## Detection Rate         0.2738   0.1657   0.1570   0.1452   0.1662
## Detection Prevalence   0.2847   0.1935   0.1744   0.1638   0.1836
## Balanced Accuracy      0.9544   0.9305   0.9099   0.9409   0.9748
```

```r
confusionMatrix(as.factor(validation_sub$classe), pred3)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1247   77  161  135   21
##          B  218  649  134   41   73
##          C  197   89  593  101   25
##          D   76   72  134  557  105
##          E   60  195   78  164  561
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6259          
##                  95% CI : (0.6132, 0.6384)
##     No Information Rate : 0.312           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5253          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6935   0.5998   0.5391  0.55812  0.71465
## Specificity            0.9006   0.9004   0.9116  0.91878  0.90016
## Pos Pred Value         0.7599   0.5821   0.5900  0.59004  0.53025
## Neg Pred Value         0.8663   0.9068   0.8934  0.90849  0.95239
## Prevalence             0.3120   0.1877   0.1909  0.17317  0.13621
## Detection Rate         0.2164   0.1126   0.1029  0.09665  0.09735
## Detection Prevalence   0.2847   0.1935   0.1744  0.16380  0.18358
## Balanced Accuracy      0.7971   0.7501   0.7254  0.73845  0.80741
```

## Model Testing
Finally, we apply the selected model to the testing dataset. 


```r
predict(mod1, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
