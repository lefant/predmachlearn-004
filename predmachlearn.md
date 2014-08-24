Prediction of exercise execution quality from accelerometer data
================================================================

by Fabian Linzberger

github repo with Rmarkdown source code:
https://github.com/lefant/predmachlearn-004

online rendered version on github pages:
http://lefant.github.io/predmachlearn-004/predmachlearn.html



## Summary

I downloaded data recorded from various accelerometers during exercise
from the Human Activity Recognition dataset at [har] and trained a
model to predict which class of exercise execution quality the
measurement is from.

As there are a decent number of samples (almost 20000) for training
and a lot of variables that could be significant (54 possible
predictors after cleanup) a blackbox model with automatic feature
selection seems promising. A first attempt using random forest
immediately performed very well on the cross validation set, so no
further models where considered. It also turned out that all of the 20
predictions on the test set where correct :)


## Data Processing

### Load dependencies, enable multicore processing


```r
library(reshape2)
library(caret)
library(randomForest)
library(doParallel)
registerDoParallel(cores = detectCores())
```


### Read in csv data, remove columns without useful data


```r
trainingRaw <- read.csv("pml-training.csv", na.strings=c("", "\"\"", "NA"))
testingRaw <- read.csv("pml-testing.csv", na.strings=c("", "\"\"", "NA"))
trainingNonEmpty <- trainingRaw[, colSums(!is.na(testingRaw)) != 0]
testingNonEmpty <- testingRaw[, colSums(!is.na(testingRaw)) != 0]

delCols <- function(data) {
    subset(data, select = -c(X,
                             raw_timestamp_part_1,
                             raw_timestamp_part_2,
                             new_window,
                             num_window))
}

trainingPre <- delCols(trainingNonEmpty)
testing <- delCols(testingNonEmpty)
```


### Split data into training data set (90%) and cross validation data set (10%)


```r
set.seed(0)
train = sample(1:dim(trainingPre)[1],size=dim(trainingPre)[1] * 0.9, replace=F)
training = trainingPre[train,]
validating = trainingPre[-train,]
```

### Train a random forest model on the training data set

```r
modFit <- train(classe ~ ., method="rf", trControl = trainControl(allowParallel = TRUE), data=training)
```

### Evaluate performance of the trained model on the cross validation data set

```r
confusionMatrix(validating$classe, predict(modFit, validating))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 558   0   0   0   0
##          B   0 378   1   0   0
##          C   0   1 341   0   0
##          D   0   0   2 314   0
##          E   0   0   0   2 366
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.993, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.991    0.994    1.000
## Specificity             1.000    0.999    0.999    0.999    0.999
## Pos Pred Value          1.000    0.997    0.997    0.994    0.995
## Neg Pred Value          1.000    0.999    0.998    0.999    1.000
## Prevalence              0.284    0.193    0.175    0.161    0.186
## Detection Rate          0.284    0.193    0.174    0.160    0.186
## Detection Prevalence    0.284    0.193    0.174    0.161    0.187
## Balanced Accuracy       1.000    0.998    0.995    0.996    0.999
```

All of the classes are well represented in the cross validation data
set. Accuracy of 0.997 indicates that the model is working very well.


### Finally predict the activity class for the samples in the testing set and write them to files for uploading to coursera

```r
x <- predict(modFit, testing)
x
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
n = length(x)
for(i in 1:n){ 
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
```


[har]: http://groupware.les.inf.puc-rio.br/har
