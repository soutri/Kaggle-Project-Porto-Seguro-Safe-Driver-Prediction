library(xgboost)
library(ROSE)
library(rpart)
library(randomForest)
library(ggplot2)
library("magrittr")
library("dplyr")
library(reshape2)
library(readr)

#Data Import
train <- read.csv("I:/AML Project/train.csv")
test <- read.csv("I:/AML Project/test.csv")


#-1 as NA values
train[train == -1] = NA
test[test == -1] = NA

#Column-wise NA values
#colSums(is.na(train)) 
#colSums(is.na(test))

train.missing = sort(colSums(is.na(train)),decreasing = TRUE)
test.missing = sort(colSums(is.na(test)),decreasing = TRUE)

train.missing = train.missing[train.missing > 0]
test.missing = test.missing[test.missing > 0]

train.missing = round(train.missing / nrow(train) * 100,3)
test.missing = round(test.missing / nrow(test) * 100,3)

#Drop features with more than 40% Missing Data
columnNames = colnames(train)
remove = names(train.missing[1:2])

columnNames = columnNames[! columnNames %in% remove]
train = as.data.frame(train[,columnNames])

columnNames = columnNames[! columnNames %in% 'target']
test = as.data.frame(test[,columnNames])



#Replacing missing data from numerical continuous fetures with mean of whole dataset

train$ps_reg_03[is.na(train$ps_reg_03)] = mean(c(mean(train$ps_reg_03,na.rm = TRUE),mean(test$ps_reg_03,na.rm = TRUE)))
test$ps_reg_03[is.na(test$ps_reg_03)] = mean(c(mean(train$ps_reg_03,na.rm = TRUE),mean(test$ps_reg_03,na.rm = TRUE)))

train$ps_car_14[is.na(train$ps_car_14)] = mean(c(mean(train$ps_car_14,na.rm = TRUE),mean(test$ps_car_14,na.rm = TRUE)))
test$ps_car_14[is.na(test$ps_car_14)] = mean(c(mean(train$ps_car_14,na.rm = TRUE),mean(test$ps_car_14,na.rm = TRUE)))

train$ps_car_12[is.na(train$ps_car_12)] = mean(c(mean(train$ps_car_12,na.rm = TRUE),mean(test$ps_car_12,na.rm = TRUE)))
test$ps_car_12[is.na(test$ps_car_12)] = mean(c(mean(train$ps_car_12,na.rm = TRUE),mean(test$ps_car_12,na.rm = TRUE)))

#Replacing missing data from binary fetures with mode of whole dataset

Mode = function (x, na.rm) {
  xtab = table(x)
  xmode = names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

train$ps_car_07_cat[is.na(train$ps_car_07_cat)] = Mode(train$ps_car_07_cat)
test$ps_car_07_cat[is.na(test$ps_car_07_cat)] = Mode(test$ps_car_07_cat)

train$ps_ind_05_cat[is.na(train$ps_ind_05_cat)] = Mode(train$ps_ind_05_cat)
test$ps_ind_05_cat[is.na(test$ps_ind_05_cat)] = Mode(test$ps_ind_05_cat)

train$ps_car_09_cat[is.na(train$ps_car_09_cat)] = Mode(train$ps_car_09_cat)
test$ps_car_09_cat[is.na(test$ps_car_09_cat)] = Mode(test$ps_car_09_cat)

train$ps_ind_02_cat[is.na(train$ps_ind_02_cat)] = Mode(train$ps_ind_02_cat)
test$ps_ind_02_cat[is.na(test$ps_ind_02_cat)] = Mode(test$ps_ind_02_cat)

train$ps_car_01_cat[is.na(train$ps_car_01_cat)] = Mode(train$ps_car_01_cat)
test$ps_car_01_cat[is.na(test$ps_car_01_cat)] = Mode(test$ps_car_01_cat)

train$ps_ind_04_cat[is.na(train$ps_ind_04_cat)] = Mode(train$ps_ind_04_cat)
test$ps_ind_04_cat[is.na(test$ps_ind_04_cat)] = Mode(test$ps_ind_04_cat)

train$ps_car_02_cat[is.na(train$ps_car_02_cat)] = Mode(train$ps_car_02_cat)
test$ps_car_02_cat[is.na(test$ps_car_02_cat)] = Mode(test$ps_car_02_cat)

train$ps_car_11[is.na(train$ps_car_11)] = Mode(train$ps_car_11)
test$ps_car_11[is.na(test$ps_car_11)] = Mode(test$ps_car_11)

train_train_data = as.data.frame(train[1:500000,1:56])
test_train_data = as.data.frame(train[500001:nrow(train),1:56])
train_train_label = train[1:500000,57]
test_train_label = train[500001:nrow(train),57]
train_train = cbind(train_train_data,train_train_label)

logistic.fit = glm(train_train_label ~ ., data = train_train, family = binomial)
prediction = predict.glm(logistic.fit,test_train_data)

roc.curve(test_train_label,prediction)

#Area under the curve (AUC): 0.500


#Data balancing - over sampling, under sampling, both, synthetic data generation
train_balance_over = ovun.sample(train_train_label ~ ., data = train_train, method = "over",N = 1100000)
table(train_balance_over$data$train_label)

train_balance_under = ovun.sample(train_label ~ ., data = train_train, method = "under", N=43000, seed = 1)
table(train_balance_under$data$train_label)

train_balance_both =  ovun.sample(train_label ~ ., data = train_train, method = "both", p=0.5, N=100000, seed = 1)
table(train_balance_both$data$train_label)


#Fit model in tree

treeimb_over = rpart(train_label ~ ., data = train_balance_over$data)
treeimb_under = rpart(train_label ~ ., data = train_balance_under$data)
treeimb_both = rpart(train_label ~ ., data = train_balance_both$data)
#treeimb_rose = rpart(train_label ~ ., data = train_balance_rose$data)

#Prediction
pred_over = predict(treeimb_over, newdata = test_train_data)

pred_under = predict(treeimb_under, newdata = test_train_data)

pred_both = predict(treeimb_both, newdata = test_train_data)

#pred_rose = predict(treeimb_rose, newdata = test_train_data)


roc.curve(test_train_label,pred_over)
#Area under the curve (AUC): 0.556

roc.curve(test_train_label,pred_under)

roc.curve(test_train_label,pred_both)
#roc.curve(test_train_label,pred_rose)
