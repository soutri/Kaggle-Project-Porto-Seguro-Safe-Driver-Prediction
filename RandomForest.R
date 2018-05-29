#install.packages("magrittr")
#install.packages("dplyr")
#install.packages("reshape2")
#install.packages("ggplot2")
#install.packages("randomForest")
library(rpart)
library(randomForest)
library(ggplot2)
library("magrittr")
library("dplyr")
library(reshape2)
library(readr)

#Data Import
train = read_csv("C:/Users/meeta/Desktop/Ashay stuff/Fall 2017/Term Project/AML/Dataset/train.csv")
test = read_csv("C:/Users/meeta/Desktop/Ashay stuff/Fall 2017/Term Project/AML/Dataset/test.csv")


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

#train.missing
png(filename="train.missing.png")
barplot(train.missing,main = "Missing Values per feature Training Data",ylab = "Number of Missing Values",ylim = c(0,80),las = 2,mgp = c(2.5,0,0))
dev.off()

#test.missing
png(filename="test.missing.png")
barplot(test.missing,main = "Missing Values per feature Testing Data",ylab = "Number of Missing Values",ylim = c(0,80),las = 2,mgp = c(2.5,0,0))
dev.off()

# Following are the features with respective % of missing values in them (Train Data)
#ps_car_03_cat = 69.09
#ps_car_05_cat = 44.78
#ps_reg_03 = 18.11
#ps_car_14 = 7.16
#ps_car_07_cat = 1.93
#ps_ind_05_cat = 0.98

# Following are the features with respective % of missing values in them (Test Data)
#ps_car_03_cat = 69.10
#ps_car_05_cat = 44.84
#ps_reg_03 = 18.11
#ps_car_14 = 7.15
#ps_car_07_cat = 1.94
#ps_ind_05_cat = 0.98

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

test_id = test[,1]
test = test[,2:ncol(test)]
train_label = train$target
train = train[,3:ncol(train)]

common_data = rbind(train,test)
for (i in 1:ncol(common_data)) {
  if(is.numeric(common_data[,i])){
    #Do Nothing
  }else{
    common_data[,i] = as.numeric(common_data[,i])
  }
}

cormat = round(cor(common_data),2)


# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}


upper_tri = get_upper_tri(cormat)

# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
# Print the heatmap
print(ggheatmap)

columnNames = colnames(common_data)
remove = columnNames[grepl("calc",colnames(common_data))]

columnNames = columnNames[! columnNames %in% remove]

train = as.data.frame(train[,columnNames])

test = as.data.frame(test[,columnNames])

common_data = rbind(train,test)

for (i in 1:35) {
  if(is.numeric(common_data[,i])){
    
  }else{
    common_data[,i] = as.numeric(common_data[,i])
  }
}

common_data = scale(common_data)

train = common_data[1:length(train_label),]
test = common_data[(length(train_label) + 1):nrow(common_data),]

train = cbind(train,train_label)

#Class imbalance
sum(train_label[train_label == 1])/length(train_label) *100


train_train = train[1:500000,]

test_train_data = train[500000:nrow(train),1:35]
test_train_label = train[500000:nrow(train),36]

#Random Forest for regression and finding feature importace
model.rf=randomForest(train_label ~ . , data = train_train,ntree = 100,nodesize = 5,mtry = 10)

prediction_rf = predict(model.rf,newdata = test_train_data,type = 'response')

roc.curve(test_train_label,prediction_rf)
#Area under the curve (AUC): 0.588

feature_importance = model.rf$importance
name = row.names(feature_importance)
name = name[match(sort(feature_importance,decreasing = TRUE)[1:20],feature_importance)]

train = as.data.frame(train[,name])
train = as.data.frame(cbind(train,train_label))
test = as.data.frame(test[,name])

#Check Logistic Regression with only 20 important features 
train_train = train[1:500000,]

test_train_data = train[500000:nrow(train),1:20]
test_train_label = train[500000:nrow(train),21]


logistic.fit = glm(train_label ~ ., data = as.data.frame(train_train), family = binomial)
fitted_results_log = predict(logistic.fit, newdata = as.data.frame(test_train_data), type = 'response')

roc.curve(test_train_label,fitted_results_log)
#Area under the curve (AUC): 0.611

f_imp = sort(feature_importance,decreasing = TRUE)[1:20]
names(f_imp) = name

barplot(f_imp,main = "20 Most Important Features using Random Forest",ylab = "Feature Importance",ylim = c(0,2200),las = 2,mgp = c(2.5,0,0))

