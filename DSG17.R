# Load data and libraries -------------------------------------------------
library(e1071)
library(magrittr)
library(data.table)
library(h2o)
library(caret)
library(stringr)
library(xgboost)
library(dummies)
library(parallel)
library(Matrix)
library(ROCR)

setwd("H:/Data Science Competitons/DSG17")
train <- read.csv("train.csv")
test <- read.csv("test.csv")
sample_submission_kaggle <- read.csv("sample_submission_kaggle.csv")

### lowering the header values ###
colnames(train) <- tolower(colnames(train))
colnames(test) <- tolower(colnames(test))

#### no missing values ###
## library(VIM)
## aggr_plot <- aggr(total, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(total), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

## storing target variable and combining both train and test data ###
sample_id = test$sample_id
test$sample_id = NULL
target = train$is_listened
train$is_listened = NULL
total = rbind(train, test)
str(total)
names(total)
summary(total)



#total$genre_id = NULL
#total$user_id = NULL
#total$artist_id = NULL
#total$media_id = NULL
#total$album_id = NULL

#feature engineering ------------------------------------------------------------------------------------


#Weekdays
date=substr(total$release_date ,1,8)
days<-weekdays(as.Date(date,"%Y%m%d"))
total$weekday=days

total$day_type="working day"
total$day_type[total$weekday=="Saturday" | total$weekday=="Sunday"]= "holiday"



total$month = substring(total$release_date, 5,6)
total$day = substring(total$release_date, 7,8)
total$year = substring(total$release_date, 1,4)



total$age_cat[total$user_age <=18] <- 1 
total$age_cat[total$user_age >18 & total$user_age <=21 ] <- 2
total$age_cat[total$user_age >21 & total$user_age <=24 ] <- 3 
total$age_cat[total$user_age >24 & total$user_age <=27 ]  <- 4
total$age_cat[total$user_age >27 & total$user_age <=30 ]  <- 5

total$age_cat <- as.factor(total$age_cat)

time = as.POSIXct(total$ts_listen, origin="1970-01-01")
total$time = time


total$ts_listen = NULL

total$listen_date <- substring(total$time,1,10)

total$time_diff <- as.Date(as.character(total$listen_date),
                    format="%Y-%m-%d")-as.Date(as.character(total$release_date), format="%Y%m%d")

date=substr(total$listen_date ,1,10)
days<-weekdays(as.Date(date,"%Y-%m-%d"))
total$weekday_listen=days

total$day_type_listen="working day"
total$day_type_listen[total$weekday=="Saturday" | total$weekday=="Sunday"]= "holiday"

total$listen_date <- NULL
total$release_date <- NULL

total$month_listen = substring(total$time, 6,7)
total$day_listen = substring(total$time, 9,10)
total$year_listen = substring(total$time, 1,4)

total$time_listen = substring(total$time, 12,19)

total$time = NULL

total$time_hour = substring(total$time_listen, 1,2)
total$time_min = substring(total$time_listen, 4,5)
total$time_sec = substring(total$time_listen, 7,8)

total$time_listen =NULL



### categorizing time variable--------------------------------------------------------------------
total$hour_cat[total$time_hour >=1 & total$time_hour < 6 ] <- 1
total$hour_cat[total$time_hour >=06 & total$time_hour < 12 ] <- 2 
total$hour_cat[total$time_hour >=12 & total$time_hour < 18 ]  <- 3
total$hour_cat[total$time_hour >=18 & total$time_hour < 24 ]  <- 4
total$hour_cat[total$time_hour ==24 ]  <- 5

#into factor-------
for(i in colnames(total)[sapply(total, is.character)])
  set(x = total, j = i, value = as.factor(total[[i]]))

total$hour_cat <- as.factor(total$hour_cat)
total$time_hour <- as.factor(total$time_hour)
total$time_min <- as.factor(total$time_min)
total$time_sec <- as.factor(total$time_sec)

#Log Transformation--------------------------------------------------------------------------------
total$log_user <- log(total$user_id)

sapply(total,class)
# One Hot Encoding -------------------------------------------------------------------------------
total_with_dummy <- dummy.data.frame(total, names=c("hour_cat","weekday_listen","day_type","day_type_listen","weekday","age_cat","platform_name","platform_family","listen_type","user_gender"), sep="_")


for(i in colnames(total_with_dummy)[sapply(total_with_dummy, is.integer)])
  set(x = total_with_dummy, j = i, value = as.numeric(total_with_dummy[[i]]))

# spliting the data in original format------------------------------------------------------------ 
training_with_dummy = total_with_dummy[1:nrow(train),]
testing_with_dummy = total_with_dummy[1911028:nrow(total_with_dummy),]

#Missing imputation in target variable -------
target[is.na(target)] <- 1
sapply(total_with_dummy,class)

### pre-processing ends
### xgboost model classification -------------------
### getting the output of 5 diff xgb models ----------------

eta <- c(0.01,0.01,0.05,0.01,0.1)
max_depth <- c(6,6,6,6,6)
subsample <- c(0.8,0.9,0.9,0.95,0.8)
colsample_bytree <- c(0.7,0.7,0.8,0.7,0.7)
nrounds <- c(300,300,200,300,100)
predicted_train = {}
predicted_test = {}

for (i in 1:5)
  
{
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eta                 = eta[i],
                  max_depth           = max_depth[i] ,
                  subsample           = subsample[i],
                  colsample_bytree    = colsample_bytree[i]
                    
  )
  
  set.seed(600)
  ### creating the model ###
  xgb_model <- xgboost(   params              = param, 
                          data                = data.matrix(training_with_dummy),
                          label               = data.matrix(target),
                          nrounds             = nrounds[i], 
                          verbose             = 1,
                          maximize            = FALSE,
                          eval_metric         = "auc",
                          missing=NA
  )
  
  prediction_test <- predict(xgb_model,data.matrix(testing_with_dummy),missing=NA)
  predicted_test<-cbind(predicted_test,prediction_test)
  
}
predicted_test <- data.frame(predicted_test)
pred_max <- apply(predicted_test,1,max)
final_pred <- rowMeans(predicted_test)
mysolution = data.frame( sample_id = sample_id, is_listened =pred_max, stringsAsFactors = FALSE)
submission_xgb = mysolution
write.csv(submission_xgb, file = "xgb_ensemble.csv", row.names = FALSE)

#Training and validation in xgb-----------------------------------------------------------------
training_with_dummy$target <- target
x <- createDataPartition(y = training_with_dummy$target,p = 0.65,list = F)
dtrain <- training_with_dummy[x,]
dval <- training_with_dummy[-x,]
t1 <- dtrain$target
t2 <- dval$target
dtrain$target <- NULL
dval$target <- NULL
dtrain <- xgb.DMatrix(data = data.matrix(dtrain),label=data.matrix(t1),missing = NA)
dval <- xgb.DMatrix(data = data.matrix(dval),label=data.matrix(t2),missing = NA)
dtest <- xgb.DMatrix(data = data.matrix(testing_with_dummy),missing = NA)

##XGB Model-------------------------------------------------------------------------------------

watchlist <- list(val=dval,train=dtrain)
xgb_params <- list(
  seed = 1101,
  booster= "gbtree",
  colsample_bytree = 0.7,
  subsample=0.8,
  eta=0.1,
  objective="binary:logistic",
  max_depth=6
)
bst1 <- xgb.train(params = xgb_params,
                  data = dtrain,
                  nrounds = 1000, #set num_rounds from cv
                  watchlist = watchlist,
                  eval_metric="auc",
                  verbose = 1,
                  maximize = F)
#Prediction and submission-------------------------------------------------------------------
pred <- predict(bst1, dtest)
mysolution = data.frame( sample_id = sample_id, is_listened =pred, stringsAsFactors = FALSE)
submission_xgb = mysolution
write.csv(submission_xgb, file = "xgb16.csv", row.names = FALSE)

#importance of features----------------------------------------------------------------------
vimp <- xgb.importance(model = bst1,feature_names = names(training_with_dummy))
gp = xgb.plot.importance(vimp)
View(gp)

##-----Grid Search & Ensembling --------###
prediction_test = {}
params <- expand.grid(max_depth = c(4,6,8,10,12), eta = c(0.3,0.1,0.05,0.01), 
                      min_child_weight = c(1,10), subsample = c(1,0.5))

for (k in 1:nrow(params)) {
  prm <- params[k,]
  print(prm)
  print(system.time({
    n_proc <- detectCores()
    md <- xgb.train(data = dtrain, nthread = n_proc, 
                    objective = "binary:logistic", nround = 1000, 
                    max_depth = prm$max_depth, eta = prm$eta, 
                    min_child_weight = prm$min_child_weight, subsample = prm$subsample, 
                    watchlist = list(valid = dval, train = dtrain), eval_metric = "auc",
                    early_stop_round = 100)
  }))
  phat <- predict(md, newdata = dtest)
  pred <- cbind(prediction_test,phat)
}
pred_max <- apply(pred,1,max)
pred_avg <- rowMeans(pred)
#pred_max
mysolution = data.frame( sample_id = sample_id, is_listened =pred_max, stringsAsFactors = FALSE)
submission_xgb = mysolution
write.csv(submission_xgb, file = "xgb16_max.csv", row.names = FALSE)
#pred_avg
mysolution1 = data.frame( sample_id = sample_id, is_listened =pred_avg, stringsAsFactors = FALSE)
submission_xgb1 = mysolution1
write.csv(submission_xgb1, file = "xgb16_avg.csv", row.names = FALSE)




##----Grid search set up for hyperparameter tuning a xgboost model---##
xgb_grid_1 = expand.grid(
  nrounds=1000,
  eta = c(0.3,0.1,0.03,0.01),
  max_depth = c(2,4,6,8,10,12,14),
  gamma = 1,
  colsample_bytree = 0.5,
  min_child_weight = c(1, 10),
  subsample = 0.5
  
)

# set the train control parameter
xgb_train_control = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

set.seed(12345)
xgb_tune <-  train(x = data.matrix(training_with_dummy[,-57]),
                   y =  factor(target, labels = c("yes", "no")),
                   method = "xgbTree",
                   trControl = xgb_train_control,
                   tuneGrid = xgb_grid_1,
                   verbose = TRUE,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic"
)


#H2O-----------------------------------------------------------------------------------------
#Give H2o your maximum memory for computation
#if you laptop is 8GB, give atleast 6GB, close all other apps while computation happens
#may be, go out take a walk! 
library(h2o)
h2o.init(nthreads = -1,max_mem_size = "10G") 

h2o_train <- as.h2o(training_with_dummy)
h2o_test <- as.h2o(testing_with_dummy)

h2o_train$target <- h2o.asfactor(h2o_train$target)


# Create a validation frame -----------------------------------------------

#Here I want to avoid doing k-fold CV since data set is large, it would take longer time
#hence doing hold out validation

xd <- h2o.splitFrame(h2o_train,ratios = 0.6)

split_val <- xd[[2]]

y <- "target"
x <- setdiff(colnames(training_with_dummy), c(y))



# Training a GBM Model ----------------------------------------------------

gbm_clf <- h2o.gbm(x = x
                   ,y = y
                   ,training_frame = h2o_train
                   ,validation_frame = split_val
                   ,ntrees = 1000, 
                   max_depth = 4, 
                   learn_rate = 0.01, 
                   seed = 1122
                   
                   
)

gbm_clf #Validation Accuracy = 0.9858

gbm_clf_pred <- as.data.table(h2o.predict(gbm_clf_3,h2o_test))
head(gbm_clf_pred,10)

sub_pred1 <- data.table(member_id = test$member_id, loan_status = gbm_clf_pred$p1)
fwrite(sub_pred1,"h2o_gbm_sub_pred1.csv") #0.936 leaderboard score

