library(forecast)
library(TTR)

train <- read.csv("data/train_clean.csv", sep = ',')
test <- read.csv("data/test_clean.csv", sep=';')

extract_order <- function(x) {
  x <- ts(x) - mean(x)
  fit <- auto.arima(x, seasonal=FALSE, ic='bic', max.order=6)
  return(rbind(cbind(arimaorder(fit)), sqrt(fit$sigma2)))
}


bills <- cbind(train$BILL_AMT_JUL, train$BILL_AMT_AUG, train$BILL_AMT_SEP, train$BILL_AMT_OCT, train$BILL_AMT_NOV, train$BILL_AMT_DEC)
bills_order <- apply(bills, 1, FUN=extract_order)
bills_df <- data.frame(t(bills_order))
colnames(bills_df) <- c('BILL_AMT_AR', 'BILL_AMT_I', 'BILL_AMT_MA', 'BILL_AMT_SIGMA')


pay_amts <- cbind(train$PAY_AMT_JUL, train$PAY_AMT_AUG, train$PAY_AMT_SEP, train$PAY_AMT_OCT, train$PAY_AMT_NOV, train$PAY_AMT_DEC)
pay_amt_orders <- apply(pay_amts, 1, FUN=extract_order)
pay_amts_df <- data.frame(t(pay_amt_orders))
colnames(pay_amts_df) <- c('PAY_AMT_AR', 'PAY_AMT_I', 'PAY_AMT_MA', 'PAY_AMT_SIGMA')


pays <- cbind(train$PAY_JUL, train$PAY_AUG, train$PAY_SEP, train$PAY_OCT, train$PAY_NOV, train$PAY_DEC)
pay_orders <- apply(pays, 1, FUN=extract_order)
pay_df <- data.frame(t(pay_orders))
colnames(pay_df) <- c('PAY_AR', 'PAY_I', 'PAY_MA', 'PAY_SIGMA')


train_orders <- data.frame(bills_df, pay_amts_df, pay_df)
write.csv(train_orders, file='data/other/train_orders.csv', row.names=FALSE)



bills <- cbind(test$BILL_AMT_JUL, test$BILL_AMT_AUG, test$BILL_AMT_SEP, test$BILL_AMT_OCT, test$BILL_AMT_NOV, test$BILL_AMT_DEC)
bills_order <- apply(bills, 1, FUN=extract_order)
bills_df <- data.frame(t(bills_order))
colnames(bills_df) <- c('BILL_AMT_AR', 'BILL_AMT_I', 'BILL_AMT_MA', 'BILL_AMT_SIGMA')


pay_amts <- cbind(test$PAY_AMT_JUL, test$PAY_AMT_AUG, test$PAY_AMT_SEP, test$PAY_AMT_OCT, test$PAY_AMT_NOV, test$PAY_AMT_DEC)
pay_amt_orders <- apply(pay_amts, 1, FUN=extract_order)
pay_amts_df <- data.frame(t(pay_amt_orders))
colnames(pay_amts_df) <- c('PAY_AMT_AR', 'PAY_AMT_I', 'PAY_AMT_MA', 'PAY_AMT_SIGMA')


pays <- cbind(test$PAY_JUL, test$PAY_AUG, test$PAY_SEP, test$PAY_OCT, test$PAY_NOV, test$PAY_DEC)
pay_orders <- apply(pays, 1, FUN=extract_order)
pay_df <- data.frame(t(pay_orders))
colnames(pay_df) <- c('PAY_AR', 'PAY_I', 'PAY_MA', 'PAY_SIGMA')

test_orders <- data.frame(bills_df, pay_amts_df, pay_df)
write.csv(test_orders, file='data/other/test_orders.csv', row.names=FALSE)



