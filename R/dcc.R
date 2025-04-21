# install.packages("rmgarch")
# install.packages("rugarch")
# install.packages("readxl")
# install.packages("dplyr")
library(rmgarch)
library(rugarch)
library(readxl)
library(dplyr)

file_path <- "output/inefficiency.xlsx"
data <- read_excel(file_path)

data <- data %>% select("ftsemib", "ftse100")

spec1 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm"
)
spec2 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm"
)

dcc_spec <- dccspec(
  uspec = multispec(replicate(2, spec1)),
  dccOrder = c(1, 1),
  distribution = "mvnorm"
)

dcc_fit <- dccfit(dcc_spec, data = as.matrix(data))

conditional_correlations <- rcov(dcc_fit)
plot(conditional_correlations, type = 'l', col = 'blue', ylab = 'Correlation conditionnelle', xlab = 'Temps')
