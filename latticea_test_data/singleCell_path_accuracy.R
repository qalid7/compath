#H&E single-cell annotations data from the LATTICe-A cohort
#KA - 20191119
library(purrr)
library(dplyr)
library(rlist)

#gt: ground truth
#dl: predictions 

#Supplementary Table 3a-b produced using this code
#change to the 'test_data' dir from: https://github.com/qalid7/compath
setwd('~/latticea_test_data')
#any Euclidean/nearest point function will do
ecudist <- function(row){
  dists <- (row[["x"]] - df2$x)^2 + (row[["y"]]- df2$y)^2
  return(cbind(df2[which.min(dists),], distance = min(dists)))
}
#paths
gt = './gt_celllabels'
dl = './dl_celllabels'

Das = dir(path=gt, pattern = '^Da*.')
csvs <- vector("list", length(Das))
names(csvs) <- Das

for (t in 1:length(Das)){
    
    DaGT <- read.csv(file.path(gt, Das[t]))
    DaDL <- read.csv(file.path(dl, Das[t]))
    print(Das[t])
    
    DaGT = DaGT[, c(2:3, 1)]
    DaDL = DaDL[, c(2:3, 1)]
    names(DaGT) <- c("x", "y", "class")
    names(DaDL) <- c("x", "y", "class")
    
    GTcells = nrow(DaGT)
    if (GTcells>0){
      df2 = DaDL
      df1 = DaGT
      DaR <- cbind(df1, do.call(rbind, lapply(1:nrow(df1), function(x) ecudist(df1[x,]))))
      names(DaR) <- c("xgt", "ygt", "classgt", "xdl", "ydl", "classdl", "distance")
 
      ## ensure no predicted cell is >5 pixels away
      #DaR[DaR$distance>5,]
      ##
      # if (nrow(DaR)>0){
         DaR$file_name = Das[t]
         csvs[[t]] <- DaR
      # }
    }
  }

#jsut to ensure no NULL
csvs = rlist::list.clean(csvs ,recursive = T)
#flatten everything into one df
combL <- bind_rows(csvs)
#or
#combL = flatten_dfr(csvs)

combL$classgt <- as.factor(as.character(combL$classgt))
combL$classdl <- as.factor(as.character(combL$classdl))
#might need to make sure the order is the same
#levels(combL$classgt) <- c("l", "t", "o", "f")
#levels(combL$classdl) <- c("f", "l", "o", "t")

confusionMatrix(combL$classdl, combL$classgt, mode="everything")

a = confusionMatrix(m1$classdl, m1$classgt, mode = "everything")
a = a$byClass
a = data.frame(t(a))
a$overall = (a$Class..l+a$Class..t+a$Class..o+a$Class..f)/4

#KA - 20200501