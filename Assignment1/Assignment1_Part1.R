#a
> setwd("/Users/saurabhthakrani/Desktop/737/all_nyt")
> data=lapply(dir(),read.csv)
> totalData=do.call("rbind",data)
> attach(totalData)
>summary(Age)
> totalData$Age_Group= cut(Age, breaks = c(0,18,25,35,45,55,65,116), labels = c('<18','18-24','25-34','35-44','45-54','55-64','65+'), right = FALSE)

#b
> d=read.csv(file.choose())
> attach(d)
> d$Age_Group= cut(Age, breaks = c(0,18,25,35,45,55,65,116), labels = c('<18','18-24','25-34','35-44','45-54','55-64','65+'), right = FALSE)
> View(d)
> install.packages("ggplot2", dependencies = TRUE)
> library(ggplot2)
> d$CTR= Clicks/Impressions
> d$CTR[is.na(d$CTR)]=0
> View(d)
> ggplot(d,aes(x=CTR,color=d$Age_Group)) + geom_density()
> ggplot(d,aes(x=Impressions,color=d$Age_Group)) + geom_density()
> ggplot(d, aes(x=d$CTR, color=d$Age_Group)) +
    geom_histogram(binwidth=0.1,fill="white")
> ggplot(d, aes(x=d$Impressions, color=d$Age_Group)) +
   geom_histogram(binwidth=0.1,fill="white")
> summary(Clicks)
> d$Clicked=ifelse(Clicks==0,0,1)
> ggplot(d, aes(x=d$Clicked, color=d$Age_Group,fill=d$Age_Group)) +geom_histogram(binwidth=0.1)
> ggplot(d,aes(Age,Clicks,color=Gender))+ geom_point(size=3)
> summary(d)
> setwd("/Users/saurabhthakrani/Desktop/737/allnyt7")
> data7=lapply(dir(),read.csv)
> totaldata7=do.call("rbind",data7)
> attach(totaldata7)
> impression_day <- aggregate(Impressions, by=list(Day),sum)
> View(impression_day)
> ggplot(impression_day, aes(Group.1, x)) +
    geom_line() +xlab("Days of a week") + ylab("Sum of Impressions") +geom_smooth(method = lm)
> clicks_day <- aggregate(Clicks, by=list(Day),sum)
> ggplot(clicks_day, aes(Group.1, x)) +
   geom_line() +xlab("Days of a week") + ylab("Sum of Clicks") +geom_smooth(method = lm)
> View(clicks_day)
> signin_day <- aggregate(Signed_In, by=list(Day),sum)
> ggplot(signin_day, aes(Group.1, x)) +
    geom_line() +xlab("Days of a week") + ylab("Sum of Sign-in") +geom_smooth(method = lm)
> View(signin_day)
> impression_mean_day <- aggregate(Impressions, by=list(Day),mean)
> ggplot(impression_mean_day, aes(Group.1, x)) +
     geom_line() +xlab("Days of a week") + ylab("Mean of Impression") +geom_smooth(method = lm)
> Clicks_mean_day <- aggregate(Clicks, by=list(Day),mean)
> ggplot(Clicks_mean_day, aes(Group.1, x)) +
   geom_line() +xlab("Days of a week") + ylab("Mean of Clicks") +geom_smooth(method = lm)
> Signin_mean_day <- aggregate(Signed_In, by=list(Day),mean)
> ggplot(Signin_mean_day, aes(Group.1, x)) +
   geom_line() +xlab("Days of a week") + ylab("Mean of Sign-ins") +geom_smooth(method = lm)


