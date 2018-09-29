> data = read.csv(file.choose())
> View(data)
> library(ggplot2)
> attach(data)
> 
> data$avg_grade=(G1+G2+G3)/3
> ggplot(data, aes(x=Dalc, color=Dalc,fill=Dalc)) +geom_histogram(binwidth=0.1)
> ggplot(data, aes(x=Dalc, y=data$avg_grade, group=Dalc))+geom_boxplot()+theme(legend.position="none")+scale_fill_manual(values=waffle.col)+ xlab("Daily Alcohol consumption")+ ylab("Average Grades")
> ggplot(data, aes(x=health, fill=sex)) + 
     geom_bar(stat="count", position="dodge", color="black")
> ggplot(data, aes(x=data$avg_grade, fill=sex)) + 
     geom_bar(stat="count", position="dodge", color="black")
> ggplot(data, aes(x=health, color=factor(Dalc),fill=factor(Dalc))) + 
       geom_histogram(binwidth = 0.1)
> ggplot(data, aes(x=age, color=factor(Dalc),fill=factor(Dalc))) + 
    geom_histogram(binwidth = 0.1)
> grade_romantic <- aggregate(data$avg_grade, by=list(data$romantic), FUN=mean)
> ggplot(grade_romantic, aes(Group.1, x))+geom_bar(stat="identity")+xlab("Romantic_Relationship")+ylab("Avg Grade")
> ggplot(data, aes(x=Dalc, color=factor(romantic),fill=factor(romantic))) + 
    geom_histogram(binwidth = 0.1)
> grade_internet <- aggregate(data$avg_grade, by=list(data$internet), FUN=mean)
> ggplot(grade_internet, aes(Group.1, x))+geom_bar(stat="identity")+xlab("internet_use")+ylab("Avg Grade")
> ggplot(data, aes(x=Walc, color=factor(sex),fill=factor(sex))) + 
    geom_histogram(binwidth = 0.1)
> ggplot(data, aes(x=Dalc, color=factor(sex),fill=factor(sex))) + 
      geom_histogram(binwidth = 0.1)
> ggplot(data, aes(x = Dalc, y = age, color = sex)) + geom_count() + xlab("Daily alcohol consumption")
> t.test(data$avg_grade~data$romantic, data=data)
> t.test(data$avg_grade~data$internet, data=data)
