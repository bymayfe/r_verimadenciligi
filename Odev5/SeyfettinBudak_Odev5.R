# Navibayes
# https://github.com/bymayfe/r_verimadenciligi/tree/main/Odev5


if (!requireNamespace("rJava", quietly = TRUE)) {
    install.packages("rJava") # J48 icin (java kurulu olmasi gerekiyor)
}
if (!requireNamespace("RWeka", quietly = TRUE)) {
    install.packages("RWeka") # J48 icin
}
if (!requireNamespace("partykit", quietly = TRUE)) {
    install.packages("partykit") # plot icin
}
if (!requireNamespace("cluster", quietly = TRUE)) {
    install.packages("cluster") # agnes icin
}
if (!requireNamespace("gmodels", quietly = TRUE)) {
    install.packages("gmodels") # CrossTable icin
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
    install.packages("ggplot2") # ggplot icin
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
    install.packages("dplyr") # revalue icin
}
if (!requireNamespace("caret", quietly = TRUE)) {
    install.packages("caret") # confusionMatrix icin
}
if (!requireNamespace("class", quietly = TRUE)) {
    install.packages("class") # knn icin
}
if (!requireNamespace("e1071", quietly = TRUE)) {
    install.packages("e1071") # Naive Bayes icin
}


# gerekli paketler yukleniyor
# library(rJava) # J48 icin
library(RWeka) # J48 icin
library(partykit) # plot icin

library(caret) # confusionMatrix icin
library(dplyr) # revalue icin

library(class) # knn icin
library(cluster) # agnes icin
library(gmodels)
library(ggplot2) # ggplot icin
library(e1071) # Naive Bayes icin

# Veri seti yukleniyor
rawURL <- "https://raw.githubusercontent.com/bymayfe/r_verimadenciligi/refs/heads/main/Odev5/Depression.csv"
rawdata <- read.csv(rawURL, sep = ",", header = TRUE)
# rawdata <- read.table(file.choose(), sep = ",", header = TRUE)
# rawdata <- read.csv("Depression.csv", sep = ",")
print(rawdata)
str(rawdata)
# summary(rawdata)

# verinin stun isimleri degistiriliyor
colnames(rawdata) <- c("age", "gender", "univ", "dept", "acad_yr", "cgpa", "waiv", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "dep_val", "dep_lab")
str(rawdata)

# rasgelelik deneyen herkeste ayni sonucu almak icin set.seed() fonksiyonu kullaniliyor
set.seed(2004)
# veriden rasgele 100 satir seciliyor
ind <- sample(1:nrow(rawdata), 100)
data <- rawdata[ind, ]

print(data)
str(data)

# ilk 7 degiskenlerini kullanmayacagiz, bu nedenle cikariliyor
data <- data[, -c(1, 2, 3, 4, 5, 6, 7)]
str(data)

# veri setindeki NA degerlerini kontrol etme
anyNA(data)

# veri setindeki NA degerlerini cikar (bizim icin gerekli degil)
# data <- na.omit(data)

# veri setindeki NA degerlerini tekrar kontrol etme
# anyNA(data)

# faktor degiskeninini belirleme veya kontrol etme

str(data)

data$dep_lab <- as.factor(data$dep_lab)
data$dep_val <- as.factor(data$dep_val)
str(data)

data <- data[, -10]

# RWeka paketi icinde C4.5 algoritmasinin J48() fonksiyonu kullanilir
kararagacimodeli <- J48(dep_lab ~ ., data = data)

print(kararagacimodeli)
summary(kararagacimodeli)
plot(kararagacimodeli)

# cikan plot gorselini kaydetme
png("karar_agaci2.png", width = 1600, height = 800)
plot(kararagacimodeli)
graphics.off()


# yeni bir degisken ile tahmin yapma
soru <- data.frame(q1 = 4, q2 = 3, q3 = 4, q4 = 5, q5 = 4, q6 = 3, q7 = 4, q8 = 5, q9 = 4)

predict(kararagacimodeli, soru)


# KNK

# once verileri normalize edelim
normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
}

str(data)

# 9 numerik degiskenin hepsini birden normalize edelim
datan <- as.data.frame(lapply(data[1:9], normalize))
data.frame(datan)

str(datan)

# veri setini egitim ve test veri seti olarak ayiralim
# %70 egitim %30 test olacak sekilde

set.seed(2004)
egitimindisleri <- createDataPartition(y = data$dep_lab, p = .70, list = F)


veriler_egitim <- datan[egitimindisleri, ]
veriler_test <- datan[-egitimindisleri, ]


# datan verisinde nitelikler olmadigi icin data verisini kullanacagiz
veriler_egitim_labels <- data[egitimindisleri, 10]
veriler_test_labels <- data[-egitimindisleri, 10]


# naive bayes

naivebayes_modeli <- naiveBayes(veriler_egitim, veriler_egitim_labels)
print(naivebayes_modeli)

ongoru <- predict(naivebayes_modeli, veriler_test)
print(ongoru)

data.frame(veriler_test_labels, ongoru)

karisiklikmatrisi <- table(ongoru, veriler_test_labels, dnn = c("tahmin edilen siniflar", "gercek siniflar"))
print(karisiklikmatrisi)

# modelin performans degerlendirme olcutleri

# dogru pozitif
TP <- karisiklikmatrisi[1]

# yanlis pozitif
FP <- karisiklikmatrisi[3]

# yanlis negatif
FN <- karisiklikmatrisi[2]

# dogru negatif
TN <- karisiklikmatrisi[4]

# performans degerlendirme olcutleri
# dogruluk oranini bulalim
Dogruluk <- (TP + TN) / sum(karisiklikmatrisi)
paste0("Dogruluk = ", Dogruluk)
# Hata oranini bulalim
Hata <- 1 - Dogruluk
paste0("Hata = ", Hata)
# TPR=Duyarlilik orani
TPR <- TP / (TP + FN)
paste0("TPR= ", TPR)
# SPC=Belirleyicilik orani
SPC <- TN / (FP + TN)
paste0("SPC= ", SPC)
# PPV=kesinlik ya da pozitif ongoru degeri
PPV <- TP / (TP + FP)
paste0("PPV= ", PPV)
# NPV=negatif ongoru degeri
NPV <- TN / (TN + FN)
paste0("NPV= ", NPV)
# FPR=Yanlis pozitif orani
FPR <- FP / sum(karisiklikmatrisi)
paste0("FPR = ", FPR)
# FNR=Yanlis negatif orani
FNR <- FN / (FN + TP)
paste0("FNR=", FNR)
# F olcutu kesinlik ve duyarlilik olcutlerinin harmonik ortalamasi
# F1 skoru
F_measure <- (2 * PPV * TPR) / (PPV + TPR)
paste0("F_measure = ", F_measure)

# F2 skoru
F2 <- (5 * PPV * TPR) / (4 * PPV + TPR)
paste0("F2 = ", F2)

# F0.5 skoru
F05 <- (1.25 * PPV * TPR) / (0.25 * PPV + TPR)
paste0("F0.5 = ", F05)
