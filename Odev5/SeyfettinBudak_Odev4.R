# Karar agaci ve knn odev. Hepsi R da yapilacak. Egitim ve test veri seti olarak bolunup model performans degerlendirme olcutlerine gore degerlendirilecek. Disaridan veri seti yuklenerek yapilacak.

if (!requireNamespace("rJava", quietly = TRUE)) {
    install.packages("rJava")
}
if (!requireNamespace("RWeka", quietly = TRUE)) {
    install.packages("RWeka")
}
if (!requireNamespace("partykit", quietly = TRUE)) {
    install.packages("partykit")
}
if (!requireNamespace("cluster", quietly = TRUE)) {
    install.packages("cluster")
}
if (!requireNamespace("gmodels", quietly = TRUE)) {
    install.packages("gmodels")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
    install.packages("dplyr")
}
if (!requireNamespace("caret", quietly = TRUE)) {
    install.packages("caret")
}
if (!requireNamespace("class", quietly = TRUE)) {
    install.packages("class")
}


# gerekli paketler yükleniyor
# library(rJava) # J48 için
library(RWeka) # J48 için
library(partykit) # plot için

library(caret) # confusionMatrix için
library(dplyr) # revalue için

library(class) # knn için
library(cluster) # agnes için
library(gmodels)
library(ggplot2) # ggplot için

# Veri seti yükleniyor
# data <- read.table(file.choose(), sep = ",", header = TRUE)
rawdata <- read.csv("Depression.csv", sep = ",")
print(rawdata)
str(rawdata)
# summary(rawdata)

# verinin stun isimleri degistiriliyor
colnames(rawdata) <- c("age", "gender", "univ", "dept", "acad_yr", "cgpa", "waiv", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "dep_val", "dep_lab")
str(rawdata)

# rasgelelik deneyen herkeste ayni sonucu almak icin set.seed() fonksiyonu kullaniliyor
set.seed(2004)
# veriden rasgele 100 satır seciliyor
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
soru <- data.frame(q1 = 4, q2 = 3, q3 = 4, q4 = 5, q5 = 4, q6 = 3, q7 = 4, q8 = 5, q9 = 4, )

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



# veri setini egitim ve test veri seti olarak ayiralim
# %70 egitim %30 test olacak sekilde

set.seed(2004)
egitimindisleri <- createDataPartition(y = data$dep_lab, p = .70, list = F)

egitimindisleri
head(egitimindisleri)


egitim <- datan[egitimindisleri, ]
test <- datan[-egitimindisleri, ]

str(test)

testnitelikleri <- test[, -9]
testhedefnitelik <- test[[9]]

egitimnitelikleri <- egitim[, -9]
egitimhedefnitelik <- egitim[[9]]

# k degeri 1 den 10 a kadar deneyelim
k_degeri <- 10
dogruluk <- NULL


for (i in 1:k_degeri)
{
    set.seed(2004)
    (tahminisiniflar <- knn(egitimnitelikleri, testnitelikleri, egitimhedefnitelik, k = i))
    dogruluk[i] <- mean(tahminisiniflar == testhedefnitelik)
    dogruluk[i] <- round(dogruluk[i], 2)
}


for (i in 1:k_degeri) {
    print(paste("k=", i, "icin elde edilen dogruluk=", dogruluk[i]))
}

tablom <- table(tahminisiniflar, testhedefnitelik, dnn = c("tahmini siniflar", "gercek siniflar"))
tablom


# buradan devam et
veriler_egitim <- datan[1:65, ]
veriler_test <- datan[66:100, ]

# hedef degiskeni de ekleyelim
str(rawdata)
veriler_egitim_labels <- rawdata[1:65, 18]
veriler_test_labels <- rawdata[66:100, 18]

# knn icin

# test data setini siniflandiralim
veriler_test_tahmini <- knn(train = veriler_egitim, test = veriler_test, cl = veriler_egitim_labels, k = 10)
print(veriler_test_tahmini)


CrossTable(x = veriler_test_labels, y = veriler_test_tahmini, prop.chisq = FALSE)
