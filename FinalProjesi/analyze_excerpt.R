# GEREKLİ KÜTÜPHANELER
# library(tm)
# library(SnowballC)
# library(wordcloud)
# library(RColorBrewer)
# library(dplyr)
# library(ggplot2)
# library(tidytext)
# library(textdata)
# library(lubridate)

# Gerekli paketleri tanımla
packages <- c(
    "tm", "SnowballC", "wordcloud", "wordcloud2", "RColorBrewer", "dplyr",
    "ggplot2", "tidytext", "textdata", "lubridate"
)

# Eksik paketleri belirle ve yükle
missing_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(missing_packages) > 0) {
    install.packages(missing_packages)
}

# Paketleri yükle
lapply(packages, library, character.only = TRUE)

# VERİYİ YÜKLE
# rawDataURL <- "https://raw.githubusercontent.com/bymayfe/r_verimadenciligi/refs/heads/main/FinalProjesi/stackoverflow_data_2025-05-10_16-24-28.csv"
rawDataURL <- "https://raw.githubusercontent.com/bymayfe/r_verimadenciligi/refs/heads/main/FinalProjesi/stackoverflow_data_2025-05-13_09-46-22.csv"
data <- read.csv(rawDataURL, stringsAsFactors = FALSE)

#-------------------------------
# 1. METİN TEMİZLEME ve KÜTÜPHANE HAZIRLIĞI
#-------------------------------
corpus <- VCorpus(VectorSource(data$excerpt))


corpus <- corpus %>%
    tm_map(content_transformer(tolower)) %>% # Metindeki tüm harfleri küçük harfe çevirir
    tm_map(content_transformer(function(x) gsub("[[:punct:]]+", " ", x))) %>% # Noktalama işaretlerini boşlukla değiştir
    tm_map(removeNumbers) %>% # Sayıları kaldırır
    tm_map(removeWords, stopwords("english")) %>% # İngilizce durak kelimeleri (stopwords) kaldırır
    tm_map(removeWords, c("m", "s", "t", "can", "just", "like", "com", "got")) %>% # Belirtilen kelimeleri kaldırır
    tm_map(stripWhitespace) # Fazladan boşlukları temizler
#-------------------------------
# 2. KELİME FREKANSI ANALİZİ
#-------------------------------
dtm <- DocumentTermMatrix(corpus)
freq <- colSums(as.matrix(dtm))
word_freq <- sort(freq, decreasing = TRUE)

# En sık geçen 20 kelimeyi çubuk grafikle görselleştir
word_df <- data.frame(word = names(word_freq), freq = word_freq)
top_words <- head(word_df, 20)

ggplot(top_words, aes(x = reorder(word, freq), y = freq)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "En Sık Geçen 20 Kelime", x = "Kelime", y = "Frekans")

#-------------------------------
# 3. KELİME BULUTU (WORDCLOUD)
#-------------------------------
set.seed(1071)
wordcloud(names(word_freq), word_freq, max.words = 300, colors = brewer.pal(8, "Dark2"))

# Wordcloud2 için veri formatını düzenleyelim
wordcloud2_data <- data.frame(word = names(word_freq), freq = word_freq)

# Wordcloud2 ile kelime bulutu oluşturma
wordcloud2(wordcloud2_data, size = 1.7, shape = "circle", color = "random-light", backgroundColor = "black")


#-------------------------------
# 4. N-GRAM (4'lü KELİME GRUPLARI) ANALİZİ
#-------------------------------
text_df <- tibble(text = data$excerpt)

bigrams <- text_df %>%
    unnest_tokens(bigram, text, token = "ngrams", n = 4)

bigrams %>%
    count(bigram, sort = TRUE) %>%
    filter(n > 10) %>%
    top_n(20) %>%
    ggplot(aes(x = reorder(bigram, n), y = n)) +
    geom_col(fill = "purple") +
    coord_flip() +
    labs(title = "En Sık Geçen 4'lü Kelime Grupları", x = "Bigram", y = "Frekans")

#-------------------------------
# 5. DUYGU ANALİZİ (SENTIMENT ANALYSIS)
#-------------------------------
bing <- get_sentiments("bing")

# Ortak kelime temizliği
clean_words <- text_df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  filter(!word %in% c(
    "flutter", "android", "java", "html", "css", "php",
    "react", "c", "cpp", "c++", "excel", "swift", "static", "plot", "cloud", "object", "dynamic"
  ))

# Pozitif & negatif kelimeler
sentiment_words <- clean_words %>%
  inner_join(bing, by = "word")

# Nötr kelimeler
neutral_words <- clean_words %>%
  anti_join(bing, by = "word") %>%
  mutate(sentiment = "neutral")

# Hepsini birleştir
all_sentiments <- bind_rows(sentiment_words, neutral_words)

# Frekans tablosu
all_sentiments %>%
  count(sentiment, sort = TRUE)

# Görselleştirme
all_sentiments %>%
  count(word, sentiment, sort = TRUE) %>%
  group_by(sentiment) %>%
  top_n(10, n) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(word, n), y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  coord_flip() +
  labs(title = "Pozitif, Negatif ve Nötr Kelimeler", x = "Kelime", y = "Frekans")

#-------------------------------
# 6. SAATE GÖRE SORU DAĞILIMI
#-------------------------------
data$time_asked <- as.POSIXct(data$time_asked)
data$hour <- hour(data$time_asked)

ggplot(data, aes(x = hour)) +
    geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
    labs(title = "Saatlik Soru Dağılımı", x = "Saat", y = "Soru Sayısı")

#-------------------------------
# 7. BAŞLIKLARDA TF-IDF ANALİZİ
#-------------------------------
title_df <- tibble(line = 1:nrow(data), text = data$title)

title_words <- title_df %>%
    unnest_tokens(word, text) %>%
    count(line, word, sort = TRUE)

tf_idf <- title_words %>%
    bind_tf_idf(word, line, n) %>%
    arrange(desc(tf_idf))

# En anlamlı 10 kelime
head(tf_idf, 10)
