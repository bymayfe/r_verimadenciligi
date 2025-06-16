library(rvest)
library(httr)
library(dplyr)
library(stringr)

# Kullanıcı ajanı
user_agent <- "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
all_data <- data.frame()

for (page_num in 1:1000) { # 1000 sayfa x ~15 soru = 15000+ veri
    url <- paste0("https://stackoverflow.com/questions?tab=newest&page=", page_num)
    response <- GET(url, user_agent(user_agent))

    if (http_error(response)) {
        message(paste("Sayfa", page_num, "çekilemedi. Atlınıyor..."))
        next
    }

    page <- read_html(response)

    questions <- page %>% html_nodes(".s-post-summary")

    for (q in questions) {
        title <- q %>%
            html_node(".s-post-summary--content-title a") %>%
            html_text(trim = TRUE)
        excerpt <- q %>%
            html_node(".s-post-summary--content-excerpt") %>%
            html_text(trim = TRUE)

        stats <- q %>%
            html_nodes(".s-post-summary--stats-item-number") %>%
            html_text(trim = TRUE)
        votes <- as.integer(stats[1])
        answers <- as.integer(stats[2])
        views <- as.integer(gsub("[^0-9]", "", stats[3]))

        tags <- q %>%
            html_nodes(".post-tag") %>%
            html_text(trim = TRUE)
        tags_combined <- paste(tags, collapse = ", ")

        user_node <- q %>%
            html_node(".s-user-card--link")
        user_name <- user_node %>%
            html_text(trim = TRUE)

        user_reputation <- q %>%
            html_node(".s-user-card--rep") %>%
            html_text(trim = TRUE) %>%
            gsub("[^0-9]", "", .) %>%
            as.integer()

        time_asked <- q %>%
            html_node(".s-user-card--time .relativetime") %>%
            html_attr("title")

        time_asked2 <- q %>%
            html_node(".s-user-card--time .relativetime") %>%
            html_text(trim = TRUE)

        all_data <- rbind(all_data, data.frame(
            title = title,
            excerpt = excerpt,
            votes = votes,
            answers = answers,
            views = views,
            tags = tags_combined,
            user_name = user_name,
            user_reputation = user_reputation,
            time_asked = time_asked,
            time_asked2 = time_asked2,
            stringsAsFactors = FALSE
        ))
    }

    print(paste("Sayfa", page_num, "çekildi. Toplam kayıt:", nrow(all_data)))
    Sys.sleep(1) # Ban riskini azaltmak için bekleme
}

# Tarih ve saat bilgisini al
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")

# Dosya adını oluştur
file_name <- paste0("stackoverflow_data_", timestamp, ".csv")

# CSV dosyasını kaydet
write.csv(all_data, file_name, row.names = FALSE)

# Kullanıcıya bilgi ver
print(paste("✅ Tüm veriler", file_name, "dosyasına kaydedildi."))
