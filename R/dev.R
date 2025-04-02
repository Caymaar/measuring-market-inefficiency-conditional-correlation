# Installer les packages si besoin
# install.packages("pracma")
# install.packages("zoo")

library(pracma)
library(zoo)

# Chargement des données
data <- read.csv("s&p500.csv", stringsAsFactors = FALSE)
# Convertir la colonne Date au format Date (adapter le format si nécessaire)
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")

# Calcul du logarithme du S&P500
ts_data <- log(as.numeric(as.character(data$s.p500)))

# Définir la fonction qui calcule l'exposant de Hurst avec la méthode de variance agrégée ("aggvar")
compute_hurst <- function(x) {
  h_est <- hurstexp(x)
  # Extraire la valeur pour "aggvar" et la convertir en numérique pur
  return(as.numeric(h_est["aggvar"]))
}


# Calcul de l'exposant de Hurst sur une fenêtre glissante de 10 jours
rolling_hurst <- rollapply(ts_data, 
                           width = 10, 
                           FUN = compute_hurst, 
                           by = 1, 
                           align = "right", 
                           fill = NA)

# Calcul de l'indice d'inefficacité : HEI = 0.5 - h
ineff_index <- 0.5 - rolling_hurst

# Création d'un data.frame avec les dates, l'exposant de Hurst et l'indice d'inefficacité
results <- data.frame(Date = data$Date, Hurst = rolling_hurst, Inefficiency = ineff_index)

# Sous-ensemble de la période 2007 à 2009
subset_results <- subset(results, Date >= as.Date("2007-01-01") & Date <= as.Date("2009-12-31"))

# Visualisation graphique de l'indice d'inefficacité sur la période 2007-2009
plot(subset_results$Date, subset_results$Inefficiency, type = "l",
     main = "Indice d'inefficacité (HEI) sur la période 2007-2009",
     xlab = "Date", ylab = "Indice d'inefficacité (0.5 - Hurst)")