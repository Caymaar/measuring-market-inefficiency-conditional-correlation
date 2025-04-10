## Tutoriel d'installation des dépendances

### Option 1 : Installation avec pip uniquement
1. Ouvrez votre terminal.
2. Placez-vous dans le dossier du projet :
    ```
    cd /chemin/vers/le/dossier
    ```
3. Installez le package via pip :
    ```
    pip install .
    ```

### Option 2 : Installation avec uv
1. Vérifiez si uv est installé (sinon, installez-le) :
    ```
    pip install uv
    ```
2. Placez-vous dans le dossier du projet :
    ```
    cd /chemin/vers/le/dossier
    ```
3. Générez l'environnement virtuel et installez toutes les dépendances en exécutant :
    ```
    uv sync
    ```



## 1. Contexte général

L’objectif est de répliquer la méthode d’estimation du Hurst exponent local décrite dans le papier (Quantitative Finance and Economics, Volume 7, Issue 3, 491–507).

Les auteurs annoncent :
- Estimer $h(t)$ à partir de la méthode de variance (“variance method building on Cannon et al., 1997”).
- Utiliser une régression log-log pour obtenir $h$ (log(σa) - log(σ) vs log(a)).
- Appliquer une fenêtre glissante de 10 jours (k = 10 days) pour obtenir un $h$ variant dans le temps.

À première vue, cela correspond à appliquer Scaled Windowed Variance (SWV) sur des fenêtres de seulement 10 points.

## 2. Problème rencontré

- En essayant de répliquer leur approche exactement comme décrite :
    - Sur 10 points, appliquer directement SWV ou même DFA :
        - Donne des estimations de Hurst extrêmement bruitées,
        - Produit des valeurs de h instables,
        - Et parfois hors bornes plausibles (h < 0 ou h > 1).
    - Les tailles de fenêtres disponibles pour faire du scaling sont très faibles (n = 2, 4, 5, 8), ce qui rend la régression log-log très peu fiable.
    - Statistiquement, avec 10 points :
        - On est plus proche de l’analyse d’un fractional Gaussian noise (fGn) (stationnaire),
        - Que d’un fractional Brownian motion (fBm) (non-stationnaire, avec vraie structure d’échelle),
        - Donc la méthode SWV devient inadaptée à de si petits échantillons.
- En revanche, dans leur papier :
    - Leurs graphiques de h(t) sont visuellement très stables,
    - Leur indicateur d’inefficience (h(t) - 0.5) oscille faiblement autour de 0,
    - Ce qui est totalement incompatible avec une estimation brute par SWV sur 10 points.

## 3. Analyse critique actuelle

Leur résultat propre suggère plusieurs possibilités :

| Hypothèse                     | Explication                                                                                                 | Commentaire                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| 1. Fenêtre plus grande que 10 | Estimation de $h$ sur une fenêtre plus large (ex : 100 points) avancée tous les 10 jours                         | Cohérent avec stabilité observée              |
| 2. Lissage ou Spline          | Application d’une pénalisation ou d’un lissage (ex : spline smoothing) sur la courbe $h(t)$ après estimation       | Mention de Zanin and Marra (2012) va dans ce sens |
| 3. Variante de méthode        | Utilisation d’une méthode inspirée de SWV mais simplifiée, par exemple mesure directe de volatilité sur 10 jours | Possible mais non précisé clairement          |

## 4. Lien avec Zanin and Marra (2012)

Le papier cite Zanin and Marra (2012), où :
- Ils montrent que rolling regression simple donne des coefficient très volatiles,
- Et que modèles de spline pénalisée produisent des coefficients lissés et fiables.

Donc il est probable que :
- Les auteurs aient adopté un modèle de régression pénalisée ou un système de lissage sur $h(t)$,
- Ce qui n’est pas expliqué explicitement dans leur méthodologie.

5. Conséquences et prochaines étapes

| Action                             | Pourquoi ?                                                                                         |
|------------------------------------|----------------------------------------------------------------------------------------------------|
| Écrire aux auteurs                 | Pour demander s’ils appliquent un lissage sur h(t) après l’estimation locale                         |
| Tester rolling avec grande fenêtre | Estimer h sur 100 points (rolling tous les 10 jours) pour voir si on retrouve leur stabilité         |
| Simuler post-traitement spline     | Appliquer un spline smoothing sur h(t) estimé sur 10 points pour voir si cela reproduit leurs figures  |
| Expérimenter DFA local             | Tester une DFA locale plutôt qu’une SWV directe pour comparaison                                   |

