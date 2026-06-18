Audit MLOps du projet churn

Défaut 1 - Token API codé en dur et inutilisé

Localisation : app.py, ligne 12 (API_TOKEN = "churn-demo-token")

Description : Le token est écrit en clair dans le code, donc versionné dans Git pour toujours. Et en plus il ne sert à rien : aucune route ne le vérifie. N'importe qui peut appeler /predict sans authentification.

Criticité : HAUTE

Justification : c'est un secret exposé qui ne protège même pas l'API, double problème. En prod n'importe qui pourrait spammer l'API sans contrôle d'accès.

Défaut 2 - Chargement du modèle sans vérification d'intégrité

Localisation : app.py, ligne 28 (model = joblib.load(MODEL_PATH))

Description : Le fichier model.pkl est chargé avec joblib sans vérifier qu'il n'a pas été modifié (pas de hash, rien). Or désérialiser un pickle peut exécuter du code arbitraire si le fichier a été altéré.

Criticité : HAUTE

Justification : risque d'exécution de code si quelqu'un arrive à remplacer le fichier model.pkl (artefact compromis, pipeline CI piraté...). C'est un classique des failles ML/pickle et rien ici ne le détecterait.

Défaut 3 - Bug d'encodage à l'inférence (drop_first=True)

Localisation : src/prepare.py ligne 17 et app.py ligne 68 (pd.get_dummies(..., drop_first=True))

Description : Le même drop_first=True est utilisé à l'entraînement et à l'inférence. Sauf qu'à l'inférence on a une seule ligne (une requête), donc la colonne contract n'a qu'une valeur possible et drop_first la supprime à chaque fois, peu importe laquelle. Le reindex qui suit remplit ça avec des zéros, donc le modèle ne reçoit pas la bonne info de contrat.

Criticité : HAUTE

Justification : bug silencieux, ça ne crashe pas, ça renvoie une prédiction quand même (code 200) mais potentiellement fausse. C'est le pire type de bug en prod parce que rien n'alerte que ça tourne mal.

Défaut 4 - Les erreurs internes sont renvoyées telles quelles au client

Localisation : app.py, ligne 85 (raise HTTPException(status_code=500, detail=str(e)))

Description : Si une exception arrive dans /predict, le message Python brut est renvoyé dans la réponse HTTP. Ça peut donner des infos internes (noms de colonnes, types d'erreur...) à n'importe quel appelant.

Criticité : MOYENNE

Justification : fuite d'info qui aide un attaquant à comprendre l'architecture interne. Pas catastrophique en soi mais facile à corriger.

Défaut 5 - Le test "non-régression" ne compare à rien + cas d'échec pas testés

Localisation : tests/test_non_regression.py (tout le fichier)

Description : Ce test ré-entraîne un modèle à chaque run et check juste que l'accuracy dépasse 0.70. Il ne compare jamais à un modèle précédent, donc ça ne détecte pas une régression de comportement, juste un seuil de qualité. À côté de ça, aucun test ne couvre les cas d'échec (modèle pas chargé, valeur de contract invalide, /metrics qui s'incrémente vraiment).

Criticité : MOYENNE

Justification : le nom du test donne une fausse confiance ("on a un test de non-régression" alors que non), et l'absence de tests sur les cas d'échec veut dire qu'une future modif peut casser des trucs sans que personne le voie.
