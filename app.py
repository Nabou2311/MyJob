from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 🤖 JobAI — Application
# ============================================================

st.set_page_config(
    page_title="JobAI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 1. Chargement des données
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHEMIN_FICHIER = BASE_DIR / "data" / "raw" / "offres_emploi.csv"


@st.cache_data
def charger_donnees():
    """Charge et prépare le jeu de données JobAI."""

    df = pd.read_csv(CHEMIN_FICHIER)

    # Même préparation que dans le notebook.
    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(
            df["publication_date"],
            errors="coerce"
        )

    df["salary_available"] = df["salary"].notna()

    df["skills_list"] = (
        df["skills"]
        .fillna("")
        .str.split(";")
    )

    df_skills = df.explode("skills_list").rename(
        columns={"skills_list": "skill"}
    )

    skills_counts = df_skills["skill"].value_counts()

    # Texte utilisé pour les embeddings.
    df["texte_offre"] = (
        df["title"].fillna("") + " " +
        df["company"].fillna("") + " " +
        df["location"].fillna("") + " " +
        df["contract"].fillna("") + " " +
        df["skills"].fillna("") + " " +
        df["description"].fillna("")
    )

    return df, df_skills, skills_counts


# ============================================================
# 2. Chargement du modèle d'embeddings
# ============================================================

@st.cache_resource
def charger_modele():
    """Charge le modèle SentenceTransformer une seule fois."""
    return SentenceTransformer("all-MiniLM-L6-v2")


df, df_skills, skills_counts = charger_donnees()
model = charger_modele()

# Génère les embeddings des offres une seule fois.
@st.cache_data
def generer_embeddings(textes):
    return model.encode(
        textes,
        show_progress_bar=False
    )


embeddings_offres = generer_embeddings(
    df["texte_offre"].tolist()
)


# ============================================================
# 3. Vocabulaire connu par JobAI
# ============================================================

job_titles = [
    "Data Analyst",
    "Data Scientist",
    "BI Analyst",
    "Analyste BI",
    "Data Engineer"
]

contracts = [
    "Alternance",
    "Stage"
]

locations = [
    "Lille",
    "Paris",
    "Lyon",
    "Villeneuve-d'Ascq"
]

skills_available = skills_counts.index.tolist()


# ============================================================
# 4. Normalisation de la question
# ============================================================

synonymes = {
    "alternance": ["alternance", "apprentissage", "apprenti"],
    "stage": ["stage", "stagiaire"],
    "data analyst": [
        "data analyst",
        "analyste data",
        "analyste de données"
    ],
    "data scientist": [
        "data scientist",
        "scientifique des données"
    ],
    "data engineer": [
        "data engineer",
        "ingénieur data"
    ],
    "python": ["python"],
    "sql": ["sql"],
    "power bi": ["power bi"],
    "machine learning": [
        "machine learning",
        "apprentissage automatique"
    ],
}


def normaliser_question(question):
    """Normalise les différentes formulations utilisateur."""

    question_normalisee = question.lower()

    for concept, variantes in synonymes.items():
        for variante in variantes:
            if variante in question_normalisee:
                question_normalisee = question_normalisee.replace(
                    variante,
                    concept
                )

    return question_normalisee


# ============================================================
# 5. Compréhension de la requête
# ============================================================

def comprendre_requete(question):
    """
    Analyse une question utilisateur et extrait :
    métier, ville, contrat et compétences.
    """

    question_normalisee = normaliser_question(question)

    title = None
    location = None
    contract = None
    skills = []

    for job in job_titles:
        if job.lower() in question_normalisee:
            title = job
            break

    for city in locations:
        if city.lower() in question_normalisee:
            location = city
            break

    for contract_type in contracts:
        if contract_type.lower() in question_normalisee:
            contract = contract_type
            break

    for skill in skills_available:
        if skill.lower() in question_normalisee:
            skills.append(skill)

    return {
        "title": title,
        "location": location,
        "contract": contract,
        "skills": skills
    }


# ============================================================
# 6. Recherche sémantique
# ============================================================

def recherche_semantique(question, top_k=5):
    """Recherche les offres les plus proches sémantiquement."""

    embedding_question = model.encode(question)

    similarites = cosine_similarity(
        embedding_question.reshape(1, -1),
        embeddings_offres
    )[0]

    resultats = df.copy()
    resultats["similarite"] = similarites

    return (
        resultats
        .sort_values("similarite", ascending=False)
        .head(top_k)
    )


# ============================================================
# 7. Recherche hybride finale
# ============================================================

def recherche_hybride(question, top_k=5):
    """
    Recherche hybride JobAI.

    60 % : similarité sémantique
    40 % : respect des critères explicites
    """

    criteres = comprendre_requete(question)

    embedding_question = model.encode(question)

    similarities = cosine_similarity(
        embedding_question.reshape(1, -1),
        embeddings_offres
    )[0]

    resultats = df.copy()
    resultats["score_semantique"] = similarities

    # Score des critères.
    resultats["score_criteres"] = 0.0
    nombre_criteres = 0

    # Métier.
    if criteres["title"] is not None:
        nombre_criteres += 1

        correspondance_title = (
            resultats["title"].fillna("").str.lower()
            == criteres["title"].lower()
        )

        resultats.loc[
            correspondance_title,
            "score_criteres"
        ] += 1

    # Localisation.
    if criteres["location"] is not None:
        nombre_criteres += 1

        correspondance_location = (
            resultats["location"].fillna("").str.lower()
            == criteres["location"].lower()
        )

        resultats.loc[
            correspondance_location,
            "score_criteres"
        ] += 1

    # Contrat.
    if criteres["contract"] is not None:
        nombre_criteres += 1

        correspondance_contract = (
            resultats["contract"].fillna("").str.lower()
            == criteres["contract"].lower()
        )

        resultats.loc[
            correspondance_contract,
            "score_criteres"
        ] += 1

    # Compétences.
    score_skills = pd.Series(
        0.0,
        index=resultats.index
    )

    if len(criteres["skills"]) > 0:
        nombre_criteres += 1

        for skill in criteres["skills"]:
            correspondance = (
                resultats["skills"]
                .fillna("")
                .str.lower()
                .str.contains(
                    skill.lower(),
                    na=False
                )
            )

            score_skills += correspondance.astype(float)

        score_skills /= len(criteres["skills"])

    resultats["score_criteres"] += score_skills

    if nombre_criteres > 0:
        resultats["score_criteres"] /= nombre_criteres

    # Score hybride final.
    resultats["score_hybride"] = (
        0.6 * resultats["score_semantique"]
        + 0.4 * resultats["score_criteres"]
    )

    # Une offre exacte respecte tous les critères détectés.
    resultats_exacts = resultats[
        resultats["score_criteres"] == 1
    ].copy()

    resultats_exacts = resultats_exacts.sort_values(
        "score_hybride",
        ascending=False
    )

    if resultats_exacts.empty:
        return resultats_exacts

    return resultats_exacts.head(top_k)


# ============================================================
# 8. Recherche d'offres proches
# ============================================================

def rechercher_offres_proches(question, top_k=3):
    """Recherche des offres proches par similarité sémantique."""

    embedding_question = model.encode(question)

    similarites = cosine_similarity(
        embedding_question.reshape(1, -1),
        embeddings_offres
    )[0]

    resultats = df.copy()
    resultats["score_semantique"] = similarites

    return (
        resultats
        .sort_values("score_semantique", ascending=False)
        .head(top_k)
    )


def expliquer_offre_proche(offre, criteres):
    """Explique les différences entre une offre et la demande."""

    differences = []

    if criteres["title"] is not None:
        if (
            str(offre["title"]).lower()
            != criteres["title"].lower()
        ):
            differences.append(
                f"métier demandé : {criteres['title']}"
            )

    if criteres["location"] is not None:
        if (
            str(offre["location"]).lower()
            != criteres["location"].lower()
        ):
            differences.append(
                f"localisation demandée : {criteres['location']}"
            )

    if criteres["contract"] is not None:
        if (
            str(offre["contract"]).lower()
            != criteres["contract"].lower()
        ):
            differences.append(
                f"contrat demandé : {criteres['contract']}"
            )

    competences_manquantes = []

    for skill in criteres["skills"]:
        if skill.lower() not in str(offre["skills"]).lower():
            competences_manquantes.append(skill)

    if competences_manquantes:
        differences.append(
            "compétences non trouvées : "
            + ", ".join(competences_manquantes)
        )

    if not differences:
        return ["Tous les critères correspondent."]

    return differences


# ============================================================
# 9. Interface JobAI
# ============================================================

st.title("🤖 JobAI")
st.subheader("Recherche intelligente d'offres d'emploi")

st.markdown(
    """
Décris simplement le poste que tu recherches en langage naturel.

**Exemples :**
- Je cherche une alternance Data Analyst à Lille avec Python et SQL
- Je cherche un stage Data Scientist à Paris avec Python
- Je cherche un poste Data Analyst à Paris
"""
)

# Informations générales dans la barre latérale.
with st.sidebar:
    st.header("⚙️ JobAI")

    st.write(f"**Offres disponibles :** {len(df)}")
    st.write(f"**Métiers connus :** {len(job_titles)}")
    st.write(f"**Compétences connues :** {len(skills_available)}")

    st.divider()

    st.caption(
        "Le moteur combine la compréhension des critères, "
        "la recherche sémantique et un score hybride."
    )

# Historique de conversation.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Affichage de l'historique.
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["type"] == "text":
            st.markdown(message["content"])

        elif message["type"] == "results":
            st.markdown(message["content"])

            if message["results_type"] == "exact":
                for _, offre in message["data"].iterrows():
                    with st.container(border=True):
                        st.markdown(
                            f"### 🏢 {offre['company']} — {offre['title']}"
                        )
                        st.write(
                            f"📍 **Localisation :** {offre['location']}"
                        )
                        st.write(
                            f"📄 **Contrat :** {offre['contract']}"
                        )
                        st.write(
                            f"🛠️ **Compétences :** {offre['skills']}"
                        )
                        st.write(
                            f"🔎 **Score :** "
                            f"{offre['score_hybride']:.3f}"
                        )

            else:
                for _, offre in message["data"].iterrows():
                    with st.container(border=True):
                        st.markdown(
                            f"### 🏢 {offre['company']} — {offre['title']}"
                        )
                        st.write(
                            f"📍 **Localisation :** {offre['location']}"
                        )
                        st.write(
                            f"📄 **Contrat :** {offre['contract']}"
                        )
                        st.write(
                            f"🔎 **Similarité :** "
                            f"{offre['score_semantique']:.3f}"
                        )

                        differences = expliquer_offre_proche(
                            offre,
                            message["criteres"]
                        )

                        st.write("⚠️ **Différences :**")
                        for difference in differences:
                            st.write(f"• {difference}")


# ============================================================
# 10. Saisie utilisateur
# ============================================================

question = st.chat_input(
    "Ex. Je cherche une alternance Data Analyst à Lille avec Python et SQL"
)

if question:

    # Message utilisateur.
    st.session_state.messages.append({
        "role": "user",
        "type": "text",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Analyse JobAI.
    criteres = comprendre_requete(question)
    resultats = recherche_hybride(question, top_k=5)

    with st.chat_message("assistant"):

        st.markdown("### 🔎 Recherche JobAI")

        # Affiche les critères compris.
        criteres_affiches = []

        if criteres["title"]:
            criteres_affiches.append(
                f"**Métier :** {criteres['title']}"
            )

        if criteres["location"]:
            criteres_affiches.append(
                f"**Ville :** {criteres['location']}"
            )

        if criteres["contract"]:
            criteres_affiches.append(
                f"**Contrat :** {criteres['contract']}"
            )

        if criteres["skills"]:
            criteres_affiches.append(
                "**Compétences :** "
                + ", ".join(criteres["skills"])
            )

        if criteres_affiches:
            st.markdown(
                " | ".join(criteres_affiches)
            )

        # Correspondances exactes.
        if not resultats.empty:

            message_intro = (
                f"✅ **{len(resultats)} offre(s) "
                "correspondent à votre recherche.**"
            )

            st.markdown(message_intro)

            for _, offre in resultats.iterrows():

                with st.container(border=True):

                    st.markdown(
                        f"### 🏢 {offre['company']} — {offre['title']}"
                    )

                    st.write(
                        f"📍 **Localisation :** {offre['location']}"
                    )

                    st.write(
                        f"📄 **Contrat :** {offre['contract']}"
                    )

                    st.write(
                        f"🛠️ **Compétences :** {offre['skills']}"
                    )

                    st.write(
                        f"🔎 **Score :** "
                        f"{offre['score_hybride']:.3f}"
                    )

            # Sauvegarde dans l'historique.
            st.session_state.messages.append({
                "role": "assistant",
                "type": "results",
                "content": message_intro,
                "results_type": "exact",
                "data": resultats,
                "criteres": criteres
            })

        # Aucune correspondance exacte.
        else:

            offres_proches = rechercher_offres_proches(
                question,
                top_k=3
            )

            message_intro = (
                "❌ **Aucune offre ne correspond exactement "
                "à votre recherche.**\n\n"
                "🔎 Voici quelques offres proches :"
            )

            st.markdown(message_intro)

            for _, offre in offres_proches.iterrows():

                with st.container(border=True):

                    st.markdown(
                        f"### 🏢 {offre['company']} — {offre['title']}"
                    )

                    st.write(
                        f"📍 **Localisation :** {offre['location']}"
                    )

                    st.write(
                        f"📄 **Contrat :** {offre['contract']}"
                    )

                    st.write(
                        f"🔎 **Similarité :** "
                        f"{offre['score_semantique']:.3f}"
                    )

                    differences = expliquer_offre_proche(
                        offre,
                        criteres
                    )

                    st.write("⚠️ **Différences :**")

                    for difference in differences:
                        st.write(f"• {difference}")

            st.session_state.messages.append({
                "role": "assistant",
                "type": "results",
                "content": message_intro,
                "results_type": "close",
                "data": offres_proches,
                "criteres": criteres
            })
