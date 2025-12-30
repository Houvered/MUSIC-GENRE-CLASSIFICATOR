import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import os
import sys

# =========================
# CONFIG
# =========================
DATASET = "songs_dataset.csv"
CONFIDENCE_THRESHOLD = 0.60

# =========================
# LOAD OR CREATE DATASET
# =========================
def load_dataset():
    if not os.path.exists(DATASET):
        df = pd.DataFrame(columns=["song", "artist", "genre"])
        df.to_csv(DATASET, index=False)
        return df

    df = pd.read_csv(DATASET)

    for col in ["song", "artist", "genre"]:
        if col not in df.columns:
            print("❌ Dataset format is invalid.")
            sys.exit(1)

    return df


# =========================
# TRAIN MODEL
# =========================
def train_model(df):
    df = df.dropna(subset=["song", "artist", "genre"])
    df = df[df["song"].str.strip() != ""]
    df = df[df["artist"].str.strip() != ""]

    if df.empty:
        return None, None

    df["text"] = (df["song"] + " " + df["artist"]).astype(str)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["genre"]

    k = min(3, len(df))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    return knn, vectorizer


# =========================
# CONFIDENCE CALCULATION
# =========================
def knn_confidence(knn, vec):
    distances, _ = knn.kneighbors(vec)
    avg_distance = distances.mean()
    return 1 / (1 + avg_distance)


# =========================
# ITUNES API FALLBACK
# =========================
def get_genre_from_api(song):
    print("\n🔎 Trying iTunes API...")
    url = "https://itunes.apple.com/search"
    params = {"term": song, "entity": "song", "limit": 1}

    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()

        if data.get("resultCount", 0) > 0:
            result = data["results"][0]
            return result.get("primaryGenreName"), result.get("artistName")
    except Exception:
        pass

    return None, None


# =========================
# VIEW DATASET
# =========================
def view_dataset(df):
    if df.empty:
        print("\n📂 Dataset is empty.")
    else:
        print("\n📊 Current Dataset:\n")
        print(df.to_string(index=False))


# =========================
# JUDGE SONG GENRE
# =========================
def judge_song(df):
    knn, vectorizer = train_model(df)

    song_input = input("\n🎧 Enter Song Name: ").strip()

    # Exact match
    exact = df[df["song"].str.lower() == song_input.lower()]
    if not exact.empty:
        print(f"\n🎵 Found in dataset → Genre: {exact.iloc[0]['genre']}")
        return df

    # KNN prediction
    if knn is not None:
        vec = vectorizer.transform([song_input])
        genre = knn.predict(vec)[0]
        confidence = knn_confidence(knn, vec)

        print(f"\n🤖 KNN Prediction: {genre}")
        print(f"📊 Confidence: {confidence:.2f}")

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"\n✅ Final Genre: {genre}")
            return df
        else:
            print("\n⚠ Low confidence → switching to API.")
    else:
        print("\n📂 Dataset too small → using API.")

    # API fallback
    api_genre, api_artist = get_genre_from_api(song_input)

    if api_genre:
        print(f"\n🎵 API Genre: {api_genre}")
        print(f"👤 Artist: {api_artist}")

        new_row = pd.DataFrame({
            "song": [song_input],
            "artist": [api_artist],
            "genre": [api_genre]
        })

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATASET, index=False)

        print("✅ Added to dataset.")
        print(f"\n🎵 Final Genre: {api_genre}")
    else:
        print("\n❌ Could not determine genre.")

    return df


# =========================
# MAIN MENU LOOP
# =========================
def main():
    df = load_dataset()

    while True:
        print("\n==============================")
        print("🎶 MUSIC GENRE CLASSIFIER")
        print("==============================")
        print("1️⃣  View Dataset")
        print("2️⃣  Judge Song Genre")
        print("3️⃣  Exit")

        choice = input("\nChoose an option (1/2/3): ").strip()

        if choice == "1":
            view_dataset(df)

        elif choice == "2":
            while True:
                df = judge_song(df)
                again = input("\n🔁 Judge another song? (y/n): ").strip().lower()
                if again != "y":
                    break

        elif choice == "3":
            print("\n👋 Exiting program. Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
