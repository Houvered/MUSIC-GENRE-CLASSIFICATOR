# 🎵 Music Genre Classifier

A menu-driven Python application that classifies music genres using **TF-IDF** and **K-Nearest Neighbors (KNN)**. The project supports **incremental learning** by fetching song metadata from the **iTunes Search API** when a song is not found in the local dataset, allowing the classifier to improve over time.

---

## 📌 Features

* 🎼 Classify songs into music genres using Machine Learning.
* 🔍 Text feature extraction with **TF-IDF (Term Frequency–Inverse Document Frequency)**.
* 🤖 Genre prediction using the **K-Nearest Neighbors (KNN)** algorithm.
* 🌐 Automatic fallback to the **iTunes Search API** for unknown songs.
* 📚 Incremental learning by adding newly discovered songs to the local dataset.
* 📋 Interactive menu-driven command-line interface.
* 💾 Persistent dataset updates for continuous improvement.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn
* **Feature Extraction:** TF-IDF Vectorizer
* **Classification:** K-Nearest Neighbors (KNN)
* **API:** iTunes Search API
* **Data Processing:** Pandas
* **Serialization:** Joblib / Pickle (if used)

---

## 📂 Project Structure

```text
Music-Genre-Classificator/
│── data/                 # Dataset
│── models/               # Saved ML model (optional)
│── main.py               # Application entry point
│── classifier.py         # Model training & prediction
│── api.py                # iTunes API integration
│── utils.py              # Helper functions
│── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Load the existing music dataset.
2. Convert song metadata into numerical features using **TF-IDF**.
3. Train a **KNN classifier** on the dataset.
4. Accept a song title from the user.
5. If the song exists locally:

   * Predict and display its genre.
6. If the song is not found:

   * Query the **iTunes Search API**.
   * Retrieve song metadata.
   * Predict the genre.
   * Optionally append the new data to the dataset for future predictions.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Music-Genre-Classificator.git
cd Music-Genre-Classificator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## 📦 Requirements

* Python 3.10+
* pandas
* scikit-learn
* requests
* numpy

Install them using:

```bash
pip install pandas scikit-learn requests numpy
```

---

## 💡 Example

```text
========== Music Genre Classifier ==========
1. Classify Song
2. Train Model
3. Exit

Enter song name: Shape of You

Predicted Genre: Pop
Confidence: 92%
```

If the song isn't available locally:

```text
Song not found in local dataset.
Searching iTunes...

Song found!
Predicting genre...
Genre: Pop

Would you like to save this song to the dataset? (Y/N)
```

---

## 📈 Future Improvements

* Deep learning models (LSTM/BERT)
* Spotify API integration
* Audio feature extraction (MFCC, Chroma, Spectral Contrast)
* Confidence score visualization
* GUI using Tkinter or Streamlit
* Web deployment with Flask or FastAPI
* Support for multilingual song metadata
* Batch prediction for playlists

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

---

## 👨‍💻 Author

**Nirmit Kumar Srivastava**

* B.Tech CSE (AI)
* Passionate about Machine Learning, AI, and Open Source

If you found this project useful, consider giving it a ⭐ on GitHub!
