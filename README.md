# Aviora

A bioacoustic search engine that lets you upload an audio clip and find the most similar bird songs in a local database, using **MFCC feature extraction**, **ChromaDB** vector search, and a **Streamlit** front end.

Upload a recording (or short snippet), and the app extracts its acoustic "fingerprint", compares it against bird songs sourced from [Xeno-Canto](https://xeno-canto.org), and returns the closest matches with audio playback and spectrograms.

## How it works

1. **Dataset collection** (`birds.py`) — Queries the Xeno-Canto API for recordings of common European bird species, downloads the audio, and builds a CSV catalog.
2. **Feature extraction** (`audio_utils.py`) — Applies HPSS, computes MFCCs, and pools them into a fixed-length feature vector.
3. **Ingestion** (`ingest.py`) — Stores the extracted vectors in a persistent ChromaDB collection using L2 distance.
4. **Search** (`search.py`) — Performs nearest-neighbor search against the ChromaDB collection.
5. **Web app** (`app.py`) — Streamlit interface for uploading audio and viewing similar recordings with playback and mel-spectrograms.

## Project structure

```text
.
├── app.py             # Streamlit web app
├── audio_utils.py     # MFCC feature extraction and audio utilities
├── birds.py           # Dataset builder — downloads bird songs
├── ingest.py          # Feature extraction and ChromaDB ingestion
├── search.py          # ChromaDB similarity search
└── data/              # Runtime-generated dataset and vector database
```

## Species covered

The default dataset targets **12 common European species**, sourced from Xeno-Canto:

| Scientific name | English name | Italian name |
|---|---|---|
| *Turdus merula* | Common Blackbird | Merlo |
| *Parus major* | Great Tit | Cinciallegra |
| *Streptopelia decaocto* | Eurasian Collared Dove | Tortora |
| *Erithacus rubecula* | European Robin | Pettirosso |
| *Fringilla coelebs* | Common Chaffinch | Fringuello |
| *Sylvia atricapilla* | Eurasian Blackcap | Capinera |
| *Carduelis carduelis* | European Goldfinch | Cardellino |
| *Luscinia megarhynchos* | Common Nightingale | Usignolo |
| *Troglodytes troglodytes* | Eurasian Wren | Scricciolo |
| *Alauda arvensis* | Eurasian Skylark | Allodola |
| *Sturnus vulgaris* | Common Starling | Storno |
| *Passer domesticus* | House Sparrow | Passero domestico |

The species list lives in the `target_birds` dictionary in `birds.py` and can be edited freely.

## Requirements

- Python 3.9+
- A free [Xeno-Canto](https://xeno-canto.org/explore/api) API key (only needed to build the dataset)

Python packages:

```text
streamlit
chromadb
soundfile
librosa
numpy
matplotlib
pandas
requests
```

## Setup

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/cestdove/aviora.git
cd aviora
pip install -r requirements.txt
```

2. Set your Xeno-Canto API key:

```bash
export XENO_CANTO_API_KEY=your_key_here
```

## Usage

Run these steps in order the first time:

**1. Build the dataset**

```bash
python birds.py
```

**2. Ingest the dataset into ChromaDB**

```bash
python ingest.py
```

**3. Launch the search app**

```bash
streamlit run app.py
```

Then upload an audio file and explore the most similar bird recordings.


## Notes

- `ingest.py` rewrites the ChromaDB database on every run by default (`rewrite = True`).
- The target species and sample count are configured in `birds.py`.
- The `data/` directory contains the local audio dataset, catalog, and ChromaDB store, generated at runtime.

## License

This project is licensed under the [MIT License](LICENSE).