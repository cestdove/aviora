# 🎵 BioAcoustic Vector Search — European Bird Songs

A bioacoustic search engine that lets you upload an audio clip and find the most similar bird songs in a local database, using **MFCC feature extraction**, **ChromaDB** vector search, and a **Streamlit** front end.

Upload a recording (or a short snippet), and the app extracts its acoustic "fingerprint," compares it against a database of bird songs sourced from [Xeno-Canto](https://xeno-canto.org), and returns the closest matches with audio playback and spectrograms.

## How it works

1. **Dataset collection** (`birds.py`) — Queries the Xeno-Canto v3 API for recordings of common European bird species, downloads the audio, and builds a CSV catalog (`data/audio_catalog.csv`).
2. **Feature extraction** (`audio_utils.py`) — For each recording, separates harmonic content from background noise (HPSS), computes MFCCs, and pools them (mean + standard deviation) into a single fixed-length feature vector.
3. **Ingestion** (`ingest.py`) — Extracts features for every catalogued file and stores the resulting vectors in a persistent ChromaDB collection (L2 / Euclidean distance).
4. **Search** (`search.py`) — Given a query vector, performs a nearest-neighbor lookup against the ChromaDB collection.
5. **Web app** (`app.py`) — A Streamlit interface where you upload an audio file, run it through the same feature extraction pipeline, and view the top matches with playback and mel-spectrogram visualizations.

## Project structure

```text
.
├── app.py             # Streamlit web app (upload, search, display results)
├── audio_utils.py     # Feature extraction (MFCC) and audio chunking utilities
├── birds.py           # Dataset builder — downloads bird songs from Xeno-Canto
├── ingest.py          # Extracts features and loads them into ChromaDB
├── search.py          # Queries ChromaDB for similar audio
└── data/              # Created at runtime: audio files, CSV catalog, ChromaDB store
```

## Species covered

The default dataset targets eleven common European species, sourced from Italian recordings on Xeno-Canto:

| Scientific name | English name | Italian name |
|---|---|---|
| *Passer italiae* | Italian Sparrow | Passero |
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

The species list lives in the `target_birds` dictionary in `birds.py` and can be edited freely to add, remove, or replace species.

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
   git clone <your-repo-url>
   cd <your-repo-folder>
   pip install -r requirements.txt
   ```

2. Set your Xeno-Canto API key as an environment variable (required only for `birds.py`):

   ```bash
   export XENO_CANTO_API_KEY=your_key_here
   ```

## Usage

Run these steps in order the first time you set up the project.

**1. Build the dataset** — downloads bird song samples and creates `data/audio_catalog.csv`:

```bash
python birds.py
```

**2. Ingest the dataset into ChromaDB** — extracts features and builds the vector database:

```bash
python ingest.py
```

**3. Launch the search app:**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your browser, upload an audio file, and explore the results.

## 🔬 Known limitations (v1.0 DSP baseline)

This baseline relies on pure **Digital Signal Processing (DSP)** and geometric vector search. Testing revealed a few intrinsic limitations of this approach:

- **The "acoustic fingerprint" problem** — Statistical pooling of MFCCs over a time window blends the bird's vocalization with background noise, reverb, and recording hardware artifacts. The system tends to behave like *Shazam* (matching the exact same recording) rather than performing true semantic species classification.
- **High-variance species (the "Blackbird" problem)** — Species with complex, highly variable, non-repetitive songs (e.g., *Turdus merula* / Common Blackbird) generate scattered, non-dense point clouds in the high-dimensional vector space. The statistical means of their chunks vary widely, making accurate L2 distance matching harder than for more repetitive singers like tits or finches.
- **Vector space sparsity** — Minority classes with fewer valid 3-second chunks are easily overwhelmed by denser clusters during nearest-neighbor queries.

## 🚀 Roadmap (v2.0)

To move past the geometric limits of DSP, the next iteration will shift to **deep learning embeddings**:

- Replace `librosa` MFCCs with a pre-trained neural network (e.g., **YAMNet** or **BirdNET**).
- Shift from "waveform math" to true semantic acoustic embeddings, to better ignore background noise and group intra-species variation (like the Blackbird) into dense, robust clusters.

## Notes

- `ingest.py` rewrites the ChromaDB database on every run by default (`rewrite = True` at the top of the script). Set it to `False` if you want to add to an existing database instead of starting fresh.
- The target species and sample count are currently hardcoded in `birds.py`; edit the `target_birds` dictionary to adjust which species are included.
- The `data/` directory (audio files, CSV catalog, and ChromaDB store) is generated locally and is a good candidate for a `.gitignore` entry if you don't want to commit raw audio data.

## License

This project is licensed under the [MIT License](LICENSE).