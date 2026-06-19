# audio_utils.py
# Extraction of features from signals and slicing them into chunks

import librosa
import numpy as np


def extract_features_from_signals(y_chunk, sr):
    """Extract the acoustic signature filtering the ambient noise and silences (Digital Signal Processing)"""

    # Separate chants (Harmonic) from ambiental noise (Percussive) for little chunks with HPSS
    y_harmonic, _ = librosa.effects.hpss(y_chunk)

    # Calculate the Mel-frequency cepstral coefficients (MFCCs) for the cleaned query audio
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=21)  # typically, 20 MFCCs are used for audio analysis, now 21

    # Throws away the first coefficient (volume)
    mfccs_core = mfccs[1:, :]

    # Statistical Pooling
    # Time-averaging the MFCCs to get a single feature vector for the query audio
    mfccs_mean = np.mean(mfccs_core, axis=1)
    # Doig the Standard Deviation variance 
    mfccs_std = np.std(mfccs_core, axis=1)

    # Convert the numpy feature vectors to a unique intelligent python list for easier use in the search process
    feature_vector = np.concatenate((mfccs_mean, mfccs_std))

    return feature_vector.tolist()


def slice_signal_into_chunks(y, sr, chunk_duration=3):
    """Takes a continuous audio signal, removes the silences and makes 3 framments out of it."""

    # Cuts away the silence at the beginning and at the end (under 20db) with TRIM
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # If the file contained silence only, and it's cut off, we put back the old file to avoid crashes
    if len(y_trimmed) == 0:
        y_trimmed = y

    # Calculates how much samples corresponds to 3s
    chunk_samples = chunk_duration * sr
    chunks = []

    # Cuts the audio into sequential blocks
    for i in range(0, len(y_trimmed), chunk_samples):
        chunk = y_trimmed[i : i + chunk_samples]

        # Throws too short final fragments (i.e. 1s)
        if len(chunk) >= sr:
            chunks.append(chunk)

    return chunks