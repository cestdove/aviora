# app.py
# the gradio interface for aviora vector search

import os

import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from audio_utils import extract_features_from_signals, slice_signal_into_chunks
from search import query_acoustic_database


def create_spectrogram(file_path):
    """Create a mel-spectrogram image for a result."""

    y, sr = librosa.load(file_path, sr=None)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=min(8000, sr // 2),
    )

    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))

    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        fmax=min(8000, sr // 2),
        ax=ax,
        cmap="magma",
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-Spectrogram")

    fig.tight_layout()

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    plt.close(fig)

    return image


def empty_results():
    """Return empty values and hide all result sections."""

    values = []

    for _ in range(5):
        values.extend(
            [
                "",
                None,
                None,
                gr.update(visible=False),
            ]
        )

    return values


def reset_results():
    """Clear previous search results."""

    return (
        "",
        *empty_results(),
    )


def show_loader():
    """Show the custom search loader."""

    return gr.update(visible=True)


def hide_loader():
    """Hide the custom search loader."""

    return gr.update(visible=False)


def search_birds(audio_path):
    """Process uploaded audio and search the acoustic database."""

    if audio_path is None:
        return (
            "Please upload an audio file.",
            *empty_results(),
        )

    try:
        y, sr = librosa.load(
            audio_path,
            sr=None,
            mono=False,
        )

        if y.ndim > 1:
            y = librosa.to_mono(y)

        # Limit query to the first 10 seconds.
        max_length = 10 * sr
        y = y[:max_length]

        # Split into 3-second chunks.
        query_chunks = slice_signal_into_chunks(
            y,
            sr,
            chunk_duration=3,
        )

        if not query_chunks:
            raise ValueError(
                "The uploaded audio is too short or contains only silence."
            )

        # Extract features from the first useful chunk.
        query_vector = extract_features_from_signals(
            query_chunks[0],
            sr,
        )

        # Search ChromaDB.
        results = query_acoustic_database(
            query_vector,
            n_results=5,
        )

        if (
            not results
            or not results.get("ids")
            or len(results["ids"][0]) == 0
        ):
            return (
                "No matches found in the database.",
                *empty_results(),
            )

        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results.get("metadatas")

        result_values = []

        for i in range(5):

            if i >= len(ids):
                result_values.extend(
                    [
                        "",
                        None,
                        None,
                        gr.update(visible=False),
                    ]
                )
                continue

            track_id = ids[i]
            distance = distances[i]

            if metadatas and metadatas[0]:
                metadata = metadatas[0][i]
            else:
                metadata = {}

            title = metadata.get(
                "title",
                f"Track {track_id}",
            )

            file_path = metadata.get(
                "file_path",
                None,
            )

            result_text = (
                f"### {i + 1}. {title}\n"
                f"**Euclidean Distance:** `{distance:.4f}`"
            )

            if file_path and os.path.exists(file_path):
                audio = file_path
                image = create_spectrogram(file_path)
                visibility = gr.update(visible=True)
            else:
                audio = None
                image = None
                visibility = gr.update(visible=True)

            result_values.extend(
                [
                    result_text,
                    audio,
                    image,
                    visibility,
                ]
            )

        return (
            "## Search completed\n"
            "Here are the most similar recordings.",
            *result_values,
        )

    except Exception as e:
        return (
            f"**Error:** `{e}`",
            *empty_results(),
        )


with gr.Blocks(title="Aviora") as demo:

    gr.Markdown(
        """
# Aviora

### Bioacoustic Bird Song Search Engine

Upload a bird recording and find the most similar songs
in the local acoustic database using **MFCC features**
and **ChromaDB vector search**.
"""
    )

    audio_input = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Upload or record audio",
    )

    search_button = gr.Button(
        "Search",
        variant="primary",
    )

    # Custom loading animation
    loader = gr.HTML(
        """
        <div class="aviora-loader">
            <div class="loader-spinner"></div>
            <div class="loader-text">
                Searching acoustic database
            </div>
        </div>

        <style>
            .aviora-loader {
                display: flex;
                align-items: center;
                gap: 12px;
                margin: 18px 0;
                padding: 14px 0;
            }

            .loader-spinner {
                width: 18px;
                height: 18px;
                border: 3px solid #d9d9d9;
                border-top-color: #555555;
                border-radius: 50%;
                animation: aviora-spin 0.8s linear infinite;
            }

            .loader-text {
                font-size: 15px;
                color: #666666;
            }

            @keyframes aviora-spin {
                from {
                    transform: rotate(0deg);
                }

                to {
                    transform: rotate(360deg);
                }
            }
        </style>
        """,
        visible=False,
    )

    status = gr.Markdown()

    result_components = []

    for i in range(5):

        result_group = gr.Column(
            visible=False
        )

        with result_group:

            if i > 0:
                gr.Markdown(
                    "<hr style='margin: 40px 0;'>"
                )

            result_text = gr.Markdown()

            result_audio = gr.Audio(
                label=f"Result {i + 1}",
                interactive=False,
            )

            result_image = gr.Image(
                label="Mel-Spectrogram",
                type="numpy",
            )

        result_components.extend(
            [
                result_text,
                result_audio,
                result_image,
                result_group,
            ]
        )

    # selecting a new file clears previous results
    # and hides the loader
    audio_input.change(
        fn=reset_results,
        inputs=[],
        outputs=[
            status,
            *result_components,
        ],
    )

    # show loader immediately when search is pressed
    show_loading = search_button.click(
        fn=show_loader,
        inputs=None,
        outputs=loader,
        queue=False,
    )

    # then run the actual search
    search_event = show_loading.then(
        fn=search_birds,
        inputs=audio_input,
        outputs=[
            status,
            *result_components,
        ],
    )

    # hide loader after the search finishes
    search_event.then(
        fn=hide_loader,
        inputs=None,
        outputs=loader,
    )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True
    )