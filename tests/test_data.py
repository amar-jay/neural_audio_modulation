from neural_audio_modulation.data.dataset import Dataset
from neural_audio_modulation.data.preprocessing import preprocess_audio
import pytest


def test_dataset_loading():
    dataset = Dataset("path/to/audio/files")
    assert len(dataset) > 0, "Dataset should not be empty after loading."


def test_preprocessing():
    audio_data = [0.1, 0.2, 0.3, 0.4]  # Example audio data
    processed_data = preprocess_audio(audio_data)
    assert len(processed_data) == len(audio_data), "Processed data length should match original."
    assert all(-1 <= x <= 1 for x in processed_data), (
        "Processed data should be normalized between -1 and 1."
    )
