import os
import numpy as np
import torch
import sounddevice as sd
import torchaudio


def load_audio(file_path):
    """Load an audio file."""
    sample_rate, data = torchaudio.load(file_path)
    return sample_rate, data


def play_audio(waveform, sample_rate):
    """Play audio from a waveform tensor."""
    waveform_np = waveform.numpy()

    # Handle stereo (multi-channel) audio
    if waveform_np.shape[0] > 1:
        # Convert to mono by averaging channels
        print(waveform_np.shape)
        waveform_np = np.mean(waveform_np, axis=0, keepdims=True)
        print(waveform_np.shape)

    # Ensure the audio is scaled properly for playback
    waveform_np = waveform_np.squeeze()
    if waveform_np.max() > 1.0 or waveform_np.min() < -1.0:
        waveform_np = waveform_np / max(abs(waveform_np.max()), abs(waveform_np.min()))

    sd.play(waveform_np, sample_rate)
    sd.wait()


def save_audio(file_path, data, sample_rate):
    """Save audio data to a file."""
    if not file_path.endswith(".wav"):
        file_path += ".wav"
    torchaudio.save(file_path, data, sample_rate)
    print(f"Audio saved to {file_path}")


# save model, sequentially
def save_model(model):
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "neural_audio_encoding.pth")
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model_path
