from matplotlib import pyplot as plt
import numpy as np
import librosa.display


def plot_waveform(signal, sample_rate, title="Waveform", xlabel="Time (s)", ylabel="Amplitude"):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sample_rate)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def plot_spectrogram(
    signal, sample_rate, title="Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)"
):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis="time", y_axis="log", cmap="coolwarm")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_training_history(history, title="Training History", xlabel="Epoch", ylabel="Loss/Metric"):
    plt.figure(figsize=(10, 4))
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
