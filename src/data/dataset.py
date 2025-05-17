from torch.utils.data import Dataset
import torchaudio
import os
# audio player


class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate, self.audio_files[idx]


if __name__ == "__main__":
    import argparse
    import numpy as np
    from src.utils.audio_utils import play_audio

    parser = argparse.ArgumentParser(description="Audio Dataset Example")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./data/esc50/audio/",
        required=True,
        help="Path to the audio directory",
    )
    parser.add_argument("--play", action="store_true", help="Play a random audio file")
    args = parser.parse_args()

    dataset = AudioDataset(args.audio_dir)
    print(f"Loaded {len(dataset)} audio files from {args.audio_dir}.")

    if args.play:
        # Play a random audio file
        idx = np.random.randint(0, len(dataset))
        waveform, sample_rate, filename = dataset[idx]
        print(f"Loaded {filename} with sample rate {sample_rate} and shape {waveform.shape}.")
        play_audio(waveform, sample_rate)
