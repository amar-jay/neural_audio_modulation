from torch.utils.data import Dataset
import torchaudio
import os
# audio player


class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.audio_files = [
            f for f in os.listdir(audio_dir) if f.endswith(".wav") or f.endswith(".flac")
        ]

    def __len__(self):
        return len(self.audio_files)

    def filter_dataset(self, min_length=0, max_length=float("inf")):
        """Filter the dataset based on the length of the audio files."""
        filtered_files = []
        for item in self:
            waveform, sample_rate, audio_file = item
            length = waveform.shape[1]
            if min_length <= length <= max_length:
                filtered_files.append(audio_file)
        self.audio_files = filtered_files

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform, sample_rate)

        return waveform, self.audio_files[idx]


if __name__ == "__main__":
    import argparse
    import numpy as np
    from src.utils.audio_utils import play_audio
    from src.data.preprocessing import transform_audio

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
    dataset.transform = transform_audio(
        min_max=True,
        softmax=False,
        sequence_length=220500,
        target_rate=44100,
    )
    print(f"Loaded {len(dataset)} audio files from {args.audio_dir}.")

    if args.play:
        # Play a random audio file
        idx = np.random.randint(0, len(dataset))
        waveform, filename = dataset[idx]
        print(f"Loaded {filename} with sample rate 44100Hz and shape {waveform.shape}.")
        play_audio(waveform, 44100)
