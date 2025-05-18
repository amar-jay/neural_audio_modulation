#!/usr/bin/env python3

import os
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm
from pathlib import Path


def flatten_librispeech_dataset(src_root: str, audio_ext=".flac"):
    src_root = Path(src_root)
    dest_root = os.path.join(src_root, "audio")
    # make sure the destination directory exists
    os.makedirs(dest_root, exist_ok=True)
    src_root = Path(os.path.join(src_root, "dev-clean"))

    # regex to match the audio files
    for audio_path in src_root.rglob(f"*{audio_ext}"):
        parts = audio_path.relative_to(src_root).with_suffix("").parts
        if len(parts) < 3:
            continue
        speaker, chapter, utterance = parts[-3], parts[-2], parts[-1]
        new_filename = f"{speaker}_{chapter}_{utterance}{audio_ext}"
        dest_path = os.path.join(dest_root, new_filename)

        shutil.copy(audio_path, dest_path)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    parser = argparse.ArgumentParser(description="Download a small audio dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="esc50",
        choices=["esc50", "librispeech", "gtzan", "urbansound8k"],
        help="Dataset to download (default: esc50)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to store the dataset (default: ./data)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Dictionary of datasets and their URLs
    datasets = {
        "esc50": {
            "url": "https://github.com/karoldvl/ESC-50/archive/master.zip",
            "extract_path": os.path.join(args.output_dir, "ESC-50-master"),
            "final_path": os.path.join(args.output_dir, "esc50"),
        },
        "librispeech": {
            "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "extract_path": os.path.join(args.output_dir, "LibriSpeech"),
            "final_path": os.path.join(args.output_dir, "librispeech"),
            "preprocess": flatten_librispeech_dataset,
        },
        "gtzan": {
            "url": "https://oramics.github.io/sampled/MIDI/GTZAN/genres.zip",
            "extract_path": os.path.join(args.output_dir, "genres"),
            "final_path": os.path.join(args.output_dir, "gtzan"),
        },
        "urbansound8k": {
            "url": "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz",
            "extract_path": os.path.join(args.output_dir, "UrbanSound8K"),
            "final_path": os.path.join(args.output_dir, "urbansound8k"),
        },
    }

    # Get selected dataset
    dataset = datasets[args.dataset]

    # Check if dataset is already downloaded
    if os.path.exists(dataset["final_path"]):
        print(f"{args.dataset} dataset already exists at {dataset['final_path']}.")
        return

    # Download and extract dataset
    download_path = os.path.join(args.output_dir, os.path.basename(dataset["url"]))

    print(f"Downloading {args.dataset} dataset...")
    download_url(dataset["url"], download_path)

    print(f"Extracting {args.dataset} dataset...")
    if download_path.endswith(".zip"):
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(args.output_dir)
    elif download_path.endswith(".tar.gz"):
        with tarfile.open(download_path, "r:gz") as tar_ref:
            tar_ref.extractall(args.output_dir)

    # Rename extracted folder to standardized name if needed
    if dataset["extract_path"] != dataset["final_path"]:
        if os.path.exists(dataset["final_path"]):
            shutil.rmtree(dataset["final_path"])
        os.rename(dataset["extract_path"], dataset["final_path"])

    # Clean up downloaded archive
    os.remove(download_path)

    # Preprocess dataset if needed
    if "preprocess" in dataset:
        print(f"Preprocessing {args.dataset} dataset...")
        dataset["preprocess"](dataset["final_path"])
    print(f"Dataset installed at: {dataset['final_path']}")

    # Print basic usage example
    print("\nUsage example with your AudioDataset:")
    print(f"from neural_audio_modulation.src.data.dataset import AudioDataset")

    if args.dataset == "esc50":
        print(f"dataset = AudioDataset('{dataset['final_path']}/audio')")
    elif args.dataset == "librispeech":
        print(f"dataset = AudioDataset('{dataset['final_path']}/dev-clean')")
    elif args.dataset == "gtzan":
        print(f"dataset = AudioDataset('{dataset['final_path']}')")
    elif args.dataset == "urbansound8k":
        print(f"dataset = AudioDataset('{dataset['final_path']}/audio')")


if __name__ == "__main__":
    main()
