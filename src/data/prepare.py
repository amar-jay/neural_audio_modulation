#!/usr/bin/env python3

import os
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm


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
