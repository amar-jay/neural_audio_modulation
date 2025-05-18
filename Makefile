#librispeech, esc50
DATASET=librispeech
AUDIO_FILE=data/librispeech/audio/84_121123_84-121123-0021.flac
#data/esc50/audio/1-977-A-39.wav
prepare:
	@python -m src.data.prepare --dataset=${DATASET}

play_random:
	@python -m src.data.dataset --play --audio_dir=data/${DATASET}/audio

lint:
	@ruff format src

train:
	@python -m src.training.train

test:
	@python -m src.training.test --model_path=trained_models/best.pth --file=${AUDIO_FILE} --play