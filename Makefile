prepare:
	@python -m src.data.prepare

play_random:
	@python -m src.data.dataset --play --audio_dir=data/esc50/audio

lint:
	@ruff format src

train:
	@python -m src.training.train

test:
	@python -m src.training.test --model_path=trained_models/neural_audio_encoding.pth --file=data/esc50/audio/1-977-A-39.wav --play