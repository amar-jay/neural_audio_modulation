prepare:
	@python -m src.data.prepare

play_random:
	@python -m src.data.dataset --play --audio_dir=data/esc50/audio

lint:
	@ruff format src

train:
	@python -m src.training.train