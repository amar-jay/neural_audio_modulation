from neural_audio_modulation.models.encoder import Encoder
from neural_audio_modulation.models.decoder import Decoder
import torch
import unittest


class TestModels(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sample_input = torch.randn(1, 1, 16000)  # Example input shape for audio

    def test_encoder_output_shape(self):
        output = self.encoder(self.sample_input)
        self.assertEqual(output.shape, (1, 128))  # Adjust based on actual encoder output shape

    def test_decoder_output_shape(self):
        encoded = self.encoder(self.sample_input)
        output = self.decoder(encoded)
        self.assertEqual(output.shape, self.sample_input.shape)  # Should match input shape

    def test_encoder_forward(self):
        output = self.encoder(self.sample_input)
        self.assertIsNotNone(output)

    def test_decoder_forward(self):
        encoded = self.encoder(self.sample_input)
        output = self.decoder(encoded)
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
