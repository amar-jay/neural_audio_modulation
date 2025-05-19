# Naa I screwed this up, mixed the channel dimensions with the sequence lenghth. ITS WRONG. not what I claims to be.
# Neural Audio Modulation Compression

This project focuses on developing a neural network for audio compression with a specific emphasis on modulation techniques. The goal is to create an efficient model that can encode and decode audio signals while maintaining high quality.

## Process
The model was trained on the ESC-50 dataset, which contains 50 classes of environmental sounds. Preprocessing was kept minimal, involving only min-max normalization. For evaluation, we used a 20% validation split of ESC-50 and the entire LibriSpeech dataset. On LibriSpeech, the model achieved 38 dB SNR, 43 dB PSNR, and a STOI score of 0.9758.

Since LibriSpeech samples vary in length and sample rate, we resampled all audio to 44.1 kHz (matching ESC-50) and trimmed each to 20,500 samples (0.5 seconds). However, performance degraded on inputs with different sample rates, indicating the model's difficulty in generalizing to unseen formats.

Notably, the model consists of only 4,943 parameters in full precision—smaller in size than the audio input itself—making it well-suited for real-time deployment. It is a fully feedforward network using sinusoidal positional encodings (inspired by transformer architectures), without any convolutional layers. Given its simplicity and size, these results are promising.


## Results
Will paste the results here from wandb later. but for now this is the average metrics on validation on libriSpeech dataset.

```
-----------------------------
Metric Name     |  Value
-----------------------------
SNR             |  tensor(38.0669)
MSE             |  0.0000
PSNR            |  tensor(43.6499)
PD              |  tensor(0.0052)
STOI            |  0.9911
embedding_loss  |  tensor(0.0008)
perplexity      |  tensor(2.6795)
loss            |  tensor(0.)
-----------------------------
```


### Conclusion

The model is a promising start for neural audio modulation compression. The results on the libriSpeech dataset are encouraging, and with further tuning and optimization, it could be a viable solution for real-time audio compression tasks. Some improvement could be made, such as using a much better metrics for benchmarking the model, rather than SNR/PSNR/STOI, benchmark against MP3/AAC/Opus codecs if possible.
The model is small and can be used for real-time applications, if scaled. The model is not yet ready for production use, but it is a good starting point for further research and development in the field of neural audio modulation compression.  
