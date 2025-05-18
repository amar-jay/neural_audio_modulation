# Neural Audio Modulation Compression

This project focuses on developing a neural network for audio compression with a specific emphasis on modulation techniques. The goal is to create an efficient model that can encode and decode audio signals while maintaining high quality.

## Process
The model was trained on esc50 dataset, which contains 50 different classes of environmental sounds. The preprocessing wasnt extensvie and kept as minimal as possible using only min-max normalization. 

However after testing, it was validated on both 20% fo esc50 and the entire of libriSpeech dataset. The results were promising achieving 38 SNR, 43 PSNR, and 0.9758 STOI on the libriSpeech dataset.
On performing validation on libriSpeech, since all the audio have varied lengths and sample rates, I resampled each to be 44.1kHz (the sample rate of the esc50 dataset on which the model was trained), and trimmed it to 20500 samples (0.5 seconds). However, when the sample rate wasn't 44.1kHz, it gave lower SNR and PSNR values. The model finds it difficult to generalize to different sample rates and lengths. However, this is a promising result since the model was trained on classic feed-forward networks with sinusoidal positional encodings(similar to transformers) and the results were promising.
We didn't use any convolutional layers across it. The total number of parameters in the model is really tiny (4943 parameters) at full precision. This is the audio itself is larger than the model. The model is really small and can be used for real-time applications, if scaled.


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
