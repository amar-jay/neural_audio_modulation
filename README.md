# Naa I screwed this up, mixed the channel dimensions with the sequence lenghth. ITS WRONG. not what I claims to be.

I initially though that MLP with positional encoding could be a much better alternative to the convolutional layers. But it turns out that the convolutional layers are much better at condensing the information into a much smaller representation. The MLP with positional encoding is much more expensive to train since it requires a very large weight matrix to be trained. The smaller weight of the convolutional layer is what gives it an advantage over the MLP.

Hmm, its mea culpa for not checking the dimensions in the first place. I should have checked the dimensions of my input before training.