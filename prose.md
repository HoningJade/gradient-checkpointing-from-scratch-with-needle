# Background
After implementing the Transformer model and running it on PTB dataset, we found that it often leads to Out-of-Memory when trained using large batch size and Adam Opimizer. Further exploration leads us to the conclusion that in many transformer-based models, memory issue is one of the most common problem when the number of parameters in the model increases. For example, in OpenAI’s 64 layers and 4 heads Transformer, the attentio memory usage is 9.6 GB for CIFAR-10 dataset with 32x32x3 pixel, 154 GB for the ImageNet dataset with 64x64x3 pixels. The amount of RAM needed for GPU to train the model is way beyond many GPU's ability in a non-commercial usage setting. Even for a few paragraphs's text with 1024 tokens, it would take 1 GB to store the intermediate embeddings. While many of the memory issue can be solve by tuning down the batch size, it is not always desirable. A very small batch size causes each batch to be a more “noisy” representation of the entire dataset, will cause a sort of “tug-and-pull” dynamic and hence leads to greater instability during training. In the meantime, smaller batch size often means longer training time as gradient is updated more frequently. Hence allowing a non-trivial batch size to be trained every iteration is often needed, and overcoming the memory bottleneck is necessary to achieve this.

# Existing solutions
We found several ways to reduce memory usages:
- Mixed-Precision Training
- Lower-Precision Training


# PyTorch Implementation


# Model Architecture
