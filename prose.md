# Background

After implementing the Transformer model and running it on PTB dataset, we found that it often leads to Out-of-Memory when trained using large batch size and Adam Opimizer. Further exploration leads us to the conclusion that in many transformer-based models, memory issue is one of the most common problem when the number of parameters in the model increases. For example, in OpenAI’s 64 layers and 4 heads Transformer, the attentio memory usage is 9.6 GB for CIFAR-10 dataset with 32x32x3 pixel, 154 GB for the ImageNet dataset with 64x64x3 pixels. The amount of RAM needed for GPU to train the model is way beyond many GPU's ability in a non-commercial usage setting. Even for a few paragraphs's text with 1024 tokens, it would take 1 GB to store the intermediate embeddings. While many of the memory issue can be solve by tuning down the batch size, it is not always desirable. A very small batch size causes each batch to be a more “noisy” representation of the entire dataset, will cause a sort of “tug-and-pull” dynamic and hence leads to greater instability during training. In the meantime, smaller batch size often means longer training time as gradient is updated more frequently. Hence allowing a non-trivial batch size to be trained every iteration is often needed, and overcoming the memory bottleneck is necessary to achieve this.

# Existing solutions

We found several ways to reduce memory usages:

- **Mixed-Precision Training**: Mixed precision training uses both 16-bit and 32-bit precision without accuracy degradation. We can compute gradient and forward pass in 16-bit representation which is much faster than the 32-bit version, and do the conversion before/after gradient comutation with a master copy of the weights is stored in 32-bit. This trick has been implemented in PyToch with the Automated Mixed Precision library.
- **Lower-Precision Training**: Instead of mixed precision training, we can also directly train with the low-precision representation at a cost of accuracy. This saves memory and can be as efficient as the original training. The key idea is to convert the 32-bit representation to a reliable Brain Floating Point (`bfloat16`).
- **Gradient Accumulation**: Gradient accumulation is a way to virtually increase the batch size during training. It essentially waits for a few iterations before updating the weights. For instance, in the following code:

```python
for batch_idx, batch in enumerate(train_loader):
	model.train()

	### FORWARD AND BACK PROP
	outputs = model(
		batch["input_ids"],
		attention_mask=batch["attention_mask"],
		labels=batch["label"]
	)

	outputs["loss"] = outputs["loss"] / accumulation_steps # accumulate loss
	fabric.backward(outputs["loss"])

	### UPDATE MODEL PARAMETERS
	if not batch_idx % accumulation_steps or (batch_idx + 1 == len(train_loader)):
		# update every accumulation_steps iterations
		optimizer.step()
		optimizer.zero_grad()
```

All we've done is to scale the loss down by a factor of `accumulation_steps`, add the gradients throughout the `accumulation_steps` iterations, and update the weights after that.

- **Distributed Training**: With multiple GPUs, there are often better ways to utilize GPU memory with data/tensor parallelism. One example is the use of FullyShardedDataParallel in PyTorch (`torch.distributed.fsdp`), which mainly serves to scale up model training across nodes but brings the benefit of less idle memory at any point in time.
- **Parameter Offloading**: When training large models with Adam optimizer, the optimizer state often takes up significant amount of GPU memory. Consequently, we can move the unused optimizer states to CPU and load it back to GPU when we need it.
- **Model initialization at GPU at target precision**: When we instantiate a model in PyTorch, we usually create it on the CPU device first, and then we transfer it onto the target device and convert it to the desired precision. This can be inefficient considering the intermediate model representation in full precision on the CPU. Instead, we can directly create the model in desired precision on the target device (e.g., GPU). Several packages have implemented this method. For example, Lightning AI has the `Fabric` library with the `init_module()` function that directly loads the model into GPU.
- **Gradient Checkpointing**[1]: Gradient checkpointing is another popular method used to save memory. It trades computation with memory. It selectively drops some of activations/weights from certain layers during forward/backward pass, and recomputes them on demand during backward. 

# Gradient Checkpointing

## PyTorch Implementation

In PyTorch, we have the `torch.utils.checkpoint` module which can apply checkpointing on the entire model (as a list of `nn.module`) or part of the model. In its implementation, a context manager is used on the function level to save and load input/output tensors as well as arguments across multiple modules for backward propagation.

## Our software architecture and design

We implemented gradient checkpoing to build memory-efficient Transformer, which is not included in the original gradient checkpointing paper [1] and can be considered new. More specifically, we applied gradient checkpoing to Layer Norm, ReLU, Linear, and Multi-head Attention layer in Transformer.

Innn_transformer.py1 layer Transformer: [TO add Transformer] 191 nodes -> 89 are part of ReLU / LayerNorm1d

With Needle, we first enabled lazy evaluation so that we won't save activations during forward computation until `realize_cached_data` is called. Since we set `LAZY_MODE` as True, we completed `make_from_op` method to enable the correct function of lazy evaluation. Then, we annotate tensors within module of interest. In `autograd.py`, we added a bool `drop` field whose default is False to the `Value` class. For all nodes in the modules, we add and set a `drop` property to True. In `nn_basic.py`, we wrapped up annotating nodes into an `annotate(out_node, in_nodes)` method for clean interface. We added a `gc` filed and `enable_gc` method to the `Module` class.In classes of layers we want to apply gradient checkpointing, we simply need to check that , we call `annotate`methodtoannotatenodesinthelayerif `gc` is True.

[To add Segment]

In forward propagation, we don’t save `cached_data` for those nodes and recompute them when necessary in backward. We achieved this by modifying `realize_cached_data` and `compute_gradient_of_variables` methods in `autograd.py`. The gif below illustrautes a process of our designed gradient checkpointing.

![https://github.com/Criss-Wang/needle/blob/main/demo.gif](demo.gif)

We measure GPU Peak memory usage with `GPUtil` and time cost with `time.time()`. 

In `gc.py`, we wrote the codes of running experiments. This includes initializing Transformer LanguageModel, loading data, training the Transformer model, setting various arguments, and output results.

# Experiments 

We tested our model performance on PTB dataset with batch_size = 256, seq_len = 20, hidden size = 32. We trained our model for 3 epochs using Adam Optimizer and reported the average results across three runs.
We evaluated Peak GPU memory usage during batch training (max. among epoch) and average time cost of each batch.

# Results and Discussions

# Conclusion and Future Work
