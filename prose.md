# Memory-efficient Transformer with Gradient Checkpointing

# Background

After implementing the Transformer model and running it on PTB dataset, we found that it often leads to Out-of-Memory when trained using large batch size and Adam Opimizer. Further exploration leads us to the conclusion that in many transformer-based models, memory issue is one of the most common problem when the number of parameters in the model increases. For example, in OpenAI's 64 layers and 4 heads Transformer, the attentio memory usage is 9.6 GB for CIFAR-10 dataset with 32x32x3 pixel, 154 GB for the ImageNet dataset with 64x64x3 pixels. The amount of RAM needed for GPU to train the model is way beyond many GPU's ability in a non-commercial usage setting. Even for a few paragraphs's text with 1024 tokens, it would take 1 GB to store the intermediate embeddings. While many of the memory issue can be solve by tuning down the batch size, it is not always desirable. A very small batch size causes each batch to be a more “noisy” representation of the entire dataset, will cause a sort of “tug-and-pull” dynamic and hence leads to greater instability during training. In the meantime, smaller batch size often means longer training time as gradient is updated more frequently. Hence allowing a non-trivial batch size to be trained every iteration is often needed, and overcoming the memory bottleneck is necessary to achieve this.

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
- **Gradient Checkpointing** [1]: Gradient checkpointing is another popular method used to save memory. It trades computation with memory. It selectively drops some of activations/weights from certain layers during forward/backward pass, and recomputes them on demand during backward. It claims to reduce the memory cost to O($\sqrt n$) when training a n layer network. 

# Gradient Checkpointing

## PyTorch Implementation

In PyTorch, we have the `torch.utils.checkpoint` module [2] which can apply checkpointing on the entire model (as a list of `nn.module`) or part of the model. In its implementation, a context manager is used on the function level to save and load input/output tensors as well as arguments across multiple modules for backward propagation.

## Our software architecture and design

We implemented gradient checkpoing to build memory-efficient Transformer, which is not included in the original gradient checkpointing paper [1] and thus can be considered as new. Since transformers often contain many layers and it is very memory intensive to store the output $N \times N$ attention matrices from every intermediate layers of multi-head attention, we applied gradient checkpoing to Multi-head Attention layer, Layer Norm, ReLU, and Linear layers in Transformer.

* **Support gradient checkpointing for a wide range of necessary layers**: With Needle, we first enabled lazy evaluation so that we won't save activations during forward computation until `realize_cached_data` is called. Since we set `LAZY_MODE` as True, we completed `make_from_op` method to enable the correct function of lazy evaluation. Then, we annotate tensors within module of interest. In `autograd.py`, we added a bool `drop` field whose default is False to the `Value` class. For all nodes in the modules, we add and set a `drop` property to True. In `nn_basic.py`, we wrapped up annotating nodes into an `annotate(out_node, in_nodes)` method for clean interface. We added a `gc` filed and `enable_gc` method to the `Module` class. Thanks to class inheritance, in classes of layers we want to apply gradient checkpointing, we simply need to call `annotate` method to annotate nodes in the layer if `gc` is True. As shown below, we enable selectively drop weights of nodes in layers by keeping activations of nodes every `segment_len`.
	```
	# utils for gradient checkpointing
	def annotate(out_node, in_nodes, cur_index=0, segment_len=None):
			for in_node in in_nodes:
					if out_node is in_node:
							return

			for node in out_node.inputs:
					if segment_len:
							node.drop = node.op is not None and (cur_index == 0 or cur_index % segment_len != 0)
					else:
							node.drop = node.op is not None
					
					annotate(node, in_nodes, cur_index=cur_index+1, segment_len=segment_len)
	```

	To ensure correct computation, we need to carefully handle dropout layers. More specifically, we separate gradient checkpointing to layers before and after each DropOut by passing corresponding input and output nodes into `annotate`. An example is given below. We added annotation to provide a complete support of gradient checkpointing for `Embedding`, `ReLU`, `LayerNorm1d`, `MultiHeadAttention`, `AttentionLayer`, `TransformerLayer`, `Linear` layers and `LanguageModel`. 
	```
	# An example to show how we handle drop out. Below is the forward method of class MultiHeadAttention(Module).

	def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        d = q_dim
        device = q.device
        QK_T = self.matmul(q, k)
        if self.causal:
            mask = self.create_causal_mask(
                keys_values_len, queries_len, device)
            mask = mask.broadcast_to(QK_T.shape)
        else:
            mask = init.zeros(*QK_T.shape, device=QK_T.device, dtype=QK_T.dtype)
        
        attn = self.softmax(QK_T / d ** .5 + mask)
        
        if self.gc:
            annotate(attn, (q, k), cur_index=1, segment_len=self.segment_len)

        probs = self.dropout(attn)
        result = self.matmul(probs, v.transpose((2,3)))
        
        if self.gc:
            annotate(result, (probs, v), cur_index=1, segment_len=self.segment_len)
        
        ### END YOUR SOLUTION

        return result, probs
	```

* 	**Drop in forward and recompute in backward**: The gif below illustrautes a process of our designed gradient checkpointing.
	![https://github.com/Criss-Wang/needle/blob/main/demo.gif](demo.gif). 

	In forward propagation, we don't save `cached_data` for those nodes and recompute them when necessary in backward. We achieved this by modifying `realize_cached_data` and `compute_gradient_of_variables` methods in `autograd.py` as follows. 
	```
	def compute_gradient_of_variables(output_tensor, out_grad):
			"""Take gradient of output node with respect to each node in node_list.

			Store the computed result in the grad field of each Variable.
			"""
			# a map from node to a list of gradient contributions from each output node
			node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
			# Special note on initializing gradient of
			# We are really taking a derivative of the scalar reduce_sum(output_node)
			# instead of the vector output_node. But this is the common case for loss function.
			node_to_output_grads_list[output_tensor] = [out_grad]

			# Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
			reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

			# used when lazy mode is off (gradient checkpointing) / cached data is realized due to some ops under lazy mode when 
			for node in reverse_topo_order:
					if node.drop:
							node.cached_data = None

			gc = sum([node.drop for node in reverse_topo_order])
			
			# BEGIN YOUR SOLUTION
			for ind, node in enumerate(reverse_topo_order):
					v_i = sum_node_list(node_to_output_grads_list[node])

					# Store the computed result in the grad field of each Variable.
					node.grad = v_i
					for input in node.inputs:
							if input not in node_to_output_grads_list:
									node_to_output_grads_list[input] = []

					if node.is_leaf():
							continue

					gradients = node.op.gradient_as_tuple(v_i, node)

					for i, input in enumerate(node.inputs):
							node_to_output_grads_list[input].append(gradients[i])

					if gc > 0 and ind > 0:
							for i in range(ind):
									if reverse_topo_order[i].op is not None:
											reverse_topo_order[i].cached_data = None

			# END YOUR SOLUTION
	```
	```
	def realize_cached_data(self):
					"""Run compute to realize the cached data"""
					# avoid recomputation
					if self.cached_data is not None:
							return self.cached_data

					# note: data implicitly calls realized cached data
					if self.drop:
							return self.op.compute(*[x.realize_cached_data() for x in self.inputs])
					else:
							self.cached_data = self.op.compute(
									*[x.realize_cached_data() for x in self.inputs]
							)
							return self.cached_data
	```

* **Training memory-efficient transformer**: In `nn_transformer.py`, we implemented the memory efficient Transformer model. A one-layer Transformer contains 191 nodes, among which 89 nodes from ReLU and LayerNorm1d.

	In `gc.py`, we wrote the codes of running experiments. This includes initializing Transformer LanguageModel, loading data, training the Transformer model, setting various arguments, and output results. We instantiated a language model with a paraeter `use_gc` to set whether we enable gradient checkpointing or not. In `models.py`, we implemented a language model consisting of an embedding layer, a Transformer model, and a linear layer. The Transformer model is one-layer with embedding_size = 20, hidden_size = 32, and seq_len = 20.

	In `simple_ml.py`, we implemented `epoch_general_ptb`, `train_ptb`, and `evaluate_ptb`. We measured GPU Peak memory usage with `GPUtil` and time cost with `time.time()` for each batch of training. 

# Experiments 

We tested our model performance on PTB dataset with batch_size = 256 and GPU device. We also varied batch size, segment length, and which layers to apply gradient checkpointing to compare performance.

For each experiment, we trained our model for 3 epochs using Adam Optimizer and learning rate 0.003. We evaluated Peak GPU memory usage during batch training (max. among epoch) and average time cost of each batch. We reported the average results across three runs.

# Results and Discussions

# Conclusion
Our gradient checkpointing effectively reduce GPU memory usage of training Transformer-based language models by > 20% saving. We support gradient checkpointing of a complete list of modules and layers in the Transformer model. We offer users a easy-to-use interface and allowed users to selectively segment nodes in layers and select which layers to apply gradient checkpointing. We wrapped up our implementation in an elegant and clean organization and layout. We conducted extensive experiments and ablation study to demonstrate our large memory saving and compare the increase in time cost.

# Reference
[1]  T. Chen, B. Xu, C. Zhang, and C. Guestrin, "Training deep nets with
sublinear memory cost," 2016. Available: arXiv:1604.06174.

[2] https://pytorch.org/docs/stable/checkpoint.html
