import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

model_id = "Qwen/Qwen2.5-1.5B"
print(f"Loading {model_id} on CPU...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

text = "The quick brown fox jumps over the lazy dog. " * 50
inputs = tokenizer(text, return_tensors="pt").to("cpu")

# 1. We will store the intercepted tensor here
intercepted_keys = []

# 2. Define a PyTorch Hook to rip the tensor straight out of the math operation
def hook_fn(module, input, output):
    intercepted_keys.append(output.detach().clone())

# 3. Attach it directly to Layer 0's Key Projection node
handle = model.model.layers[0].self_attn.k_proj.register_forward_hook(hook_fn)

print("Running forward pass to intercept keys directly from PyTorch...")
with torch.no_grad():
    model(**inputs)

# 4. Remove the hook
handle.remove()

# 5. Process the raw tensor
# The raw shape is (batch_size, seq_len, num_kv_heads * head_dim)
raw_keys = intercepted_keys[0].squeeze() # Removes batch size. Shape becomes: (seq_len, 256)

# Grab just the first attention head (dim 0 to 128) to match our C++ d=128
keys_f32 = raw_keys[:, :128].float().numpy()

# Save to binary
filename = "qwen_cache.bin"
keys_f32.tofile(filename)

seq_len, head_dim = keys_f32.shape
print(f"Success! Intercepted {seq_len} tokens of dimension {head_dim}. Saved to {filename}.")