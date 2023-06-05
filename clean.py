import torch

torch.cuda.empty_cache()
cached_memory = torch.cuda.memory_cached()
print("当前缓存的CUDA显存:", cached_memory)