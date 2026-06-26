---
library_name: kernels
license: apache-2.0
---

This is the repository card of kernels-community/finegrained-fp8 that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels). This card was automatically generated.

## How to use

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("kernels-community/finegrained-fp8")
fp8_act_quant = kernel_module.fp8_act_quant

fp8_act_quant(...)
```

## Available functions
- `fp8_act_quant`
- `w8a8_fp8_matmul`
- `w8a8_block_fp8_matmul`
- `w8a8_tensor_fp8_matmul`
- `w8a8_fp8_matmul_batched`
- `w8a8_block_fp8_matmul_batched`
- `w8a8_tensor_fp8_matmul_batched`
- `w8a8_fp8_matmul_grouped`
- `w8a8_block_fp8_matmul_grouped`
- `w8a8_tensor_fp8_matmul_grouped`

## Benchmarks

No benchmark available yet.
