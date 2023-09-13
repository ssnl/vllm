from vllm import quantization_ops
from vllm.config import WeightQuantizationConfig


def awq_linear(input_, qweight, scales, qzeros, bias=None, *, quant_config: WeightQuantizationConfig):
    if quant_config.method == 'awq_gemm':
        out_shape = (input_.shape[-2], qweight.shape[-1] * quant_config.pack_factor)
        out = quantization_ops.gemm_forward_cuda(
            input_.reshape(-1, input_.shape[-1]), qweight, scales, qzeros,
            quant_config.pack_factor)
    elif quant_config.method == 'awq_gemv':
        out_shape = (input_.shape[-2], qweight.shape[0])
        out = quantization_ops.gemv_forward_cuda(
            input_.reshape(-1, input_.shape[-1]), qweight, scales, qzeros,
            quant_config.group_size)
    else:
        raise ValueError(f'Unknown quantization method: {quant_config.method}')
    out = out + bias if bias is not None else out
    return out.reshape(out_shape)
