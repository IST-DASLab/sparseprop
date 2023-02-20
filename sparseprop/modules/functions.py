import torch
import math

from sparseprop import backend as sppb

TRANSPOSE_BLOCK_SIZE = 16

class SparseLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputT, W_val, W_idx, bias, N):
        input_flat_t = inputT.reshape(-1, inputT.shape[-1])
        B, M = input_flat_t.shape
        if B % TRANSPOSE_BLOCK_SIZE == 0 and M % TRANSPOSE_BLOCK_SIZE == 0:  
            input_flat = torch.zeros(M, B)
            sppb.transpose(input_flat_t, input_flat, TRANSPOSE_BLOCK_SIZE)
        else:
            input_flat = input_flat_t.t().contiguous()
        ctx.inputT_shape = inputT.shape

        M, B = input_flat.shape

        W_idx_N, W_idx_M = W_idx

        output = torch.zeros(N, B).float()
        sppb.sparse_linear_vectorized_forward(input_flat, W_idx_N, W_idx_M, W_val, output)

        ctx.save_for_backward(W_val, bias)
        ctx.svd = (input_flat, W_idx_N, W_idx_M)

        if bias is not None:
            output += bias.view(-1, 1)

        if B % TRANSPOSE_BLOCK_SIZE == 0 and N % TRANSPOSE_BLOCK_SIZE == 0:  
            output_t = torch.zeros(B, N)
            sppb.transpose(output, output_t, TRANSPOSE_BLOCK_SIZE)
        else:
            output_t = output.t() # (B, N)
        output_t = output_t.reshape(*ctx.inputT_shape[:-1], N)
        return output_t

    @staticmethod
    def backward(ctx, grad_output_t):
        W_val, bias = ctx.saved_tensors
        input_flat, W_idx_N, W_idx_M = ctx.svd

        grad_output_t = grad_output_t.reshape(-1, grad_output_t.shape[-1]).contiguous()
        B, N = grad_output_t.shape
        if B % TRANSPOSE_BLOCK_SIZE == 0 and N % TRANSPOSE_BLOCK_SIZE == 0:
            grad_output = torch.zeros(N, B)
            sppb.transpose(grad_output_t, grad_output, TRANSPOSE_BLOCK_SIZE)
        else:
            grad_output = grad_output_t.t().contiguous()

        grad_input = torch.zeros_like(input_flat).float().contiguous() # (M, B)
        grad_W_val = torch.zeros_like(W_val).float().contiguous()
        
        sppb.sparse_linear_vectorized_backward(
            input_flat,
            W_idx_N,
            W_idx_M,
            W_val,
            grad_output,
            grad_input,
            grad_W_val
        )

        M = input_flat.shape[0]
        if B % TRANSPOSE_BLOCK_SIZE == 0 and M % TRANSPOSE_BLOCK_SIZE == 0:  
            grad_input_t = torch.zeros(B, M)
            sppb.transpose(grad_input, grad_input_t, TRANSPOSE_BLOCK_SIZE)
        else:
            grad_input_t = grad_input.t() # (B, M)
        grad_input_t = grad_input_t.reshape(ctx.inputT_shape)
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output_t.sum([i for i in range(len(grad_output_t.shape) - 1)])

        return grad_input_t, grad_W_val, None, grad_bias, None


class SparseConvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, W_val, W_idx, bias, OC, K, padding, stride, vectorizing_fwd_over_on, vectorizing_bwd_over_on):
        orig_input = input
        
        assert stride in [1, 2], 'only strides 1 and 2 are supported'

        B, IC, M, N = orig_input.shape
        OM = math.ceil((M + 2 * padding - K + 1) / stride)
        ON = math.ceil((N + 2 * padding - K + 1) / stride)
        
        if vectorizing_fwd_over_on:
            assert stride == 1 # only stride 1 is supported in this case, for now
            output = torch.zeros(B, OC, OM, ON).float()
            sppb.sparse_conv2d_vectorized_forward_over_on_stride_1(input, *W_idx, W_val, output, K, padding)
        else:
            input = input.permute(1, 2, 3, 0).contiguous()
            output = torch.zeros(OC, OM, ON, B).float()
            if stride == 1:
                sppb.sparse_conv2d_vectorized_forward_stride_1(input, *W_idx, W_val, output, K, padding)
            elif stride == 2:
                sppb.sparse_conv2d_vectorized_forward_stride_2(input, *W_idx, W_val, output, K, padding)

            output = output.permute(3, 0, 1, 2)
        
        if vectorizing_bwd_over_on: # backward needs the original shape
            ctx.save_for_backward(W_val, bias)
            ctx.svd = (orig_input, *W_idx)
        else:
            ctx.save_for_backward(W_val, bias)
            ctx.svd = (input, *W_idx)
        ctx.K, ctx.padding, ctx.stride = K, padding, stride
        ctx.vectorizing_bwd_over_on = vectorizing_bwd_over_on
        
        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_val, bias = ctx.saved_tensors
        input, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y = ctx.svd
        K, padding, stride = ctx.K, ctx.padding, ctx.stride
        
        vectorizing_bwd_over_on = ctx.vectorizing_bwd_over_on
        
        grad_input = torch.zeros_like(input).float()
        grad_W_val = torch.zeros_like(W_val).float()

        assert stride in [1, 2], 'only stride 1 and 2 are supported'
                
        if vectorizing_bwd_over_on:
            if stride == 1:
                sppb.sparse_conv2d_vectorized_backward_over_on_stride_1(input, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, grad_output, grad_input, grad_W_val, K, padding)
            else:
                sppb.sparse_conv2d_vectorized_backward_over_on_stride_2(input, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, grad_output, grad_input, grad_W_val, K, padding)
        else:
            go = grad_output.permute(1, 2, 3, 0).contiguous()
            if stride == 1:
                sppb.sparse_conv2d_vectorized_backward_stride_1(input, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, go, grad_input, grad_W_val, K, padding)
            else:
                sppb.sparse_conv2d_vectorized_backward_stride_2(input, W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y, W_val, go, grad_input, grad_W_val, K, padding)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        if not vectorizing_bwd_over_on:
            grad_input = grad_input.permute(3, 0, 1, 2)

        return grad_input, grad_W_val, None, grad_bias, None, None, None, None, None, None

