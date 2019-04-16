#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace at;

template <typename scalar_t>
__global__ void unfold_backward_cuda_kernel(
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ result,
    const int64_t result_size,
    const int64_t n_rows,
    const int64_t n_cols,
    const int64_t size,
    const int64_t step
) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= result_size) return;

    int row = 0;
    int start_col = i;
    int col_inc = size - step;

    if (i >= size) {
        row = (i - size) / step + 1;    
        col_inc = size - step + (i - size) % step;
        start_col = col_inc;
    }
    
    // Iterates over the positions of element i in the result
    for (int col = start_col; col >= 0; col-=step) {
        if (row >= n_rows) break;

        int output_index = n_cols * row + col;
        result[i] += grad[output_index];
        row++;   
    }
  }

Tensor unfold_backward_cuda(
    const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step
) {

  int64_t input_elements = input_sizes[0];
  auto result = at::zeros(input_sizes, grad.options());

  const int threads = 1024;
  const dim3 blocks((input_elements / threads) + 1);
  
  AT_DISPATCH_FLOATING_TYPES(grad.type(), "unfold_backward_cuda", ([&] {
    unfold_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        grad.data<scalar_t>(),
        result.data<scalar_t>(),
        input_elements,
        grad.size(0),
        grad.size(1),
        size,
        step
        );
  }));

  return result;
}
