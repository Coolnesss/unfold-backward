#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace at;

template <typename scalar_t>
__global__ void unfold_backward_cuda_kernel(
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ result,
    const int64_t grad_size,
    const int64_t result_size,
    const int64_t size,
    const int64_t step
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= result_size) return;

    // Used to determine where, in the result, the element at index i
    // maps to for the first time
    int start_row = 0;
    int start_col = i;

    if (i >= size) {
        if ((i-size) % step == 0) inc++;

        start_row = inc;
        start_col = col_inc;

        col_inc++;
        if (col_inc >= size) col_inc = size - step;
    }

    // Iterates over the positions of element i in the result
    int row = start_row;
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
        grad.size(0),
        input_elements,
        size,
        step
        );
  }));

  return result;
}
