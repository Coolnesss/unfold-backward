#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace at;

template <typename scalar_t>
__global__ void unfold_backward_cuda_kernel(
    const scalar_t* __restrict__ grad,
    const int64_t* __restrict__ input_sizes,
    const int64_t dim,
    const int64_t size,
    const int64_t step
) {
    
  }

Tensor unfold_backward_cuda(
    const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step
) {

  const int threads = 1024;
  const dim3 blocks(2);

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "unfold_backward_cuda", ([&] {
    unfold_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        grad.data<scalar_t>(),
        input_sizes,
        dim,
        size,
        step
        );
  }));

  return zeros({20}, grad.options());
}
