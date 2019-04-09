#include <torch/extension.h>
#include <iostream>
#include <vector>

using namespace at;
using namespace std;

// Copypaste current version
Tensor unfold_backward_real(const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {

    int64_t numel = 1;
    for (auto size : input_sizes) {
        numel *= size;
    }

    auto idx = at::arange(0, numel, grad.options().dtype(at::kLong)).view(input_sizes);
    auto idx_unfolded = idx.unfold(dim, size, step).contiguous().view(-1);
    auto grad_input = at::zeros({numel}, grad.options());
    grad_input.index_add_(0, idx_unfolded, grad.contiguous().view(-1));
    return grad_input.view(input_sizes);
}

// New version
Tensor unfold_backward_cpu(const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {

    int64_t numel = 1;
    for (auto size : input_sizes) {
        numel *= size;
    }
    /*
    int64_t before = 1;
    for (int i = 0; i < dim; i++) before *= input_sizes[i];
    int64_t after = 1;
    for (int i = dim+1; i < input_sizes.size(); i++) after *= input_sizes[i];
    */

    auto grad_view = grad.view(-1);
    auto grad_accessor = grad_view.accessor<float, 1>();
    
    auto grad_input = at::zeros({numel}, grad.options());
    auto grad_input_accessor = grad_input.accessor<float, 1>();
    
    int n_rows = grad.size(0);
    int n_cols = grad.size(1);

    int inc = 0;
    int col_inc = size - step;

    for (int i = 0; i < numel; i++) {
        
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
            grad_input_accessor[i] += grad_accessor[output_index];
            row++;   
        }
    }

    return grad_input.view(input_sizes);
}

at::Tensor unfold_backward_cuda(
    const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step
);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor unfold_backward(
    const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step
) {

  CHECK_INPUT(grad);
  return unfold_backward_cuda(grad, input_sizes, dim, size, step);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("real_backward", &unfold_backward_real, "unfold real backward");
  m.def("backward", &unfold_backward, "unfold backward (CUDA)");
  m.def("backward_cpu", &unfold_backward_cpu, "unfold backward (CPU)");
}
