#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "knn.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("knn_check_version", &KnnCheckVersion);
    m.def("knn_points_idx", &KNearestNeighborIdx);
    m.def("knn_points_backward", &KNearestNeighborBackward);
}
