#include "autograd/tensor.hpp"
#include <algorithm>
#include <cstring>

namespace autograd {

// ── private helpers ───────────────────────────────────────────────────────────

void Tensor::alloc_heap() {
  heap_data_ = new float[size()];
}

void Tensor::copy_shape(const Tensor &other) noexcept {
  ndim_ = other.ndim_;
  std::memcpy(shape_, other.shape_, sizeof(shape_));
}

// ── size / batch_size ─────────────────────────────────────────────────────────

uint32_t Tensor::size() const noexcept {
  if (ndim_ == 0) return 1;
  uint32_t s = 1;
  for (uint8_t i = 0; i < ndim_; i++) s *= shape_[i];
  return s;
}

uint32_t Tensor::batch_size() const noexcept {
  if (ndim_ <= 2) return 1;
  uint32_t s = 1;
  for (uint8_t i = 0; i + 2 < ndim_; i++) s *= shape_[i];
  return s;
}

// ── constructors ──────────────────────────────────────────────────────────────

Tensor::Tensor() noexcept : ndim_(0) {
  std::memset(shape_, 0, sizeof(shape_));
  inline_data_[0] = 0.0f;
}

Tensor::Tensor(float scalar) : ndim_(2) {
  shape_[0] = 1; shape_[1] = 1; shape_[2] = 0; shape_[3] = 0;
  inline_data_[0] = scalar;
}

Tensor::Tensor(uint32_t rows, uint32_t cols) : ndim_(2) {
  shape_[0] = rows; shape_[1] = cols; shape_[2] = 0; shape_[3] = 0;
  if (!is_inline()) alloc_heap();
  std::fill(raw_data(), raw_data() + size(), 0.0f);
}

Tensor::Tensor(uint32_t rows, uint32_t cols, const float *data) : ndim_(2) {
  shape_[0] = rows; shape_[1] = cols; shape_[2] = 0; shape_[3] = 0;
  if (!is_inline()) alloc_heap();
  std::memcpy(raw_data(), data, size() * sizeof(float));
}

Tensor::Tensor(std::span<const uint32_t> shape) {
  assert(shape.size() <= kMaxDims);
  ndim_ = static_cast<uint8_t>(shape.size());
  std::memset(shape_, 0, sizeof(shape_));
  std::copy(shape.begin(), shape.end(), shape_);
  if (!is_inline()) alloc_heap();
  std::fill(raw_data(), raw_data() + size(), 0.0f);
}

Tensor::Tensor(std::span<const uint32_t> shape, const float *data) {
  assert(shape.size() <= kMaxDims);
  ndim_ = static_cast<uint8_t>(shape.size());
  std::memset(shape_, 0, sizeof(shape_));
  std::copy(shape.begin(), shape.end(), shape_);
  if (!is_inline()) alloc_heap();
  std::memcpy(raw_data(), data, size() * sizeof(float));
}

Tensor::Tensor(const Eigen::MatrixXf &m) : ndim_(2) {
  shape_[0] = static_cast<uint32_t>(m.rows());
  shape_[1] = static_cast<uint32_t>(m.cols());
  shape_[2] = 0; shape_[3] = 0;
  if (!is_inline()) alloc_heap();
  std::memcpy(raw_data(), m.data(), size() * sizeof(float));
}

// ── rule of five ──────────────────────────────────────────────────────────────

Tensor::Tensor(const Tensor &other) {
  copy_shape(other);
  if (!is_inline()) alloc_heap();
  std::memcpy(raw_data(), other.raw_data(), size() * sizeof(float));
}

Tensor::Tensor(Tensor &&other) noexcept {
  copy_shape(other);
  if (is_inline()) {
    std::memcpy(inline_data_, other.inline_data_, size() * sizeof(float));
  } else {
    heap_data_       = other.heap_data_;
    other.heap_data_ = nullptr;
  }
  other.ndim_ = 0;
  std::memset(other.shape_, 0, sizeof(other.shape_));
}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this == &other) return *this;
  if (!is_inline()) delete[] heap_data_;
  copy_shape(other);
  if (!is_inline()) alloc_heap();
  std::memcpy(raw_data(), other.raw_data(), size() * sizeof(float));
  return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this == &other) return *this;
  if (!is_inline()) delete[] heap_data_;
  copy_shape(other);
  if (is_inline()) {
    std::memcpy(inline_data_, other.inline_data_, size() * sizeof(float));
  } else {
    heap_data_       = other.heap_data_;
    other.heap_data_ = nullptr;
  }
  other.ndim_ = 0;
  std::memset(other.shape_, 0, sizeof(other.shape_));
  return *this;
}

Tensor::~Tensor() {
  if (!is_inline()) delete[] heap_data_;
}

// ── factories ─────────────────────────────────────────────────────────────────

Tensor Tensor::ones(uint32_t rows, uint32_t cols) {
  Tensor t(rows, cols);
  std::fill(t.raw_data(), t.raw_data() + t.size(), 1.0f);
  return t;
}

Tensor Tensor::zeros(uint32_t rows, uint32_t cols) {
  return Tensor(rows, cols);
}

// ── Eigen bridge ──────────────────────────────────────────────────────────────

Eigen::Map<Eigen::MatrixXf> Tensor::map() noexcept {
  return Eigen::Map<Eigen::MatrixXf>(raw_data(), rows(), cols());
}

Eigen::Map<const Eigen::MatrixXf> Tensor::map() const noexcept {
  return Eigen::Map<const Eigen::MatrixXf>(raw_data(), rows(), cols());
}

Eigen::Map<Eigen::MatrixXf> Tensor::matrix_slice(uint32_t batch) noexcept {
  return Eigen::Map<Eigen::MatrixXf>(
      raw_data() + batch * rows() * cols(), rows(), cols());
}

Eigen::Map<const Eigen::MatrixXf> Tensor::matrix_slice(uint32_t batch) const noexcept {
  return Eigen::Map<const Eigen::MatrixXf>(
      raw_data() + batch * rows() * cols(), rows(), cols());
}

} // namespace autograd
