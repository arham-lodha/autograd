#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

namespace autograd {

// N-dimensional tensor with Small Buffer Optimization.
// Up to kMaxDims dimensions; shape is stored inline.
// Up to kInline floats are stored inline (no heap); larger tensors
// heap-allocate. Data is column-major in the last two dimensions to match
// Eigen::MatrixXf.
class Tensor {
public:
  static constexpr uint32_t kInline = 24;
  static constexpr uint8_t kMaxDims = 4;

  // ── Construction
  // ────────────────────────────────────────────────────────────
  Tensor() noexcept;
  explicit Tensor(float scalar);                              // shape {1,1}
  Tensor(uint32_t rows, uint32_t cols);                       // 2D, zero-init
  Tensor(uint32_t rows, uint32_t cols, const float *data);    // 2D, copy data
  Tensor(std::span<const uint32_t> shape);                    // n-D, zero-init
  Tensor(std::span<const uint32_t> shape, const float *data); // n-D, copy data
  explicit Tensor(const Eigen::MatrixXf &m);

  Tensor(const Tensor &);
  Tensor(Tensor &&) noexcept;
  Tensor &operator=(const Tensor &);
  Tensor &operator=(Tensor &&) noexcept;
  ~Tensor();

  // ── Factories
  // ───────────────────────────────────────────────────────────────
  static Tensor ones(uint32_t rows, uint32_t cols);
  static Tensor zeros(uint32_t rows, uint32_t cols);

  // ── Shape
  // ───────────────────────────────────────────────────────────────────
  uint8_t ndim() const noexcept { return ndim_; }
  uint32_t shape(uint8_t i) const noexcept { return shape_[i]; }
  uint32_t size() const noexcept;       // product of all dims
  uint32_t batch_size() const noexcept; // product of dims 0..ndim-3
  bool is_scalar() const noexcept { return size() == 1; }

  // Convenience: last two dims treated as a matrix.
  uint32_t rows() const noexcept { return ndim_ >= 2 ? shape_[ndim_ - 2] : 1; }
  uint32_t cols() const noexcept { return ndim_ >= 1 ? shape_[ndim_ - 1] : 1; }

  // ── Data access
  // ─────────────────────────────────────────────────────────────
  float *data() noexcept { return raw_data(); }
  const float *data() const noexcept { return raw_data(); }

  // Element access into the last-two-dim matrix of the first batch slice.
  float operator()(uint32_t r, uint32_t c) const noexcept {
    return raw_data()[r + c * rows()];
  }
  float &operator()(uint32_t r, uint32_t c) noexcept {
    return raw_data()[r + c * rows()];
  }

  // ── Eigen bridge
  // ──────────────────────────────────────────────────────────── map() views
  // the first batch slice (or the whole tensor if ndim <= 2).
  Eigen::Map<Eigen::MatrixXf> map() noexcept;
  Eigen::Map<const Eigen::MatrixXf> map() const noexcept;

  // matrix_slice(b) views the b-th batch slice as a (rows x cols) matrix.
  Eigen::Map<Eigen::MatrixXf> matrix_slice(uint32_t batch) noexcept;
  Eigen::Map<const Eigen::MatrixXf> matrix_slice(uint32_t batch) const noexcept;

  Eigen::MatrixXf to_eigen() const { return map(); }

private:
  uint8_t ndim_ = 0;
  uint32_t shape_[kMaxDims] = {};
  union {
    float inline_data_[kInline];
    float *heap_data_;
  };

  bool is_inline() const noexcept { return size() <= kInline; }
  float *raw_data() noexcept { return is_inline() ? inline_data_ : heap_data_; }
  const float *raw_data() const noexcept {
    return is_inline() ? inline_data_ : heap_data_;
  }
  void alloc_heap();
  void copy_shape(const Tensor &other) noexcept;
};

} // namespace autograd
