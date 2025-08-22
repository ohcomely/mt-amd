#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <immintrin.h>

namespace paramd
{
  constexpr size_t CACHE_LINE_SIZE = 64;

  // Cache-aligned allocation helper
  template <typename T>
  T *aligned_alloc_cache(size_t count)
  {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, count * sizeof(T)) != 0)
    {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

// Cache line aligned struct
#define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)

  typedef int32_t vtype;
  typedef int32_t etype;

  // config used for paramd
  struct config
  {
    double mult, lim, mem;
    uint64_t seed;
    bool breakdown, stat, sym;

    // Default constructor for config
    config();

    // Print config
    void print() const;
  };

  // matrix storage
  struct matrix
  {
    vtype n;
    etype nnz;
    std::vector<etype> rowptr;
    std::vector<vtype> colidx;

    // Construct a matrix from fp
    matrix(std::string fp);

    // Apply iperm symmetrically to the matrix
    void apply_inverse_permutation(const std::vector<vtype> &iperm);
  };

  // Generate a length n random inverse permutation from seed
  std::vector<vtype> random_inverse_permutation(const vtype n, const uint64_t seed);

  // Get the inverse permutation of a permutation
  std::vector<vtype> inverse_permutation(const std::vector<vtype> &perm);

  // Apply function f to a vector src
  std::vector<vtype> apply_function(const std::vector<vtype> &src, const std::vector<vtype> &f);

  // Write permutation to fp
  void write_perm(std::string fp, const std::vector<vtype> &perm);

  // Parallel Approximate Minimum Degree Ordering Algorithm
  uint64_t paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config);

  // Optimized Parallel Approximate Minimum Degree Ordering Algorithm
  uint64_t paramd_optimized(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config);
} // end of namespace paramd