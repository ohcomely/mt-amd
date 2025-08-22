#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include "paramd/paramd.h"
#include <random>
#include <vector>

namespace paramd {
// Generate a length n random inverse permutation from seed
std::vector<vtype> random_inverse_permutation(const vtype n, const uint64_t seed) {
  std::vector<vtype> iperm(n);
  std::iota(iperm.begin(), iperm.end(), vtype(0));
  if (seed != 0) {
    std::mt19937_64 gen64(seed);
    std::shuffle(iperm.begin(), iperm.end(), gen64);
  }
  return iperm;
}

// Get the inverse permutation of a permutation
std::vector<vtype> inverse_permutation(const std::vector<vtype> &perm) {
  std::vector<vtype> iperm(perm.size());
  for (vtype i = 0; i < perm.size(); ++i)
    iperm[perm[i]] = i;
  return iperm;
}

// Apply function f to a vector src
std::vector<vtype> apply_function(const std::vector<vtype> &src, const std::vector<vtype> &f) {
  std::vector<vtype> res(src.size());
  for (vtype i = 0; i < src.size(); ++i)
    res[i] = f[src[i]];
  return res;
}

// Write permutation to fp
void write_perm(std::string fp, const std::vector<vtype> &perm) {
  std::ofstream output(fp);
  std::vector<bool> chk(perm.size());

  for (vtype x : perm) {
    if (x < 0 || x >= perm.size()) {
      std::cerr << "Invalid permutation (not within range)!" << std::endl;
      exit(1); 
    }

    if (chk[x]) {
      std::cerr << "Invalid permutation (duplicates)!" << std::endl;
      exit(1); 
    }
    
    chk[x] = true;
  }

  for (vtype x : perm)
    output << x << '\n';
  
  output.close();
}

} // end of namespace paramd