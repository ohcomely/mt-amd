#include <iostream>
#include <omp.h>
#include "paramd/paramd.h"

namespace paramd {
  // Default constructor for config
  config::config() {
    mult = 1.1;
    lim = 8192;
    mem = 1.5;
    seed = 1;
    breakdown = false;
    stat = false;
    sym = false;
  }

  // Print config
  void config::print() const {
    std::cout << "Multiplicative relaxation factor: " << mult << "\n";
    std::cout << "Limitation factor: " << lim << "\n";
    std::cout << "Extra memory factor: " << mem << "\n";
    std::cout << "Seed: " << seed << "\n";
    std::cout << "Time breakdown enabled: " << (breakdown ? "Yes" : "No") << "\n";
    std::cout << "Statistics enabled: " << (stat ? "Yes" : "No") << "\n";
    std::cout << "Symmetry enabled: " << (stat ? "Yes" : "No") << "\n";
    std::cout << "Number of threads: " << omp_get_max_threads() << "\n";
  }
} // end of namespace paramd
