#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <parallel/algorithm>
#include "paramd/paramd.h"
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>


namespace paramd {
  // Construct a matrix from fp
  matrix::matrix(std::string fp) {
    int fd = open(fp.c_str(), O_RDONLY);
    if (fd == -1) {
      std::cerr << "Cannot open file: " << fp << std::endl;
      exit(1);
    }

    struct stat st;
    if (fstat(fd, &st)) {
      std::cerr << "Cannot get file stat: " << fp << std::endl;
      exit(1);
    }
    size_t fsize = st.st_size;

    char *fdata = (char *) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (!fdata) {
      std::cerr << "Cannot mmap: " << fp << std::endl;
      exit(1);
    }

    std::istringstream banner(std::string(fdata, std::find(fdata, fdata + fsize, '\n')));
    std::vector<std::string> banner_str;
    std::string line;
    while (banner >> line)  banner_str.emplace_back(line);
    
    if (banner_str[2] != "coordinate") {
      std::cerr << "Only accepting sparse/coordinate matrix!" << std::endl;
      exit(1);
    }
    
    if (banner_str.back() != "general" && banner_str.back() != "symmetric" && banner_str.back() != "hermitian") {
      std::cerr << "Only accepting general or symmetric/hermitian matrix!" << std::endl;
      exit(1);
    }

    bool sym = banner_str.back() == "symmetric" || banner_str.back() == "hermitian";
    
    auto get_integer = [](char *&ptr) {
      uint64_t res = 0;
      while (*ptr < '0' || *ptr > '9')  ++ptr;
      while (*ptr >= '0' && *ptr <= '9') {
        res = res * 10 + ((*ptr) - '0');
        ++ptr;
      }
      return res;
    };

    char *firstline = fdata;
    while(*firstline == '%')  firstline = std::find(firstline, fdata + fsize, '\n') + 1;
    
    {
      uint64_t nrows = get_integer(firstline);
      if (nrows > std::numeric_limits<vtype>::max()) {
        std::cerr << "Number of rows can not fit into vtype as " << sizeof(vtype) << " bytes!" << std::endl;
        exit(1);
      }
      
      uint64_t ncols = get_integer(firstline);
      if (ncols > std::numeric_limits<vtype>::max()) {
        std::cerr << "Number of cols can not fit into vtype as " << sizeof(vtype) << " bytes!" << std::endl;
        exit(1);
      }

      uint64_t nonzeros = get_integer(firstline);
      if (nonzeros > std::numeric_limits<etype>::max()) {
        std::cerr << "Number of nonzeros can not fit into etype as " << sizeof(etype) << " bytes!" << std::endl;
        exit(1);
      }

      if (nrows != ncols) {
        std::cerr << "Number of rows != Number of cols!" << std::endl;
        exit(1);
      }
      n = nrows;
      nnz = nonzeros;
    }

    std::vector<std::vector<std::pair<vtype, vtype>>> local_edges(omp_get_max_threads());
    const size_t block_size = 1 << 22;
    const size_t num_blocks = (fsize + block_size - 1) / block_size;
    #pragma omp parallel for
    for (size_t i = 0; i < num_blocks; ++i) {
      char *st = fdata + i * block_size, *ed = fdata + std::min(fsize, (i + 1) * block_size);
      size_t tid = omp_get_thread_num();
      if (st != fdata && *(st - 1) != '\n')
        st = std::find(st, ed, '\n') + 1;
      if (st < ed) {
        if (*(ed - 1) != '\n')
          ed = std::find(ed, fdata + fsize, '\n') + 1;
        while (st != ed) {
          if (st >= firstline) {
            vtype u = get_integer(st), v = get_integer(st);
            local_edges[tid].emplace_back(u - 1, v - 1);
            if (sym && u != v)
              local_edges[tid].emplace_back(v - 1, u - 1);
          }
          st = std::find(st, fdata + fsize, '\n') + 1;
        }
      }
    }

    if (munmap(fdata, fsize)) {
      std::cerr << "Cannot munmap: " << fp << std::endl;
      exit(1);
    }

    if (close(fd)) {
      std::cerr << "Cannot close file: " << fp << std::endl;
      exit(1);
    }

    std::vector<size_t> gather_size(omp_get_max_threads() + 1);
    for (size_t i = 0; i < omp_get_max_threads(); ++i) {
      gather_size[i + 1] = gather_size[i] + local_edges[i].size();
    }
    std::vector<std::pair<vtype, vtype>> global_edges(gather_size.back()), unique_global_edges(gather_size.back());
    #pragma omp parallel
    {
      size_t tid = omp_get_thread_num();
      std::copy(local_edges[tid].begin(), local_edges[tid].end(), global_edges.begin() + gather_size[tid]);
    }
    for (size_t i = 0; i < omp_get_max_threads(); ++i) {
      local_edges[i] = std::vector<std::pair<vtype, vtype>>();
    }
    __gnu_parallel::sort(global_edges.begin(), global_edges.end());
    unique_global_edges.resize(__gnu_parallel::unique_copy(global_edges.begin(), global_edges.end(), unique_global_edges.begin()) - unique_global_edges.begin());
    global_edges = std::vector<std::pair<vtype, vtype>>();
    rowptr = std::vector<etype>(n + 1);
    rowptr.back() = unique_global_edges.size();
    colidx.resize(rowptr.back());
    nnz = rowptr.back();
    
    #pragma omp parallel for
    for (etype i = 0; i < (etype) unique_global_edges.size(); ++i) {
      colidx[i] = unique_global_edges[i].second;
      if (i == 0 || unique_global_edges[i].first != unique_global_edges[i - 1].first) {
        rowptr[unique_global_edges[i].first] = i;
      }
    }
    for (vtype i = n; i > 1; --i)
      if (!rowptr[i - 1])
        rowptr[i - 1] = rowptr[i];
  }

  // Apply iperm symmetrically to the matrix
  void matrix::apply_inverse_permutation(const std::vector<vtype> &iperm) {
    std::vector<etype> newrowptr(rowptr.size());
    std::vector<vtype> newcolidx(colidx.size());
    
    #pragma omp parallel for
    for (vtype i = 0; i < n; ++i)
      newrowptr[iperm[i] + 1] = rowptr[i + 1] - rowptr[i];
    
    for (vtype i = 0; i < n; ++i)
      newrowptr[i + 1] += newrowptr[i];
    
    #pragma omp parallel for
    for (vtype i = 0; i < n; ++i) {
      vtype ni = iperm[i];
      etype newbase = newrowptr[ni];
      for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
        newcolidx[newbase + j - rowptr[i]] = iperm[colidx[j]];
      std::sort(newcolidx.begin() + newrowptr[ni], newcolidx.begin() + newrowptr[ni + 1]);
    }
    rowptr.swap(newrowptr);
    colidx.swap(newcolidx);
  }
} // end of namespace paramd
