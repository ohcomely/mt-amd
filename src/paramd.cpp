#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <omp.h>
#include "paramd/paramd.h"
#include <random>
#include <vector>


namespace paramd {
// Internal timer for ParAMD
struct timer {
  bool on;
  double start;
  std::map<std::string, double> count;
  timer(bool on) : on(on), start(omp_get_wtime()){}
  void time(const std::string &str) {
    if (on) {
      if (omp_in_parallel()) {
        #pragma omp barrier
        #pragma omp master
        {
          double end = omp_get_wtime();
          count[str] += end - start;
          start = end;
        }
        #pragma omp barrier
      } else {
        double end = omp_get_wtime();
        count[str] += end - start;
        start = end;
      }
    }
  }

  void print() const {
    if (on) {
      for (const auto [v, s] : count)
        std::cout << v << ": " << s << " seconds\n";
    }
  }
};

// Sequential Approximate Degree Lists
struct approximate_degree_lists {
  vtype n, min_deg, cand_end;
  vtype *base_, *head, *next, *prev, *loc, *cand;

  void init(const vtype &n_ = 0) {
    n = n_;
    min_deg = n;
    base_ = (vtype*) std::malloc(sizeof(vtype) * 5 * n);
    std::memset(base_, -1, sizeof(vtype) * 4 * n);
    head = base_;
    next = base_ + n;
    prev = base_ + n * 2;
    loc = base_ + n * 3;
    cand = base_ + n * 4;
  }

  void finalize() {std::free(base_);}

  void remove(const vtype deg, const vtype index) {
    vtype prv = prev[index], nxt = next[index];
    if (nxt != -1)  
      prev[nxt] = prv;
    if (prv != -1)
      next[prv] = nxt;
    else
      head[deg] = nxt;
    loc[index] = -1;
  }
  
  void insert(const vtype deg, const vtype index) {
    if (loc[index] != -1)
      remove(loc[index], index);
    if (head[deg] != -1)
      prev[head[deg]] = index;
    next[index] = head[deg];
    prev[index] = -1;
    head[deg] = index;
    loc[index] = deg;
    min_deg = min_deg > deg ? deg : min_deg;
  }

  vtype get_min_deg(const size_t tid, vtype *affinity) {
    while (min_deg < n) {
      for (vtype index = head[min_deg]; index != -1;) {
        vtype nxt = next[index];  
        if (affinity[index] != tid)
          remove(min_deg, index);
          index = nxt;
      }
      if (head[min_deg] != -1)
        break;
      ++min_deg;
    }
    return min_deg;
  }

  void traverse(const size_t tid, const vtype from, vtype to, vtype *affinity, const vtype lim) {
    cand_end = 0;
    to = to >= n ? n - 1 : to;
    for (vtype deg = from; deg <= to; ++deg) {
      for (vtype index = head[deg]; index != -1;) {
        vtype nxt = next[index];  
        if (affinity[index] != tid)
          remove(deg, index);
        else {
          cand[cand_end++] = index;
          if (cand_end >= lim)
            break;
        }
        index = nxt;
      }
      if (cand_end >= lim)
        break;
    }
  }
};

// Concurrent Approximate Degree Lists
struct concurrent_approximate_degree_lists {
  vtype n;
  approximate_degree_lists *deglists;
  vtype *affinity;
  concurrent_approximate_degree_lists(const vtype& n = 0) : n(n) {
    deglists = (approximate_degree_lists*) std::malloc(sizeof(approximate_degree_lists) * omp_get_max_threads());
    affinity = (vtype*) std::malloc(sizeof(vtype) * n);
    #pragma omp parallel for
    for (vtype i = 0; i < omp_get_max_threads(); ++i)
      deglists[i].init(n);
  }

  ~concurrent_approximate_degree_lists() {
    #pragma omp parallel for
    for (vtype i = 0; i < omp_get_max_threads(); ++i)
      deglists[i].finalize();
    std::free(affinity);
    std::free(deglists);
  }

  void traverse(vtype &min_deg, vtype &num_candidates, vtype *candidates, const double mult, const double lim) {
    const size_t tid = omp_get_thread_num();
    #pragma omp master
    {min_deg = n; num_candidates = 0;}
    #pragma omp barrier
    vtype local_min_deg = deglists[tid].get_min_deg(tid, affinity);
    #pragma omp atomic compare
    min_deg = min_deg > local_min_deg ? local_min_deg : min_deg;
    #pragma omp barrier
    deglists[tid].traverse(tid, min_deg, mult * min_deg, affinity, lim / omp_get_num_threads());
    vtype base = 0;
    #pragma omp atomic capture
    {base = num_candidates; num_candidates += deglists[tid].cand_end;}
    std::memcpy(candidates + base, deglists[tid].cand, sizeof(vtype) * deglists[tid].cand_end);
    #pragma omp barrier
  }

  void insert(const vtype deg, const vtype index) {
    const size_t tid = omp_get_thread_num();
    deglists[tid].insert(deg, index);
    affinity[index] = tid;
  }

  void remove(const vtype deg, const vtype index) {
    affinity[index] = -1;
  }
};

// Hash Lists
struct hashlists {
  vtype n;
  vtype *base_, *head, *next, *hash;

  hashlists(const vtype& n = 0) : n(n) {
    base_ = (vtype*) std::malloc(sizeof(vtype) * n * 3);
    std::memset(base_, -1, sizeof(vtype) * 3 * n);
    head = base_;
    next = base_ + n;
    hash = base_ + n * 2;
  }

  ~hashlists() {std::free(base_);}

  void insert(vtype hsh, vtype index) {
    next[index] = head[hsh];
    head[hsh] = index;
    hash[index] = hsh;
  }

  bool empty(vtype hsh) const {return head[hsh] == -1;}

  bool is_tail(vtype index) const {return next[index] == -1;}

  vtype get_hash(vtype index) const {return hash[index];}

  vtype pop(vtype hsh) {
    vtype index = head[hsh];
    head[hsh] = next[head[hsh]];
    return index;
  }

  vtype get_next(vtype index) const {return next[index];}
  
  void remove(vtype index, vtype prev_index) {
    if (prev_index != -1) {
      next[prev_index] = next[index];
    } else {
      head[hash[index]] = next[index];
    }
  }

  vtype get_nil() const {return -1;}
};

// Clear timestamp
void clear_stp(vtype *stp, const vtype n, vtype &tstp, const vtype tlim) {
  if (tstp < 2 || tstp >= tlim) {
    for (vtype i = 0; i < n; ++i)
      if (stp[i] != 0)
        stp[i] = 1;
    tstp = 2;
  }
}

// A + AT
void symmetrize(const vtype n, etype &free_start, const vtype *rowptr, const etype *colidx,
                etype *&symrowptr, vtype *&neighborhood, const double mem, const bool sym) {
  if (sym) {
    symrowptr = (etype*) std::malloc(sizeof(etype) * (n + 1));
    symrowptr[0] = 0;
    #pragma omp parallel for
    for (vtype i = 0; i < n; ++i) {
      bool diag = false;
      for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
        if (i == colidx[j]) {
          diag = true;
          break;
        }
      symrowptr[i + 1] = rowptr[i + 1] - rowptr[i] - diag;
    }
    for (vtype i = 0; i < n; ++i)
      symrowptr[i + 1] += symrowptr[i];
    free_start = symrowptr[n];
    neighborhood = (vtype*) std::malloc(sizeof(vtype) * free_start * (1 + mem));
    #pragma omp parallel for
    for (vtype i = 0; i < n; ++i) {
      vtype idx = 0;
      for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
        if (i != colidx[j])
          neighborhood[symrowptr[i] + (idx++)] = colidx[j];
    }
    return;
  }

  etype *newrowptr = (etype*) std::malloc(sizeof(etype) * (n + 1));
  symrowptr = (etype*) std::malloc(sizeof(etype) * (n + 1));
  vtype *newcolidx = (vtype*) std::malloc(sizeof(vtype) * rowptr[n] * 2);
  #pragma omp parallel for
  for (vtype i = 0; i <= n; ++i)
    newrowptr[i] = 0;
  const vtype stride = (n + 15) / 16 * 16;
  vtype *cnt_ = (vtype*) std::malloc(sizeof(vtype) * stride * omp_get_max_threads());
  #pragma omp parallel
  {
    const size_t tid = omp_get_thread_num();
    vtype *cnt = cnt_ + tid * stride;
    for (vtype i = 0; i < n; ++i)
      cnt[i] = 0;
    vtype l = n * tid / omp_get_num_threads();
    vtype r = n * (tid + 1) / omp_get_num_threads();
    for (vtype i = l; i < r; ++i)
      for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
        if (i != colidx[j]) {
          ++cnt[i];
          ++cnt[colidx[j]];
        }
    for (vtype i = 0; i < n; ++i) {
      vtype add = cnt[i];
      if (add != 0) {
        #pragma omp atomic capture
        {cnt[i] = newrowptr[i + 1]; newrowptr[i + 1] += add;}
      }
    }
    #pragma omp barrier
    #pragma omp master
    {
      for (vtype i = 0; i < n; ++i) {
        newrowptr[i + 1] += newrowptr[i];
      }
    }
    #pragma omp barrier
    for (vtype i = l; i < r; ++i)
      for (etype j = rowptr[i]; j < rowptr[i + 1]; ++j)
        if (i != colidx[j]) {
          newcolidx[(cnt[i]++) + newrowptr[i]] = colidx[j];
          newcolidx[(cnt[colidx[j]]++) + newrowptr[colidx[j]]] = i; // bottleneck
        }
    #pragma omp barrier
    for (vtype i = l; i < r; ++i) {
      vtype unique = 0, stp = -i - 1;
      for (etype j = newrowptr[i]; j < newrowptr[i + 1]; ++j) {
        if (cnt[newcolidx[j]] != stp) {
          cnt[newcolidx[j]] = stp;
          newcolidx[newrowptr[i] + (unique++)] = newcolidx[j];      
        }
      }
      symrowptr[i + 1] = unique;
    }
    #pragma omp barrier
    #pragma omp master
    {
      symrowptr[0] = 0;
      for (vtype i = 0; i < n; ++i) {
        symrowptr[i + 1] += symrowptr[i];
      }
      free_start = symrowptr[n];
      neighborhood = (vtype*) std::malloc(sizeof(vtype) * free_start * (1 + mem));
    }
    #pragma omp barrier
    for (vtype i = l; i < r; ++i) {
      for (etype j = symrowptr[i]; j < symrowptr[i + 1]; ++j) {
        neighborhood[j] = newcolidx[newrowptr[i] + j - symrowptr[i]];
      }
    }
  }
  std::free(cnt_);
  std::free(newrowptr);
  std::free(newcolidx);
}

// Parallel Approximate Minimum Degree Ordering Algorithm
uint64_t paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config) {
  #define EMPTY (-1)
  #define FLIP(x) (-(x) - 2)
  #define UNFLIP(x) ((x < EMPTY) ? FLIP (x) : (x))
  struct Node {
    etype neighborhood_ptr;

    vtype stp;
    
    vtype neighborhood_len;
    vtype deg;
    
    vtype supernode_size;
    vtype num_hyperedge;
    
    vtype edge_head;
    vtype edge_next;
    vtype order;

    vtype central_pivot;

    uint64_t luby, luby_min;
    uint32_t valid;
    char pad[4];
  };

  constexpr double alpha = 10.0;
  constexpr bool aggressive = true;
  constexpr uint64_t INF = 1ULL << 63;
  
  timer timer(config.breakdown);
  etype free_start = 0;
  etype *symrowptr = nullptr;
  vtype *neighborhood = nullptr;
  symmetrize(n, free_start, rowptr, colidx, symrowptr, neighborhood, config.mem, config.sym);
  
  timer.time("A + AT");

  vtype* const iperm = (vtype*) std::malloc(n * sizeof(vtype));
  vtype* const stk = (vtype*) std::malloc(n * sizeof(vtype));
  vtype* const inv_rank = (vtype*) std::malloc(n * sizeof(vtype));
  vtype* const candidates = (vtype*) std::malloc(n * sizeof(vtype));
  Node* const s = (Node*) std::malloc(n * sizeof(Node));
  
  vtype num_dense = 0, num_eliminated = 0;
  const vtype dense_threshold = std::min(n, std::max(16, vtype(alpha < 0 ? n - 2 : alpha * std::sqrt(n))));

  concurrent_approximate_degree_lists deglists(n);
  vtype min_deg = 1;

  #pragma omp parallel reduction(+: num_eliminated, num_dense)
  {
    #pragma omp for 
    for (vtype i = 0; i < n; ++i) {
      s[i].stp = 1;
      s[i].supernode_size = 1;
      s[i].neighborhood_ptr = symrowptr[i];
      s[i].num_hyperedge = 0;
      s[i].deg = s[i].neighborhood_len = symrowptr[i + 1] - symrowptr[i];
      s[i].edge_head = s[i].edge_next = s[i].order = EMPTY;
      s[i].central_pivot = EMPTY;
    
      iperm[i] = 0;
      stk[i] = EMPTY;
      inv_rank[i] = EMPTY;
      
      if (s[i].deg == 0) {
        s[i].num_hyperedge = FLIP(1);
        ++num_eliminated;
        s[i].neighborhood_ptr = EMPTY;
        s[i].stp = 0;
      } else if (s[i].deg > dense_threshold) {
        ++num_dense;
        ++num_eliminated;
        s[i].neighborhood_ptr = EMPTY;
        s[i].stp = 0;
        s[i].supernode_size = 0;
        s[i].num_hyperedge = EMPTY;
      } else {
        deglists.insert(s[i].deg, i);
      }
    }
  }
  
  uint64_t lnz = uint64_t(num_dense) * (num_dense - 1) / 2;
  vtype num_candidates = 0;

  std::vector<vtype> size_profile;
  vtype size_profile_total = 0;
  
  #pragma omp parallel
  {
    hashlists hashlists(n);
    vtype* const private_ = (vtype*) std::malloc(n * sizeof(vtype) * 3);
    vtype* const private_stp = private_;
    vtype* const workspace = private_ + n;
    vtype* const private_cand = private_ + n * 2;
    std::fill(private_stp, private_stp + n, 1);
    vtype num_private_cand = 0, workspace_end = 0, round = 1;
    vtype private_tstp = 2, private_t_max_step = 0;
    const vtype private_tlim = std::numeric_limits<vtype>::max() - n;
    uint64_t private_lnz = 0;
    std::mt19937 gen(omp_get_thread_num());
    std::uniform_int_distribution<uint64_t> dis(0, n - 1);
    timer.time("Other");
    while (num_eliminated < n) {
      #pragma omp barrier
      deglists.traverse(min_deg, num_candidates, candidates, config.mult, config.lim);
      {
        num_private_cand = 0;
        ++round;
        #pragma omp for
        for (vtype i = 0; i < num_candidates; ++i) {
          const vtype cand = candidates[i];
          private_cand[num_private_cand++] = cand;
          s[cand].luby = dis(gen) << 32 | cand;
          s[cand].stp = round;
          s[cand].valid = 1;
        }
        for (vtype i = 0; i < num_private_cand; ++i) {
          const vtype cand = private_cand[i];
          uint32_t &valid = s[cand].valid;
          s[cand].luby_min = INF;
          const etype hyper_start = s[cand].neighborhood_ptr;
          const etype hyper_end = hyper_start + s[cand].num_hyperedge;
          for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr) {
            const vtype hyper = neighborhood[hyper_ptr];
            const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
            for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr) {
              const vtype nei = neighborhood[nei_ptr];
              if (s[nei].stp != 0) {
                if (s[nei].stp == round && s[nei].luby < s[cand].luby) {
                  valid = false;
                } else if (s[nei].luby_min != INF) {
                  s[nei].luby_min = INF;
                }
              }
            }
          }
          if (!valid)  continue;
          const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
          for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr) {
            const vtype nei = neighborhood[nei_ptr];
            if (s[nei].stp != 0) {
              if (s[nei].stp == round && s[nei].luby < s[cand].luby) {
                valid = false;
              } else if (s[nei].luby_min != INF) {
                s[nei].luby_min = INF;
              }
            }
          }
        }
        #pragma omp barrier
        for (vtype i = 0; i < num_private_cand; ++i) {
          const vtype cand = private_cand[i];
          const uint64_t luby_cand = s[cand].luby;
          uint32_t &valid = s[cand].valid;
          if (!valid)  continue;
          #pragma omp atomic compare
          s[cand].luby_min = s[cand].luby_min > luby_cand ? luby_cand : s[cand].luby_min;
          if (s[cand].luby_min != luby_cand)
            valid = false;
          if (!valid)  continue;
          const etype hyper_start = s[cand].neighborhood_ptr;
          const etype hyper_end = hyper_start + s[cand].num_hyperedge;
          for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr) {
            const vtype hyper = neighborhood[hyper_ptr];
            const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
            for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr) {
              const vtype nei = neighborhood[nei_ptr];
              if (s[nei].stp != 0) {
                #pragma omp atomic compare
                s[nei].luby_min = s[nei].luby_min > luby_cand ? luby_cand : s[nei].luby_min;
                if (s[nei].luby_min != luby_cand)
                  valid = false;
              }
            }
          }
          if (!valid)  continue;
          const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
          for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr) {
            const vtype nei = neighborhood[nei_ptr];
            if (s[nei].stp != 0) {
              #pragma omp atomic compare
              s[nei].luby_min = s[nei].luby_min > luby_cand ? luby_cand : s[nei].luby_min;
              if (s[nei].luby_min != luby_cand)
                valid = false;
            }
          }
        }
        #pragma omp barrier
        vtype resize = 0;
        for (vtype i = 0; i < num_private_cand; ++i) {
          const vtype cand = private_cand[i];
          const uint64_t luby_cand = s[cand].luby;
          bool valid = s[cand].luby_min == luby_cand && s[cand].valid;
          if (!valid)  continue;
          const etype hyper_start = s[cand].neighborhood_ptr;
          const etype hyper_end = hyper_start + s[cand].num_hyperedge;
          for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr) {
            const vtype hyper = neighborhood[hyper_ptr];
            const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
            for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr) {
              const vtype nei = neighborhood[nei_ptr];
              if (s[nei].stp != 0 && s[nei].luby_min != luby_cand) {
                valid = false;
              }
            }
          }
          if (!valid)  continue;
          const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
          for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr) {
            const vtype nei = neighborhood[nei_ptr];
            if (s[nei].stp != 0 && s[nei].luby_min != luby_cand) {
              valid = false;
            }
          }
          if (valid) {
            private_cand[resize++] = cand;
          }
        }
        num_private_cand = resize;
      }
      timer.time("Distance-2 Independent Sets");
      
      if (config.stat) {
        #pragma omp master
        {size_profile_total = 0;}
        #pragma omp barrier
        #pragma omp atomic
        size_profile_total += num_private_cand;
        #pragma omp barrier
        #pragma omp master
        {size_profile.emplace_back(size_profile_total);}  
      }

      vtype private_num_eliminated = 0;   
      workspace_end = 0;
      for (vtype cand_idx = 0; cand_idx < num_private_cand; ++cand_idx) {
        const vtype pivot = private_cand[cand_idx];
        vtype npiv = s[pivot].supernode_size;
        private_num_eliminated += npiv;
        s[pivot].central_pivot = pivot;
        s[pivot].supernode_size = -npiv;
        
        vtype pivot_deg = 0;
        etype new_nei_start = workspace_end;
        deglists.remove(s[pivot].deg, pivot);
        const etype hyper_start = s[pivot].neighborhood_ptr, hyper_end = hyper_start + s[pivot].num_hyperedge; 
        for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr) {
          const vtype hyper = neighborhood[hyper_ptr];
          const etype nei_start = s[hyper].neighborhood_ptr, nei_end = s[hyper].neighborhood_ptr + s[hyper].neighborhood_len;
          for (etype nei_ptr = nei_start; nei_ptr < nei_end; ++nei_ptr) {
            const vtype nei = neighborhood[nei_ptr];
            const vtype nei_size = s[nei].supernode_size;
            if (nei_size > 0) {
              pivot_deg += nei_size;
              s[nei].supernode_size = -nei_size;
              workspace[workspace_end++] = nei;
              s[nei].central_pivot = pivot;
              deglists.remove(s[nei].deg, nei);
            }
          }
          s[hyper].neighborhood_ptr = FLIP(pivot);
          s[hyper].stp = 0; 
        }
        const etype super_end = s[pivot].neighborhood_ptr + s[pivot].neighborhood_len;
        for (etype nei_ptr = hyper_end; nei_ptr < super_end; ++nei_ptr) {
          const vtype nei = neighborhood[nei_ptr];
          const vtype nei_size = s[nei].supernode_size;
          if (nei_size > 0) {
            pivot_deg += nei_size;
            s[nei].supernode_size = -nei_size;
            workspace[workspace_end++] = nei;
            s[nei].central_pivot = pivot;
            deglists.remove(s[nei].deg, nei);
          }
        }
        etype new_nei_end = workspace_end;
        
        s[pivot].neighborhood_ptr = new_nei_start;
        s[pivot].num_hyperedge = FLIP(npiv + pivot_deg);
        clear_stp(private_stp, n, private_tstp, private_tlim);
        private_t_max_step = 0;
        
        for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr) {
          const vtype nei = workspace[nei_ptr];
          const etype hyper_start = s[nei].neighborhood_ptr, hyper_end = s[nei].neighborhood_ptr + s[nei].num_hyperedge;
          for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr) {
            const vtype hyper = neighborhood[hyper_ptr];
            if (private_stp[hyper] >= private_tstp) {
              private_stp[hyper] += s[nei].supernode_size;
            } else if (s[hyper].stp != 0) {
              private_stp[hyper] = s[hyper].deg + private_tstp + s[nei].supernode_size;
              private_t_max_step = std::max(private_t_max_step, s[hyper].deg);
            }
          }
        }

        for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr) {
          const vtype nei = workspace[nei_ptr];
          const etype hyper_start = s[nei].neighborhood_ptr, hyper_end = s[nei].neighborhood_ptr + s[nei].num_hyperedge;
          etype new_end = hyper_start;
          vtype nei_deg = 0;
          vtype hash = 0;
          for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr) {
            const vtype hyper = neighborhood[hyper_ptr];
            if (s[hyper].stp != 0) {
              const vtype external_deg = private_stp[hyper] - private_tstp;
              if (!aggressive || external_deg > 0) {
                nei_deg += external_deg;
                neighborhood[new_end++] = hyper;
                hash += hyper;
              } else {
                s[hyper].neighborhood_ptr = FLIP(pivot);
                s[hyper].stp = 0;
              }
            }
          }
          s[nei].num_hyperedge = new_end - hyper_start + 1;
          const etype super_end = hyper_start + s[nei].neighborhood_len;
          const etype super_start = new_end;
          for (etype super_ptr = hyper_end; super_ptr < super_end; ++super_ptr) {
            const vtype super = neighborhood[super_ptr];
            if (s[super].supernode_size != 0 && s[super].central_pivot != pivot) {
              nei_deg += std::abs(s[super].supernode_size); 
              neighborhood[new_end++] = super;
              hash += super;
            }
          }

          if (s[nei].num_hyperedge == 1 && new_end == super_start) {
            s[nei].neighborhood_ptr = FLIP(pivot);
            const vtype nei_size = -s[nei].supernode_size;
            pivot_deg -= nei_size;
            private_num_eliminated += nei_size;
            npiv += nei_size;
            s[nei].num_hyperedge = EMPTY;
            s[nei].supernode_size = 0;
          } else {
            s[nei].deg = std::min(s[nei].deg, nei_deg);
            neighborhood[new_end] = neighborhood[super_start];
            neighborhood[super_start] = neighborhood[hyper_start];
            neighborhood[hyper_start] = pivot;
            s[nei].neighborhood_len = new_end - hyper_start + 1;
            hash = (hash % n);
            if (hash < 0) hash += n;
            hashlists.insert(hash, nei);
          }
        }

        s[pivot].deg = pivot_deg;
        s[pivot].supernode_size = npiv;
        private_tstp += private_t_max_step;
        clear_stp(private_stp, n, private_tstp, private_tlim);

        for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr) {
          const vtype nei = workspace[nei_ptr];
          if (s[nei].supernode_size < 0) {
            const vtype hash = hashlists.get_hash(nei);
            while (!hashlists.empty(hash)) {
              const vtype nei_i = hashlists.pop(hash);
              const vtype nei_len_i = s[nei_i].neighborhood_len;
              const vtype nhe_i = s[nei_i].num_hyperedge;
              if (hashlists.empty(hash))  break;
              for (etype nei_i_ptr = s[nei_i].neighborhood_ptr + 1; nei_i_ptr < s[nei_i].neighborhood_ptr + nei_len_i; ++nei_i_ptr) {
                private_stp[neighborhood[nei_i_ptr]] = private_tstp;
              }
              vtype nei_j = nei_i, prev_nei_j = hashlists.get_nil();
              while (!hashlists.is_tail(nei_j)) {
                nei_j = hashlists.get_next(nei_j);
                const vtype nei_len_j = s[nei_j].neighborhood_len;
                const vtype nhe_j = s[nei_j].num_hyperedge;
                bool same = (nei_len_i == nei_len_j) && (nhe_i == nhe_j);
                for (etype nei_j_ptr = s[nei_j].neighborhood_ptr + 1; nei_j_ptr < s[nei_j].neighborhood_ptr + nei_len_j; ++nei_j_ptr) {
                  same &= (private_stp[neighborhood[nei_j_ptr]] == private_tstp);
                }
                if (same) {
                  s[nei_j].neighborhood_ptr = FLIP(nei_i);
                  s[nei_i].supernode_size += s[nei_j].supernode_size;
                  s[nei_j].supernode_size = 0;
                  s[nei_j].num_hyperedge = EMPTY;
                  hashlists.remove(nei_j, prev_nei_j);
                } else {
                  prev_nei_j = nei_j;
                }
              }
              ++private_tstp;
            }
          }
        }

        etype final_nei_end = new_nei_start;
        for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr) {
          const vtype nei = workspace[nei_ptr];
          const vtype nei_size = -s[nei].supernode_size;
          if (nei_size > 0) {
            s[nei].supernode_size = nei_size;
            s[nei].deg = std::min(s[nei].deg + pivot_deg - nei_size, n - num_eliminated - nei_size);
            workspace[final_nei_end++] = nei;
            deglists.insert(s[nei].deg, nei);
          }
        }
        s[pivot].neighborhood_len = final_nei_end - new_nei_start;
        workspace_end = final_nei_end;
        private_lnz += uint64_t(s[pivot].supernode_size) * (s[pivot].deg + num_dense) + uint64_t(s[pivot].supernode_size) * (s[pivot].supernode_size - 1) / 2;
        if (s[pivot].neighborhood_len == 0) {
          s[pivot].neighborhood_ptr = EMPTY;
          s[pivot].stp = 0;
        }
      }
      
      etype base = 0;
      #pragma omp atomic capture
      {base = free_start; free_start += workspace_end;}
      std::copy(workspace, workspace + workspace_end, neighborhood + base);
        
      for (vtype cand_idx = 0; cand_idx < num_private_cand; ++cand_idx) {
        const vtype pivot = private_cand[cand_idx];
        if (s[pivot].neighborhood_len != 0)
          s[pivot].neighborhood_ptr += base;
      }
  
      #pragma omp atomic
      num_eliminated += private_num_eliminated;
      #pragma omp barrier
      timer.time("Core");
    }
    #pragma omp atomic
    lnz += private_lnz;
    std::free(private_);
  }

  #pragma omp parallel for
  for (vtype i = 0; i < n; ++i) {
    s[i].neighborhood_ptr = FLIP(s[i].neighborhood_ptr);
    s[i].num_hyperedge = FLIP(s[i].num_hyperedge);
  }

  #pragma omp parallel for
  for (vtype i = 0; i < n; ++i) {
    if (s[i].supernode_size == 0) {
      vtype p = s[i].neighborhood_ptr;
      if (p != EMPTY) {
        while(s[p].supernode_size == 0)
          p = s[p].neighborhood_ptr;
        const vtype hyper = p;
        for (vtype ptr = i; ptr != hyper;) {
          p = s[ptr].neighborhood_ptr;
          s[ptr].neighborhood_ptr = hyper;
          ptr = p;
        }
      }
    }
  }

  #pragma omp parallel for
  for (vtype i = n - 1; i >= 0; --i) {
    if (s[i].supernode_size > 0) {
      vtype p = s[i].neighborhood_ptr;
      if (p != EMPTY) {
        #pragma omp atomic capture
        {
          s[i].edge_next = s[p].edge_head; s[p].edge_head = i;
        }
      }
    }
  }

  #pragma omp parallel for
  for (vtype i = 0; i < n; ++i) {
    if (s[i].supernode_size > 0 && s[i].edge_head != EMPTY) {
      vtype prev = EMPTY, mxsz = EMPTY, mxprev = EMPTY, mxptr = EMPTY;
      for (vtype j = s[i].edge_head; j != EMPTY; j = s[j].edge_next) {
        if (s[j].num_hyperedge >= mxsz) {
          mxsz = s[j].num_hyperedge;
          mxprev = prev;
          mxptr = j;
        }
        prev = j;
      } 
      
      if (s[mxptr].edge_next != EMPTY) {
        if (mxprev == EMPTY)
          s[i].edge_head = s[mxptr].edge_next;
        else
          s[mxprev].edge_next = s[mxptr].edge_next;
        s[mxptr].edge_next = EMPTY;
        s[prev].edge_next = mxptr;
      }
    }
  }

  vtype stk_head = 0, k = 0;

  for (vtype i = 0; i < n; ++i) {
    if (s[i].neighborhood_ptr == EMPTY && s[i].supernode_size > 0) {
      stk[stk_head = 0] = i;
      while (stk_head != EMPTY) {
        vtype cur = stk[stk_head];
        if (s[cur].edge_head != EMPTY) {
          for (vtype j = s[cur].edge_head; j != EMPTY; j = s[j].edge_next) {
            ++stk_head;
          }
          for (vtype j = s[cur].edge_head, h = stk_head; j != EMPTY; j = s[j].edge_next) {
            stk[h--] = j;
          }
          s[cur].edge_head = EMPTY;
        } else {
          --stk_head;
          s[cur].order = k++;
        }
      }
    }
  }

  #pragma omp parallel for
  for (vtype i = 0; i < n; ++i) {
    if (s[i].order != EMPTY) {
      inv_rank[s[i].order] = i;
    }
  }

  num_eliminated = 0;

  for (vtype i = 0; i < n; ++i) {
    vtype hyper = inv_rank[i];
    if (hyper == EMPTY) break;
    iperm[hyper] = num_eliminated;
    num_eliminated += s[hyper].supernode_size;
  }

  #pragma omp parallel for
  for (vtype i = 0; i < n; ++i) {
    if (s[i].supernode_size == 0) {
      if (s[i].neighborhood_ptr != EMPTY) {
        #pragma omp atomic capture
        {
          iperm[i] = iperm[s[i].neighborhood_ptr]; ++iperm[s[i].neighborhood_ptr];
        }
      } else {
        #pragma omp atomic capture
        {
          iperm[i] = num_eliminated; ++num_eliminated;
        }
      }
    }
  }

  #pragma omp parallel for
  for (vtype i = 0; i < n; ++i) {
    perm[iperm[i]] = i;
  }
  
  std::free(s);
  std::free(candidates);
  std::free(inv_rank);
  std::free(stk);
  std::free(iperm);
  std::free(neighborhood);
  std::free(symrowptr);

  timer.time("Other");
  timer.print();
  
  if (config.stat) {
    std::cout << "Size of distance-2 independent sets: [";
    for (auto x : size_profile)
      std::cout << x << ", ";
    std::cout << "]\n";
  }

  return lnz;
  
  #undef UNFLIP
  #undef FLIP
  #undef EMPTY
}

} // end of namespace paramd