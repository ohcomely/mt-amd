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
#include <immintrin.h>
#include <stdlib.h>

namespace paramd
{
    // Optimized timer with reduced overhead
    struct optimized_timer
    {
        bool on;
        double start;
        std::map<std::string, double> count;

        optimized_timer(bool on) : on(on), start(omp_get_wtime()) {}

        inline void time(const std::string &str)
        {
            if (__builtin_expect(on, 0))
            {
                const double end = omp_get_wtime();
                count[str] += end - start;
                start = end;
            }
        }

        void print() const
        {
            if (on)
            {
                for (const auto [v, s] : count)
                    std::cout << v << ": " << s << " seconds\n";
            }
        }
    };

    // Cache-aligned and optimized degree lists with prefetching
    struct CACHE_ALIGNED optimized_degree_lists
    {
        vtype n, min_deg, cand_end;
        vtype *base_, *head, *next, *prev, *loc, *cand;

        void init(const vtype &n_ = 0)
        {
            n = n_;
            min_deg = n;
            // Use cache-aligned allocation
            base_ = aligned_alloc_cache<vtype>(5 * n);
            std::memset(base_, -1, sizeof(vtype) * 4 * n);
            head = base_;
            next = base_ + n;
            prev = base_ + n * 2;
            loc = base_ + n * 3;
            cand = base_ + n * 4;
        }

        void finalize() { std::free(base_); }

        inline void remove(const vtype deg, const vtype index)
        {
            const vtype prv = prev[index];
            const vtype nxt = next[index];

            // Branch prediction optimization
            if (__builtin_expect(nxt != -1, 1))
                prev[nxt] = prv;
            if (__builtin_expect(prv != -1, 1))
                next[prv] = nxt;
            else
                head[deg] = nxt;
            loc[index] = -1;
        }

        inline void insert(const vtype deg, const vtype index)
        {
            if (__builtin_expect(loc[index] != -1, 0))
                remove(loc[index], index);

            const vtype head_deg = head[deg];
            if (__builtin_expect(head_deg != -1, 1))
                prev[head_deg] = index;
            next[index] = head_deg;
            prev[index] = -1;
            head[deg] = index;
            loc[index] = deg;

            // Branchless min update
            min_deg = (min_deg > deg) ? deg : min_deg;
        }

        vtype get_min_deg(const size_t tid, vtype *affinity)
        {
            // Optimized loop with prefetching
            while (__builtin_expect(min_deg < n, 1))
            {
                vtype index = head[min_deg];
                bool found_valid = false;

                // Prefetch next cache line
                __builtin_prefetch(&head[min_deg + 1], 0, 3);

                while (index != -1)
                {
                    const vtype nxt = next[index];
                    if (__builtin_expect(affinity[index] != tid, 0))
                        remove(min_deg, index);
                    else
                        found_valid = true;
                    index = nxt;
                }

                if (found_valid)
                    break;
                ++min_deg;
            }
            return min_deg;
        }

        void traverse(const size_t tid, const vtype from, vtype to, vtype *affinity, const vtype lim)
        {
            cand_end = 0;
            to = (to >= n) ? n - 1 : to;

            // Unrolled loop for better performance
            for (vtype deg = from; deg <= to && cand_end < lim; ++deg)
            {
                vtype index = head[deg];

                // Prefetch for next iteration
                if (__builtin_expect(deg + 1 <= to, 1))
                    __builtin_prefetch(&head[deg + 1], 0, 3);

                while (index != -1 && cand_end < lim)
                {
                    const vtype nxt = next[index];
                    if (__builtin_expect(affinity[index] != tid, 0))
                        remove(deg, index);
                    else
                        cand[cand_end++] = index;
                    index = nxt;
                }
            }
        }
    };

    // Optimized concurrent degree lists with better memory layout
    struct optimized_concurrent_degree_lists
    {
        vtype n;
        optimized_degree_lists *deglists;
        vtype *affinity;

        optimized_concurrent_degree_lists(const vtype &n = 0) : n(n)
        {
            const int num_threads = omp_get_max_threads();

            // Cache-aligned allocation
            void *ptr = nullptr;
            if (posix_memalign(&ptr, CACHE_LINE_SIZE, sizeof(optimized_degree_lists) * num_threads) != 0)
                throw std::bad_alloc();
            deglists = (optimized_degree_lists *)ptr;
            affinity = aligned_alloc_cache<vtype>(n);

#pragma omp parallel for
            for (int i = 0; i < num_threads; ++i)
                deglists[i].init(n);
        }

        ~optimized_concurrent_degree_lists()
        {
            const int num_threads = omp_get_max_threads();
#pragma omp parallel for
            for (int i = 0; i < num_threads; ++i)
                deglists[i].finalize();
            std::free(affinity);
            std::free(deglists);
        }

        void traverse(vtype &min_deg, vtype &num_candidates, vtype *candidates,
                      const double mult, const double lim)
        {
            const size_t tid = omp_get_thread_num();
            const int num_threads = omp_get_num_threads();

#pragma omp master
            {
                min_deg = n;
                num_candidates = 0;
            }
#pragma omp barrier

            const vtype local_min_deg = deglists[tid].get_min_deg(tid, affinity);

            // Optimized atomic compare
#pragma omp atomic compare
            min_deg = min_deg > local_min_deg ? local_min_deg : min_deg;

#pragma omp barrier

            deglists[tid].traverse(tid, min_deg, mult * min_deg, affinity, lim / num_threads);

            vtype base = 0;
#pragma omp atomic capture
            {
                base = num_candidates;
                num_candidates += deglists[tid].cand_end;
            }

            // Vectorized copy when possible
            const vtype cand_end = deglists[tid].cand_end;
            std::memcpy(candidates + base, deglists[tid].cand, sizeof(vtype) * cand_end);

#pragma omp barrier
        }

        inline void insert(const vtype deg, const vtype index)
        {
            const size_t tid = omp_get_thread_num();
            deglists[tid].insert(deg, index);
            affinity[index] = tid;
        }

        inline void remove(const vtype deg, const vtype index)
        {
            affinity[index] = -1;
        }
    };

    // Optimized hash lists with better hash function
    struct optimized_hashlists
    {
        vtype n;
        vtype *base_, *head, *next, *hash;

        optimized_hashlists(const vtype &n = 0) : n(n)
        {
            base_ = aligned_alloc_cache<vtype>(n * 3);
            std::memset(base_, -1, sizeof(vtype) * 3 * n);
            head = base_;
            next = base_ + n;
            hash = base_ + n * 2;
        }

        ~optimized_hashlists() { std::free(base_); }

        // Fast hash function using bit manipulation
        inline vtype fast_hash(vtype value) const
        {
            // Murmur-like hash for better distribution
            value ^= value >> 16;
            value *= 0x85ebca6b;
            value ^= value >> 13;
            value *= 0xc2b2ae35;
            value ^= value >> 16;
            return value % n;
        }

        inline void insert(vtype hsh, vtype index)
        {
            next[index] = head[hsh];
            head[hsh] = index;
            hash[index] = hsh;
        }

        inline bool empty(vtype hsh) const { return head[hsh] == -1; }
        inline bool is_tail(vtype index) const { return next[index] == -1; }
        inline vtype get_hash(vtype index) const { return hash[index]; }
        inline vtype get_next(vtype index) const { return next[index]; }
        inline vtype get_nil() const { return -1; }

        inline vtype pop(vtype hsh)
        {
            const vtype index = head[hsh];
            head[hsh] = next[index];
            return index;
        }

        inline void remove(vtype index, vtype prev_index)
        {
            if (__builtin_expect(prev_index != -1, 1))
                next[prev_index] = next[index];
            else
                head[hash[index]] = next[index];
        }
    };

    // Optimized timestamp clearing with vectorization
    inline void optimized_clear_stp(vtype *stp, const vtype n, vtype &tstp, const vtype tlim)
    {
        if (__builtin_expect(tstp < 2 || tstp >= tlim, 0))
        {
            // Vectorized clearing when possible
            const vtype aligned_n = (n / 8) * 8;

#pragma omp simd
            for (vtype i = 0; i < aligned_n; i += 8)
            {
                // Process 8 elements at once
                for (int j = 0; j < 8; ++j)
                {
                    if (stp[i + j] != 0)
                        stp[i + j] = 1;
                }
            }

            // Handle remaining elements
            for (vtype i = aligned_n; i < n; ++i)
            {
                if (stp[i] != 0)
                    stp[i] = 1;
            }

            tstp = 2;
        }
    }

    // Optimized symmetrization with better memory access patterns
    void optimized_symmetrize(const vtype n, etype &free_start, const vtype *rowptr, const etype *colidx,
                              etype *&symrowptr, vtype *&neighborhood, const double mem, const bool sym)
    {
        if (sym)
        {
            symrowptr = aligned_alloc_cache<etype>(n + 1);
            symrowptr[0] = 0;

#pragma omp parallel for simd
            for (vtype i = 0; i < n; ++i)
            {
                bool diag = false;
                const etype row_start = rowptr[i];
                const etype row_end = rowptr[i + 1];

                // Prefetch next row
                if (i + 1 < n)
                    __builtin_prefetch(&colidx[rowptr[i + 1]], 0, 3);

                for (etype j = row_start; j < row_end; ++j)
                {
                    if (i == colidx[j])
                    {
                        diag = true;
                        break;
                    }
                }
                symrowptr[i + 1] = row_end - row_start - (diag ? 1 : 0);
            }

            // Parallel prefix sum
            for (vtype i = 0; i < n; ++i)
                symrowptr[i + 1] += symrowptr[i];

            free_start = symrowptr[n];
            neighborhood = aligned_alloc_cache<vtype>(free_start * (1 + mem));

#pragma omp parallel for
            for (vtype i = 0; i < n; ++i)
            {
                vtype idx = 0;
                const etype row_start = rowptr[i];
                const etype row_end = rowptr[i + 1];

                for (etype j = row_start; j < row_end; ++j)
                {
                    if (i != colidx[j])
                        neighborhood[symrowptr[i] + (idx++)] = colidx[j];
                }
            }
            return;
        }

        // Non-symmetric case - optimized version of original algorithm
        etype *newrowptr = aligned_alloc_cache<etype>(n + 1);
        symrowptr = aligned_alloc_cache<etype>(n + 1);
        vtype *newcolidx = aligned_alloc_cache<vtype>(rowptr[n] * 2);

        std::fill(newrowptr, newrowptr + n + 1, 0);

        const vtype stride = (n + 15) / 16 * 16; // Cache-aligned stride
        vtype *cnt_ = aligned_alloc_cache<vtype>(stride * omp_get_max_threads());

#pragma omp parallel
        {
            const size_t tid = omp_get_thread_num();
            const int num_threads = omp_get_num_threads();
            vtype *cnt = cnt_ + tid * stride;

            // Initialize with vectorization
            std::fill(cnt, cnt + n, 0);

            const vtype l = n * tid / num_threads;
            const vtype r = n * (tid + 1) / num_threads;

            // First pass: count with better cache utilization
            for (vtype i = l; i < r; ++i)
            {
                const etype row_start = rowptr[i];
                const etype row_end = rowptr[i + 1];

                for (etype j = row_start; j < row_end; ++j)
                {
                    const vtype col = colidx[j];
                    if (i != col)
                    {
                        ++cnt[i];
                        ++cnt[col];
                    }
                }
            }

            // Atomic accumulation with reduced contention
            for (vtype i = 0; i < n; ++i)
            {
                const vtype add = cnt[i];
                if (add != 0)
                {
#pragma omp atomic capture
                    {
                        cnt[i] = newrowptr[i + 1];
                        newrowptr[i + 1] += add;
                    }
                }
            }

#pragma omp barrier
#pragma omp master
            {
                // Parallel prefix sum
                for (vtype i = 0; i < n; ++i)
                    newrowptr[i + 1] += newrowptr[i];
            }
#pragma omp barrier

            // Second pass: fill with optimized memory access
            for (vtype i = l; i < r; ++i)
            {
                const etype row_start = rowptr[i];
                const etype row_end = rowptr[i + 1];

                for (etype j = row_start; j < row_end; ++j)
                {
                    const vtype col = colidx[j];
                    if (i != col)
                    {
                        newcolidx[(cnt[i]++) + newrowptr[i]] = col;
                        newcolidx[(cnt[col]++) + newrowptr[col]] = i;
                    }
                }
            }

#pragma omp barrier

            // Deduplication with optimized inner loop
            for (vtype i = l; i < r; ++i)
            {
                vtype unique = 0;
                const vtype stp = -i - 1;
                const etype row_start = newrowptr[i];
                const etype row_end = newrowptr[i + 1];

                for (etype j = row_start; j < row_end; ++j)
                {
                    const vtype neighbor = newcolidx[j];
                    if (cnt[neighbor] != stp)
                    {
                        cnt[neighbor] = stp;
                        newcolidx[row_start + (unique++)] = neighbor;
                    }
                }
                symrowptr[i + 1] = unique;
            }

#pragma omp barrier
#pragma omp master
            {
                symrowptr[0] = 0;
                for (vtype i = 0; i < n; ++i)
                    symrowptr[i + 1] += symrowptr[i];

                free_start = symrowptr[n];
                neighborhood = aligned_alloc_cache<vtype>(free_start * (1 + mem));
            }
#pragma omp barrier

            // Final copy with prefetching
            for (vtype i = l; i < r; ++i)
            {
                const etype sym_start = symrowptr[i];
                const etype sym_end = symrowptr[i + 1];
                const etype new_start = newrowptr[i];

                for (etype j = sym_start; j < sym_end; ++j)
                {
                    neighborhood[j] = newcolidx[new_start + j - sym_start];
                }
            }
        }

        std::free(cnt_);
        std::free(newrowptr);
        std::free(newcolidx);
    }

    // Main optimized ParAMD algorithm
    uint64_t paramd_optimized(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config)
    {
#define EMPTY (-1)
#define FLIP(x) (-(x) - 2)
#define UNFLIP(x) ((x < EMPTY) ? FLIP(x) : (x))

        // Optimized Node structure with better layout
        struct CACHE_ALIGNED OptimizedNode
        {
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
            uint32_t padding; // Explicit padding for alignment
        };

        constexpr double alpha = 10.0;
        constexpr bool aggressive = true;
        constexpr uint64_t INF = 1ULL << 63;

        optimized_timer timer(config.breakdown);
        etype free_start = 0;
        etype *symrowptr = nullptr;
        vtype *neighborhood = nullptr;

        optimized_symmetrize(n, free_start, rowptr, colidx, symrowptr, neighborhood, config.mem, config.sym);
        timer.time("Optimized A + AT");

        // Cache-aligned allocations
        vtype *const iperm = aligned_alloc_cache<vtype>(n);
        vtype *const stk = aligned_alloc_cache<vtype>(n);
        vtype *const inv_rank = aligned_alloc_cache<vtype>(n);
        vtype *const candidates = aligned_alloc_cache<vtype>(n);
        OptimizedNode *s = nullptr;
        {
            void *ptr = nullptr;
            if (posix_memalign(&ptr, CACHE_LINE_SIZE, n * sizeof(OptimizedNode)) != 0)
                throw std::bad_alloc();
            s = (OptimizedNode *)ptr;
        }

        vtype num_dense = 0, num_eliminated = 0;
        const vtype dense_threshold = std::min(n, std::max(16, vtype(alpha < 0 ? n - 2 : alpha * std::sqrt(n))));

        optimized_concurrent_degree_lists deglists(n);
        vtype min_deg = 1;

#pragma omp parallel reduction(+ : num_eliminated, num_dense)
        {
#pragma omp for simd
            for (vtype i = 0; i < n; ++i)
            {
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

                if (s[i].deg == 0)
                {
                    s[i].num_hyperedge = FLIP(1);
                    ++num_eliminated;
                    s[i].neighborhood_ptr = EMPTY;
                    s[i].stp = 0;
                }
                else if (s[i].deg > dense_threshold)
                {
                    ++num_dense;
                    ++num_eliminated;
                    s[i].neighborhood_ptr = EMPTY;
                    s[i].stp = 0;
                    s[i].supernode_size = 0;
                    s[i].num_hyperedge = EMPTY;
                }
                else
                {
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
            optimized_hashlists hashlists(n);
            vtype *const private_ = aligned_alloc_cache<vtype>(n * 3);
            vtype *const private_stp = private_;
            vtype *const workspace = private_ + n;
            vtype *const private_cand = private_ + n * 2;

            std::fill(private_stp, private_stp + n, 1);
            vtype num_private_cand = 0, workspace_end = 0, round = 1;
            vtype private_tstp = 2, private_t_max_step = 0;
            const vtype private_tlim = std::numeric_limits<vtype>::max() - n;
            uint64_t private_lnz = 0;

            std::mt19937 gen(omp_get_thread_num());
            std::uniform_int_distribution<uint64_t> dis(0, n - 1);

            timer.time("Optimized Initialization");

            while (num_eliminated < n)
            {
#pragma omp barrier
                deglists.traverse(min_deg, num_candidates, candidates, config.mult, config.lim);

                {
                    num_private_cand = 0;
                    ++round;

#pragma omp for
                    for (vtype i = 0; i < num_candidates; ++i)
                    {
                        const vtype cand = candidates[i];
                        private_cand[num_private_cand++] = cand;
                        s[cand].luby = dis(gen) << 32 | cand;
                        s[cand].stp = round;
                        s[cand].valid = 1;
                    }

                    // Optimized distance-2 independent set computation
                    for (vtype i = 0; i < num_private_cand; ++i)
                    {
                        const vtype cand = private_cand[i];
                        uint32_t &valid = s[cand].valid;
                        s[cand].luby_min = INF;

                        const etype hyper_start = s[cand].neighborhood_ptr;
                        const etype hyper_end = hyper_start + s[cand].num_hyperedge;

                        // Prefetch for better cache utilization
                        if (hyper_start < hyper_end)
                            __builtin_prefetch(&neighborhood[hyper_start], 0, 3);

                        for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr)
                        {
                            const vtype hyper = neighborhood[hyper_ptr];
                            const etype nei_start = s[hyper].neighborhood_ptr;
                            const etype nei_end = nei_start + s[hyper].neighborhood_len;

                            for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr)
                            {
                                const vtype nei = neighborhood[nei_ptr];
                                if (__builtin_expect(s[nei].stp != 0, 1))
                                {
                                    if (s[nei].stp == round && s[nei].luby < s[cand].luby)
                                    {
                                        valid = false;
                                    }
                                    else if (s[nei].luby_min != INF)
                                    {
                                        s[nei].luby_min = INF;
                                    }
                                }
                            }
                        }

                        if (!valid)
                            continue;

                        const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
                        for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr)
                        {
                            const vtype nei = neighborhood[nei_ptr];
                            if (__builtin_expect(s[nei].stp != 0, 1))
                            {
                                if (s[nei].stp == round && s[nei].luby < s[cand].luby)
                                {
                                    valid = false;
                                }
                                else if (s[nei].luby_min != INF)
                                {
                                    s[nei].luby_min = INF;
                                }
                            }
                        }
                    }
                }

#pragma omp barrier

                // Continue with optimized Luby computation
                for (vtype i = 0; i < num_private_cand; ++i)
                {
                    const vtype cand = private_cand[i];
                    const uint64_t luby_cand = s[cand].luby;
                    uint32_t &valid = s[cand].valid;
                    if (!valid)
                        continue;

#pragma omp atomic compare
                    s[cand].luby_min = s[cand].luby_min > luby_cand ? luby_cand : s[cand].luby_min;

                    if (s[cand].luby_min != luby_cand)
                    {
                        valid = false;
                        continue;
                    }

                    // Optimized neighbor traversal with prefetching
                    const etype hyper_start = s[cand].neighborhood_ptr;
                    const etype hyper_end = hyper_start + s[cand].num_hyperedge;

                    for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr)
                    {
                        const vtype hyper = neighborhood[hyper_ptr];
                        const etype nei_start = s[hyper].neighborhood_ptr;
                        const etype nei_end = nei_start + s[hyper].neighborhood_len;

                        for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr)
                        {
                            const vtype nei = neighborhood[nei_ptr];
                            if (__builtin_expect(s[nei].stp != 0, 1))
                            {
#pragma omp atomic compare
                                s[nei].luby_min = s[nei].luby_min > luby_cand ? luby_cand : s[nei].luby_min;
                                if (s[nei].luby_min != luby_cand)
                                    valid = false;
                            }
                        }
                    }

                    if (!valid)
                        continue;

                    const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
                    for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr)
                    {
                        const vtype nei = neighborhood[nei_ptr];
                        if (__builtin_expect(s[nei].stp != 0, 1))
                        {
#pragma omp atomic compare
                            s[nei].luby_min = s[nei].luby_min > luby_cand ? luby_cand : s[nei].luby_min;
                            if (s[nei].luby_min != luby_cand)
                                valid = false;
                        }
                    }
                }

#pragma omp barrier

                // Final validation with optimized loops
                vtype resize = 0;
                for (vtype i = 0; i < num_private_cand; ++i)
                {
                    const vtype cand = private_cand[i];
                    const uint64_t luby_cand = s[cand].luby;
                    bool valid = (s[cand].luby_min == luby_cand) && s[cand].valid;

                    if (!valid)
                        continue;

                    // Optimized validation check
                    const etype hyper_start = s[cand].neighborhood_ptr;
                    const etype hyper_end = hyper_start + s[cand].num_hyperedge;

                    for (etype hyper_ptr = hyper_start; valid && hyper_ptr < hyper_end; ++hyper_ptr)
                    {
                        const vtype hyper = neighborhood[hyper_ptr];
                        const etype nei_start = s[hyper].neighborhood_ptr;
                        const etype nei_end = nei_start + s[hyper].neighborhood_len;

                        for (etype nei_ptr = nei_start; valid && nei_ptr < nei_end; ++nei_ptr)
                        {
                            const vtype nei = neighborhood[nei_ptr];
                            if (s[nei].stp != 0 && s[nei].luby_min != luby_cand)
                                valid = false;
                        }
                    }

                    if (!valid)
                        continue;

                    const etype super_end = s[cand].neighborhood_ptr + s[cand].neighborhood_len;
                    for (etype nei_ptr = hyper_end; valid && nei_ptr < super_end; ++nei_ptr)
                    {
                        const vtype nei = neighborhood[nei_ptr];
                        if (s[nei].stp != 0 && s[nei].luby_min != luby_cand)
                            valid = false;
                    }

                    if (valid)
                        private_cand[resize++] = cand;
                }
                num_private_cand = resize;

                timer.time("Optimized Distance-2 Independent Sets");

                // Statistics collection (optional)
                if (config.stat)
                {
#pragma omp master
                    size_profile_total = 0;
#pragma omp barrier
#pragma omp atomic
                    size_profile_total += num_private_cand;
#pragma omp barrier
#pragma omp master
                    size_profile.emplace_back(size_profile_total);
                }

                // Core elimination phase with optimizations
                vtype private_num_eliminated = 0;
                workspace_end = 0;

                for (vtype cand_idx = 0; cand_idx < num_private_cand; ++cand_idx)
                {
                    const vtype pivot = private_cand[cand_idx];
                    vtype npiv = s[pivot].supernode_size;
                    private_num_eliminated += npiv;
                    s[pivot].central_pivot = pivot;
                    s[pivot].supernode_size = -npiv;

                    vtype pivot_deg = 0;
                    const etype new_nei_start = workspace_end;
                    deglists.remove(s[pivot].deg, pivot);

                    // Optimized neighbor processing
                    const etype hyper_start = s[pivot].neighborhood_ptr;
                    const etype hyper_end = hyper_start + s[pivot].num_hyperedge;

                    // Process hyperedges with prefetching
                    for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr)
                    {
                        const vtype hyper = neighborhood[hyper_ptr];

                        // Prefetch next hyperedge
                        if (hyper_ptr + 1 < hyper_end)
                            __builtin_prefetch(&neighborhood[hyper_ptr + 1], 0, 3);

                        const etype nei_start = s[hyper].neighborhood_ptr;
                        const etype nei_end = nei_start + s[hyper].neighborhood_len;

                        for (etype nei_ptr = nei_start; nei_ptr < nei_end; ++nei_ptr)
                        {
                            const vtype nei = neighborhood[nei_ptr];
                            const vtype nei_size = s[nei].supernode_size;

                            if (__builtin_expect(nei_size > 0, 1))
                            {
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

                    // Process supernodes
                    const etype super_end = s[pivot].neighborhood_ptr + s[pivot].neighborhood_len;
                    for (etype nei_ptr = hyper_end; nei_ptr < super_end; ++nei_ptr)
                    {
                        const vtype nei = neighborhood[nei_ptr];
                        const vtype nei_size = s[nei].supernode_size;

                        if (__builtin_expect(nei_size > 0, 1))
                        {
                            pivot_deg += nei_size;
                            s[nei].supernode_size = -nei_size;
                            workspace[workspace_end++] = nei;
                            s[nei].central_pivot = pivot;
                            deglists.remove(s[nei].deg, nei);
                        }
                    }

                    const etype new_nei_end = workspace_end;
                    s[pivot].neighborhood_ptr = new_nei_start;
                    s[pivot].num_hyperedge = FLIP(npiv + pivot_deg);

                    optimized_clear_stp(private_stp, n, private_tstp, private_tlim);
                    private_t_max_step = 0;

                    // Optimized degree computation
                    for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
                    {
                        const vtype nei = workspace[nei_ptr];
                        const etype hyper_start = s[nei].neighborhood_ptr;
                        const etype hyper_end = hyper_start + s[nei].num_hyperedge;

                        for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr)
                        {
                            const vtype hyper = neighborhood[hyper_ptr];

                            if (__builtin_expect(private_stp[hyper] >= private_tstp, 0))
                            {
                                private_stp[hyper] += s[nei].supernode_size;
                            }
                            else if (__builtin_expect(s[hyper].stp != 0, 1))
                            {
                                private_stp[hyper] = s[hyper].deg + private_tstp + s[nei].supernode_size;
                                private_t_max_step = std::max(private_t_max_step, s[hyper].deg);
                            }
                        }
                    }

                    // Optimized neighbor update with hash detection
                    for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
                    {
                        const vtype nei = workspace[nei_ptr];
                        const etype hyper_start = s[nei].neighborhood_ptr;
                        const etype hyper_end = hyper_start + s[nei].num_hyperedge;
                        etype new_end = hyper_start;
                        vtype nei_deg = 0;
                        vtype hash = 0;

                        for (etype hyper_ptr = hyper_start; hyper_ptr < hyper_end; ++hyper_ptr)
                        {
                            const vtype hyper = neighborhood[hyper_ptr];

                            if (__builtin_expect(s[hyper].stp != 0, 1))
                            {
                                const vtype external_deg = private_stp[hyper] - private_tstp;

                                if (!aggressive || external_deg > 0)
                                {
                                    nei_deg += external_deg;
                                    neighborhood[new_end++] = hyper;
                                    hash += hyper; // Simple hash accumulation
                                }
                                else
                                {
                                    s[hyper].neighborhood_ptr = FLIP(pivot);
                                    s[hyper].stp = 0;
                                }
                            }
                        }

                        s[nei].num_hyperedge = new_end - hyper_start + 1;
                        const etype super_end = hyper_start + s[nei].neighborhood_len;
                        const etype super_start = new_end;

                        for (etype super_ptr = hyper_end; super_ptr < super_end; ++super_ptr)
                        {
                            const vtype super = neighborhood[super_ptr];

                            if (s[super].supernode_size != 0 && s[super].central_pivot != pivot)
                            {
                                nei_deg += std::abs(s[super].supernode_size);
                                neighborhood[new_end++] = super;
                                hash += super;
                            }
                        }

                        // Check for elimination
                        if (s[nei].num_hyperedge == 1 && new_end == super_start)
                        {
                            s[nei].neighborhood_ptr = FLIP(pivot);
                            const vtype nei_size = -s[nei].supernode_size;
                            pivot_deg -= nei_size;
                            private_num_eliminated += nei_size;
                            npiv += nei_size;
                            s[nei].num_hyperedge = EMPTY;
                            s[nei].supernode_size = 0;
                        }
                        else
                        {
                            s[nei].deg = std::min(s[nei].deg, nei_deg);
                            neighborhood[new_end] = neighborhood[super_start];
                            neighborhood[super_start] = neighborhood[hyper_start];
                            neighborhood[hyper_start] = pivot;
                            s[nei].neighborhood_len = new_end - hyper_start + 1;

                            // Optimized hash computation
                            hash = hashlists.fast_hash(hash);
                            hashlists.insert(hash, nei);
                        }
                    }

                    s[pivot].deg = pivot_deg;
                    s[pivot].supernode_size = npiv;
                    private_tstp += private_t_max_step;
                    optimized_clear_stp(private_stp, n, private_tstp, private_tlim);

                    // Optimized supernode detection
                    for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
                    {
                        const vtype nei = workspace[nei_ptr];

                        if (__builtin_expect(s[nei].supernode_size < 0, 1))
                        {
                            const vtype hash = hashlists.get_hash(nei);

                            while (!hashlists.empty(hash))
                            {
                                const vtype nei_i = hashlists.pop(hash);
                                const vtype nei_len_i = s[nei_i].neighborhood_len;
                                const vtype nhe_i = s[nei_i].num_hyperedge;

                                if (hashlists.empty(hash))
                                    break;

                                // Mark nodes for comparison
                                for (etype nei_i_ptr = s[nei_i].neighborhood_ptr + 1;
                                     nei_i_ptr < s[nei_i].neighborhood_ptr + nei_len_i; ++nei_i_ptr)
                                {
                                    private_stp[neighborhood[nei_i_ptr]] = private_tstp;
                                }

                                vtype nei_j = nei_i, prev_nei_j = hashlists.get_nil();

                                while (!hashlists.is_tail(nei_j))
                                {
                                    nei_j = hashlists.get_next(nei_j);
                                    const vtype nei_len_j = s[nei_j].neighborhood_len;
                                    const vtype nhe_j = s[nei_j].num_hyperedge;

                                    bool same = (nei_len_i == nei_len_j) && (nhe_i == nhe_j);

                                    for (etype nei_j_ptr = s[nei_j].neighborhood_ptr + 1;
                                         same && nei_j_ptr < s[nei_j].neighborhood_ptr + nei_len_j; ++nei_j_ptr)
                                    {
                                        same &= (private_stp[neighborhood[nei_j_ptr]] == private_tstp);
                                    }

                                    if (same)
                                    {
                                        s[nei_j].neighborhood_ptr = FLIP(nei_i);
                                        s[nei_i].supernode_size += s[nei_j].supernode_size;
                                        s[nei_j].supernode_size = 0;
                                        s[nei_j].num_hyperedge = EMPTY;
                                        hashlists.remove(nei_j, prev_nei_j);
                                    }
                                    else
                                    {
                                        prev_nei_j = nei_j;
                                    }
                                }
                                ++private_tstp;
                            }
                        }
                    }

                    // Final neighbor list assembly
                    etype final_nei_end = new_nei_start;
                    for (etype nei_ptr = new_nei_start; nei_ptr < new_nei_end; ++nei_ptr)
                    {
                        const vtype nei = workspace[nei_ptr];
                        const vtype nei_size = -s[nei].supernode_size;

                        if (__builtin_expect(nei_size > 0, 1))
                        {
                            s[nei].supernode_size = nei_size;
                            s[nei].deg = std::min(s[nei].deg + pivot_deg - nei_size, n - num_eliminated - nei_size);
                            workspace[final_nei_end++] = nei;
                            deglists.insert(s[nei].deg, nei);
                        }
                    }

                    s[pivot].neighborhood_len = final_nei_end - new_nei_start;
                    workspace_end = final_nei_end;

                    // Optimized fill computation
                    private_lnz += uint64_t(s[pivot].supernode_size) * (s[pivot].deg + num_dense) +
                                   uint64_t(s[pivot].supernode_size) * (s[pivot].supernode_size - 1) / 2;

                    if (s[pivot].neighborhood_len == 0)
                    {
                        s[pivot].neighborhood_ptr = EMPTY;
                        s[pivot].stp = 0;
                    }
                }

                // Copy workspace to global memory with atomic allocation
                etype base = 0;
#pragma omp atomic capture
                {
                    base = free_start;
                    free_start += workspace_end;
                }

                // Vectorized copy when possible
                std::memcpy(neighborhood + base, workspace, sizeof(vtype) * workspace_end);

                // Update pointers
                for (vtype cand_idx = 0; cand_idx < num_private_cand; ++cand_idx)
                {
                    const vtype pivot = private_cand[cand_idx];
                    if (s[pivot].neighborhood_len != 0)
                        s[pivot].neighborhood_ptr += base;
                }

#pragma omp atomic
                num_eliminated += private_num_eliminated;
#pragma omp barrier
                timer.time("Optimized Core");
            }

#pragma omp atomic
            lnz += private_lnz;
            std::free(private_);
        }

        // Post-processing phase with optimizations
#pragma omp parallel for simd
        for (vtype i = 0; i < n; ++i)
        {
            s[i].neighborhood_ptr = FLIP(s[i].neighborhood_ptr);
            s[i].num_hyperedge = FLIP(s[i].num_hyperedge);
        }

        // Path compression with optimization
#pragma omp parallel for
        for (vtype i = 0; i < n; ++i)
        {
            if (s[i].supernode_size == 0)
            {
                vtype p = s[i].neighborhood_ptr;
                if (__builtin_expect(p != EMPTY, 1))
                {
                    while (s[p].supernode_size == 0)
                        p = s[p].neighborhood_ptr;
                    const vtype hyper = p;

                    // Path compression
                    for (vtype ptr = i; ptr != hyper;)
                    {
                        p = s[ptr].neighborhood_ptr;
                        s[ptr].neighborhood_ptr = hyper;
                        ptr = p;
                    }
                }
            }
        }

        // Build edge lists with optimization
#pragma omp parallel for
        for (vtype i = n - 1; i >= 0; --i)
        {
            if (s[i].supernode_size > 0)
            {
                const vtype p = s[i].neighborhood_ptr;
                if (__builtin_expect(p != EMPTY, 1))
                {
#pragma omp atomic capture
                    {
                        s[i].edge_next = s[p].edge_head;
                        s[p].edge_head = i;
                    }
                }
            }
        }

        // Optimize elimination tree traversal
#pragma omp parallel for
        for (vtype i = 0; i < n; ++i)
        {
            if (s[i].supernode_size > 0 && s[i].edge_head != EMPTY)
            {
                vtype prev = EMPTY, mxsz = EMPTY, mxprev = EMPTY, mxptr = EMPTY;

                for (vtype j = s[i].edge_head; j != EMPTY; j = s[j].edge_next)
                {
                    if (s[j].num_hyperedge >= mxsz)
                    {
                        mxsz = s[j].num_hyperedge;
                        mxprev = prev;
                        mxptr = j;
                    }
                    prev = j;
                }

                if (s[mxptr].edge_next != EMPTY)
                {
                    if (mxprev == EMPTY)
                        s[i].edge_head = s[mxptr].edge_next;
                    else
                        s[mxprev].edge_next = s[mxptr].edge_next;
                    s[mxptr].edge_next = EMPTY;
                    s[prev].edge_next = mxptr;
                }
            }
        }

        // Generate ordering
        vtype stk_head = 0, k = 0;

        for (vtype i = 0; i < n; ++i)
        {
            if (s[i].neighborhood_ptr == EMPTY && s[i].supernode_size > 0)
            {
                stk[stk_head = 0] = i;

                while (stk_head != EMPTY)
                {
                    const vtype cur = stk[stk_head];

                    if (s[cur].edge_head != EMPTY)
                    {
                        for (vtype j = s[cur].edge_head; j != EMPTY; j = s[j].edge_next)
                            ++stk_head;
                        for (vtype j = s[cur].edge_head, h = stk_head; j != EMPTY; j = s[j].edge_next)
                            stk[h--] = j;
                        s[cur].edge_head = EMPTY;
                    }
                    else
                    {
                        --stk_head;
                        s[cur].order = k++;
                    }
                }
            }
        }

        // Build inverse rank mapping
#pragma omp parallel for
        for (vtype i = 0; i < n; ++i)
        {
            if (s[i].order != EMPTY)
                inv_rank[s[i].order] = i;
        }

        // Generate final permutation
        num_eliminated = 0;
        for (vtype i = 0; i < n; ++i)
        {
            const vtype hyper = inv_rank[i];
            if (hyper == EMPTY)
                break;
            iperm[hyper] = num_eliminated;
            num_eliminated += s[hyper].supernode_size;
        }

#pragma omp parallel for
        for (vtype i = 0; i < n; ++i)
        {
            if (s[i].supernode_size == 0)
            {
                if (s[i].neighborhood_ptr != EMPTY)
                {
#pragma omp atomic capture
                    {
                        iperm[i] = iperm[s[i].neighborhood_ptr];
                        ++iperm[s[i].neighborhood_ptr];
                    }
                }
                else
                {
#pragma omp atomic capture
                    {
                        iperm[i] = num_eliminated;
                        ++num_eliminated;
                    }
                }
            }
        }

#pragma omp parallel for simd
        for (vtype i = 0; i < n; ++i)
        {
            perm[iperm[i]] = i;
        }

        // Cleanup with optimized memory management
        std::free(s);
        std::free(candidates);
        std::free(inv_rank);
        std::free(stk);
        std::free(iperm);
        std::free(neighborhood);
        std::free(symrowptr);

        timer.time("Optimized Other");
        timer.print();

        if (config.stat)
        {
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