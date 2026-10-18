#pragma once
#include "amx.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
namespace local_amx {
class Pool {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable ready, done;
    std::atomic<int> next{0};
    int count = 0, generation = 0, remaining = 0;
    bool stop = false;
    std::function<void(int)> task;

  public:
    explicit Pool(int n) {
        for (int i = 0; i < n; i++)
            threads.emplace_back([this] {
                int seen = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready.wait(lock,
                               [&] { return stop || generation != seen; });
                    if (stop)
                        return;
                    seen = generation;
                    lock.unlock();
                    for (int i = next.fetch_add(1); i < count;
                         i = next.fetch_add(1))
                        task(i);
                    lock.lock();
                    if (--remaining == 0)
                        done.notify_one();
                }
            });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        ready.notify_all();
        for (auto &thread : threads)
            thread.join();
    }
    void run(int n, std::function<void(int)> fn) {
        std::unique_lock<std::mutex> lock(mutex);
        task = std::move(fn);
        count = n;
        next = 0;
        remaining = threads.size();
        generation++;
        ready.notify_all();
        done.wait(lock, [&] { return remaining == 0; });
    }
};
inline Pool &pool() {
    static Pool value(
        std::getenv("TF_AMX_THREADS")
            ? std::max(1, std::atoi(std::getenv("TF_AMX_THREADS")))
            : 8);
    return value;
}
inline std::vector<_Float16> pack_input(const float *a, int m, int k) {
    std::vector<_Float16> out(((m + 31) / 32) * k * 32, 0);
    for (int m0 = 0; m0 < m; m0 += 32)
        for (int p = 0; p < k; p++)
            for (int j = 0; j < std::min(32, m - m0); j++)
                out[(m0 / 32 * k + p) * 32 + j] = (_Float16)a[(m0 + j) * k + p];
    return out;
}
}
namespace shared_fp16 {
struct Weight {
    int n, k, b;
    const _Float16 *data;
    Weight(const _Float16 *packed, int nn, int kk, int bb)
        : n(nn), k(kk), b(bb), data(packed) {}
    const _Float16 *at(int n0, int p) const {
        return data + (n0 / b * k + p) * b + n0 % b;
    }
};
inline void prefill(const float *x, int m, const Weight &w, float *out,
                    int taskcols, bool cyclic = false) {
    auto a = local_amx::pack_input(x, m, w.k);
    int panels = (w.n + taskcols - 1) / taskcols;
    local_amx::pool().run(cyclic ? 8 : panels, [&](int task) {
        alignas(128) float z[32];
        AMX_SET();
        for (int panel = task; panel < panels; panel += cyclic ? 8 : panels)
            for (int n0 = panel * taskcols;
                 n0 < std::min(w.n, (panel + 1) * taskcols); n0 += 32)
                for (int m0 = 0; m0 < m; m0 += 32) {
                    for (int p = 0; p < w.k; p++) {
                        AMX_LDX(ldxy{}.bind((void *)w.at(n0, p)));
                        AMX_LDY(
                            ldxy{}.bind(a.data() + (m0 / 32 * w.k + p) * 32));
                        auto op = fma16{}.f32_z();
                        AMX_FMA16(p == 0 ? op.skip_z() : op);
                    }
                    for (int row = 0; row < std::min(32, m - m0); row++) {
                        AMX_STZ(ldstz{}.row_index(row * 2).bind(z));
                        AMX_STZ(ldstz{}.row_index(row * 2 + 1).bind(z + 16));
                        for (int j = 0; j < 32; j += 8) {
                            auto pair = vzipq_f32(vld1q_f32(z + j / 2),
                                                  vld1q_f32(z + 16 + j / 2));
                            vst1q_f32(out + (m0 + row) * w.n + n0 + j,
                                      pair.val[0]);
                            vst1q_f32(out + (m0 + row) * w.n + n0 + j + 4,
                                      pair.val[1]);
                        }
                    }
                }
        AMX_CLR();
    });
}
}
