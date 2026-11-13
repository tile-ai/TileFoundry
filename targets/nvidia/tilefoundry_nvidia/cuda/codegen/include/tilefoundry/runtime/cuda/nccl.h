#pragma once

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <tvm/ffi/container/tensor.h>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace tilefoundry::nccl {

inline void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
inline void check(ncclResult_t status) {
    if (status != ncclSuccess)
        throw std::runtime_error(ncclGetErrorString(status));
}
inline cudaStream_t stream(int64_t value) {
    return reinterpret_cast<cudaStream_t>(value);
}
inline cudaEvent_t event(int64_t value) {
    return reinterpret_cast<cudaEvent_t>(value);
}
inline ncclComm_t communicator(int64_t value) {
    return reinterpret_cast<ncclComm_t>(value);
}

inline void tensor(const tvm::ffi::Tensor &value,
                   std::initializer_list<int64_t> shape, int code, int bits,
                   DLDeviceType device) {
    auto dtype = value.dtype();
    if (value.device().device_type != device || dtype.code != code ||
        dtype.bits != bits || dtype.lanes != 1 ||
        value.ndim() != static_cast<int>(shape.size()) ||
        !std::equal(shape.begin(), shape.end(), value.shape().begin()))
        throw std::runtime_error(
            "communication tensor shape, dtype or device mismatch");
    int64_t stride = 1;
    for (int i = value.ndim() - 1; i >= 0; --i) {
        if (value.size(i) > 1 && value.stride(i) != stride)
            throw std::runtime_error(
                "communication tensors must be contiguous");
        stride *= value.size(i);
    }
    if (device == kDLCUDA) {
        int current;
        check(cudaGetDevice(&current));
        if (current != value.device().device_id)
            throw std::runtime_error(
                "communication tensor is on another CUDA device");
    }
}

inline void disjoint(const tvm::ffi::Tensor &first,
                     const tvm::ffi::Tensor &second) {
    if (first.device().device_type != second.device().device_type ||
        first.device().device_id != second.device().device_id)
        return;
    auto a = reinterpret_cast<uintptr_t>(first.data_ptr());
    auto b = reinterpret_cast<uintptr_t>(second.data_ptr());
    auto an = first.numel() * first.dtype().bits / 8;
    auto bn = second.numel() * second.dtype().bits / 8;
    if (an && bn && a < b + bn && b < a + an)
        throw std::runtime_error("undeclared tensor storage alias");
}

inline std::vector<int64_t> counts(const tvm::ffi::Tensor &value, int n,
                                   cudaStream_t s) {
    std::vector<int64_t> result(n * n);
    auto copy = [&](void *destination, size_t bytes) {
        if (value.device().device_type == kDLCPU)
            std::copy_n(static_cast<const char *>(value.data_ptr()), bytes,
                        static_cast<char *>(destination));
        else {
            check(cudaMemcpyAsync(destination, value.data_ptr(), bytes,
                                  cudaMemcpyDeviceToHost, s));
            check(cudaStreamSynchronize(s));
        }
    };
    if (value.dtype().bits == 64)
        copy(result.data(), result.size() * sizeof(int64_t));
    else {
        std::vector<int32_t> narrow(n * n);
        copy(narrow.data(), narrow.size() * sizeof(int32_t));
        std::copy(narrow.begin(), narrow.end(), result.begin());
    }
    return result;
}

struct Task {
    int kind;
    const void *source;
    void *destination;
    size_t send_capacity, receive_capacity, outer, bytes;
    ncclDataType_t dtype;
    ncclRedOp_t reduction;
    int root, peers, rank;
    ncclComm_t comm;
    cudaStream_t stream;
    std::vector<int64_t> counts;
    void *scratch = nullptr;

    void validate() const {
        int size, local, device, current;
        check(ncclCommCount(comm, &size));
        check(ncclCommUserRank(comm, &local));
        check(ncclCommCuDevice(comm, &device));
        check(cudaGetDevice(&current));
        if (size != peers || local != rank || device != current)
            throw std::runtime_error("NCCL communicator does not match the "
                                     "declared rank domain or device");
        if (source == destination && kind >= 3 && send_capacity)
            throw std::runtime_error("this communication requires distinct "
                                     "send and receive buffers");
        if (kind == 7) {
            if (counts.size() != static_cast<size_t>(peers * peers))
                throw std::runtime_error("runtime count matrix has wrong size");
            for (int i = 0; i < peers; ++i) {
                size_t row = 0, column = 0;
                for (int j = 0; j < peers; ++j) {
                    auto send = counts[i * peers + j],
                         recv = counts[j * peers + i];
                    if (send < 0 || recv < 0 ||
                        static_cast<uint64_t>(send) > send_capacity - row ||
                        static_cast<uint64_t>(recv) > receive_capacity - column)
                        throw std::runtime_error(
                            "runtime count matrix is negative or exceeds "
                            "capacity");
                    row += send;
                    column += recv;
                }
            }
        }
    }

    void transpose(void *packed, bool pack) {
        size_t local = (pack ? send_capacity : receive_capacity) / peers;
        size_t width = outer ? local / outer * bytes : 0;
        if (!width || !outer)
            return;
        for (int peer = 0; peer < peers; ++peer) {
            auto *contiguous =
                static_cast<char *>(packed) + peer * local * bytes;
            if (pack)
                check(cudaMemcpy2DAsync(contiguous, width,
                                        static_cast<const char *>(source) +
                                            peer * width,
                                        width * peers, width, outer,
                                        cudaMemcpyDeviceToDevice, stream));
            else
                check(cudaMemcpy2DAsync(
                    static_cast<char *>(destination) + peer * width,
                    width * peers, contiguous, width, width, outer,
                    cudaMemcpyDeviceToDevice, stream));
        }
    }

    void prepare() {
        if (outer > 1 && kind >= 3 && kind <= 6) {
            size_t size =
                (kind == 3 || kind == 5 ? receive_capacity : send_capacity) *
                bytes;
            if (size)
                check(cudaMallocAsync(&scratch, size, stream));
            if (scratch && (kind == 4 || kind == 6))
                transpose(scratch, true);
        }
        if (kind == 7 && receive_capacity)
            check(cudaMemsetAsync(destination, 0, receive_capacity * bytes,
                                  stream));
    }

    void issue() {
        auto *send = static_cast<const char *>(
            scratch && (kind == 4 || kind == 6) ? scratch : source);
        auto *recv = static_cast<char *>(
            scratch && (kind == 3 || kind == 5) ? scratch : destination);
        switch (kind) {
        case 0:
            check(ncclBroadcast(send, recv, send_capacity, dtype, root, comm,
                                stream));
            break;
        case 1:
            check(ncclReduce(send, recv, send_capacity, dtype, reduction, root,
                             comm, stream));
            break;
        case 2:
            check(ncclAllReduce(send, recv, send_capacity, dtype, reduction,
                                comm, stream));
            break;
        case 3:
            check(ncclSend(send, send_capacity, dtype, root, comm, stream));
            if (rank == root)
                for (int peer = 0; peer < peers; ++peer)
                    check(ncclRecv(recv + peer * send_capacity * bytes,
                                   send_capacity, dtype, peer, comm, stream));
            break;
        case 4:
            if (rank == root)
                for (int peer = 0; peer < peers; ++peer)
                    check(ncclSend(send + peer * receive_capacity * bytes,
                                   receive_capacity, dtype, peer, comm,
                                   stream));
            check(ncclRecv(recv, receive_capacity, dtype, root, comm, stream));
            break;
        case 5:
            check(
                ncclAllGather(send, recv, send_capacity, dtype, comm, stream));
            break;
        case 6:
            check(ncclReduceScatter(send, recv, receive_capacity, dtype,
                                    reduction, comm, stream));
            break;
        case 7: {
            size_t sent = 0, received = 0;
            for (int peer = 0; peer < peers; ++peer) {
                auto outgoing = counts[rank * peers + peer],
                     incoming = counts[peer * peers + rank];
                if (outgoing)
                    check(ncclSend(send + sent * bytes, outgoing, dtype, peer,
                                   comm, stream));
                if (incoming)
                    check(ncclRecv(recv + received * bytes, incoming, dtype,
                                   peer, comm, stream));
                sent += outgoing;
                received += incoming;
            }
            break;
        }
        default:
            throw std::runtime_error("unknown NCCL operation");
        }
    }

    void finish() {
        if (scratch && (kind == 5 || (kind == 3 && rank == root)))
            transpose(scratch, false);
        if (scratch) {
            check(cudaFreeAsync(scratch, stream));
            scratch = nullptr;
        }
    }
};

class Session {
    int depth = 0;
    bool failed = false;
    std::vector<Task> tasks;

    void join() {
        std::vector<cudaStream_t> streams;
        for (const auto &task : tasks)
            if (std::find(streams.begin(), streams.end(), task.stream) ==
                streams.end())
                streams.push_back(task.stream);
        if (streams.size() < 2)
            return;
        std::vector<cudaEvent_t> events(streams.size());
        for (size_t i = 0; i < streams.size(); ++i) {
            check(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
            check(cudaEventRecord(events[i], streams[i]));
        }
        for (auto s : streams)
            for (auto e : events)
                check(cudaStreamWaitEvent(s, e, 0));
        for (auto e : events)
            check(cudaEventDestroy(e));
    }

    void discard() noexcept {
        for (const auto &task : tasks)
            cudaStreamSynchronize(task.stream);
        for (auto &task : tasks)
            if (task.scratch) {
                cudaFreeAsync(task.scratch, task.stream);
                task.scratch = nullptr;
            }
        tasks.clear();
    }

    void flush() {
        try {
            for (const auto &task : tasks)
                task.validate();
            join();
            for (auto &task : tasks)
                task.prepare();
            join();
            check(ncclGroupStart());
            try {
                for (auto &task : tasks)
                    task.issue();
            } catch (...) {
                ncclGroupEnd();
                throw;
            }
            check(ncclGroupEnd());
            for (auto &task : tasks)
                task.finish();
            join();
            tasks.clear();
        } catch (...) {
            failed = true;
            discard();
            throw;
        }
    }

    void require_usable() const {
        if (failed)
            throw std::runtime_error("communication session is poisoned");
    }

  public:
    ~Session() { discard(); }
    void start() {
        require_usable();
        ++depth;
    }
    void end() {
        require_usable();
        if (!depth)
            throw std::runtime_error("unmatched communication group end");
        if (!--depth)
            flush();
    }
    void enqueue(Task task) {
        require_usable();
        tasks.push_back(std::move(task));
        if (!depth)
            flush();
    }
};

}
