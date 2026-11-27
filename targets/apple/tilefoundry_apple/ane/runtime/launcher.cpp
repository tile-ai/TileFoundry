#include "tilefoundry/runtime/ane/launcher.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <limits>
#include <memory>
#include <algorithm>
#include <IOSurface/IOSurface.h>
#include <Security/Security.h>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <string>
#include <sys/sysctl.h>
#include <vector>

extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *);

namespace {

const char *const FRAMEWORK =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
    "AppleNeuralEngine";
const char *const CACHE_ROOT = "/Library/Caches/com.apple.aned";
/** @brief The two digests that follow the image digest in a cache key; they are
 * fixed for an image with no separate weight file and no compile options. */
const char *const KEY_SUFFIX =
    "_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119"
    "_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
const unsigned QOS = 21;
const size_t SURFACE_ALIGN = 0x4000;

template <class Result, class... Args>
Result send(id receiver, const char *selector, Args... args) {
    return ((Result(*)(id, SEL, Args...))objc_msgSend)(
        receiver, sel_registerName(selector), args...);
}

id class_named(const char *name) { return (id)objc_getClass(name); }

struct Pool {
    void *token = objc_autoreleasePoolPush();
    ~Pool() { objc_autoreleasePoolPop(token); }
};

template <class Ref> struct Owned {
    Ref ref;
    explicit Owned(Ref ref) : ref(ref) {}
    Owned(const Owned &) = delete;
    Owned &operator=(const Owned &) = delete;
    ~Owned() {
        if (ref)
            CFRelease(ref);
    }
};

CFStringRef text(const std::string &value) {
    return CFStringCreateWithCString(nullptr, value.c_str(),
                                     kCFStringEncodingUTF8);
}

CFNumberRef number(int64_t value) {
    return CFNumberCreate(nullptr, kCFNumberSInt64Type, &value);
}

std::string utf8(CFStringRef value) {
    char buffer[256] = {0};
    CFStringGetCString(value, buffer, sizeof buffer, kCFStringEncodingUTF8);
    return buffer;
}

/** The signing identifier of the running executable, which names its cache. */
std::string signing_identifier() {
    SecCodeRef self = nullptr;
    if (SecCodeCopySelf(kSecCSDefaultFlags, &self) != errSecSuccess)
        return "";
    Owned<SecCodeRef> code(self);
    CFDictionaryRef fields = nullptr;
    if (SecCodeCopySigningInformation((SecStaticCodeRef)self,
                                      kSecCSDefaultFlags,
                                      &fields) != errSecSuccess)
        return "";
    Owned<CFDictionaryRef> info(fields);
    auto name =
        (CFStringRef)CFDictionaryGetValue(fields, kSecCodeInfoIdentifier);
    return name ? utf8(name) : "";
}

std::string os_build() {
    char build[64] = {0};
    size_t size = sizeof build;
    if (sysctlbyname("kern.osversion", build, &size, nullptr, 0) != 0)
        return "";
    return build;
}

size_t element_bytes(const TensorDesc &desc) {
    if (std::strcmp(desc.dtype, "f16") == 0)
        return 2;
    if (std::strcmp(desc.dtype, "f32") == 0)
        return 4;
    return 0;
}

IOSurfaceRef make_surface(size_t bytes) {
    const void *keys[] = {kIOSurfaceWidth,           kIOSurfaceHeight,
                          kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow,
                          kIOSurfaceAllocSize,       kIOSurfacePixelFormat};
    Owned<CFNumberRef> size(number((int64_t)bytes)), one(number(1)),
        zero(number(0));
    const void *values[] = {size.ref, one.ref,  one.ref,
                            size.ref, size.ref, zero.ref};
    Owned<CFDictionaryRef> properties(CFDictionaryCreate(
        nullptr, keys, values, 6, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks));
    return IOSurfaceCreate(properties.ref);
}

bool matches(const TensorDesc &desc, const DLTensor *tensor) {
    return ane_check_tensor(tensor, desc.shape, 8 * element_bytes(desc), false);
}

/** Copy every element between a DLPack tensor and its declared surface. */
bool transfer(const TensorDesc &desc, DLTensor &tensor, IOSurfaceRef surface,
              bool to_surface) {
    const size_t width = element_bytes(desc);
    int64_t dense[4];
    int64_t step = 1;
    for (int axis = 3; axis >= 0; --axis) {
        dense[axis] = tensor.strides ? tensor.strides[axis] : step;
        step *= desc.shape[axis];
    }
    auto *host = (uint8_t *)tensor.data + tensor.byte_offset;
    if (IOSurfaceLock(surface, to_surface ? 0 : kIOSurfaceLockReadOnly,
                      nullptr) != kIOReturnSuccess)
        return false;
    auto *device = (uint8_t *)IOSurfaceGetBaseAddress(surface);
    if (!device) {
        IOSurfaceUnlock(surface, to_surface ? 0 : kIOSurfaceLockReadOnly,
                        nullptr);
        return false;
    }
    if (to_surface)
        std::memset(device, 0, IOSurfaceGetAllocSize(surface));
    for (int64_t n = 0; n < desc.shape[0]; ++n)
        for (int64_t c = 0; c < desc.shape[1]; ++c)
            for (int64_t h = 0; h < desc.shape[2]; ++h)
                for (int64_t w = 0; w < desc.shape[3]; ++w) {
                    const int64_t at[4] = {n, c, h, w};
                    int64_t from = 0, to = 0;
                    for (int axis = 0; axis < 4; ++axis) {
                        from += at[axis] * dense[axis];
                        to += at[axis] * desc.strides[axis];
                    }
                    uint8_t *a = host + from * width, *b = device + to * width;
                    std::memcpy(to_surface ? b : a, to_surface ? a : b, width);
                }
    return IOSurfaceUnlock(surface, to_surface ? 0 : kIOSurfaceLockReadOnly,
                           nullptr) == kIOReturnSuccess;
}

struct Request {
    std::vector<IOSurfaceRef> surfaces;
    CFMutableArrayRef objects =
        CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);
    CFMutableArrayRef indices =
        CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);

    Request() = default;
    Request(const Request &) = delete;
    Request &operator=(const Request &) = delete;

    bool add(const TensorDesc &desc) {
        if (!objects || !indices || !desc.storage_bytes ||
            desc.storage_bytes > static_cast<size_t>(INT64_MAX) - SURFACE_ALIGN)
            return false;
        const size_t bytes = (desc.storage_bytes + SURFACE_ALIGN - 1) /
                             SURFACE_ALIGN * SURFACE_ALIGN;
        Owned<IOSurfaceRef> surface(make_surface(bytes));
        if (!surface.ref ||
            IOSurfaceGetAllocSize(surface.ref) < desc.storage_bytes)
            return false;
        Owned<CFNumberRef> index(number((int64_t)surfaces.size()));
        id object = send<id>(class_named("_ANEIOSurfaceObject"),
                             "objectWithIOSurface:", surface.ref);
        if (!object || !index.ref)
            return false;
        CFArrayAppendValue(objects, object);
        CFArrayAppendValue(indices, index.ref);
        surfaces.push_back(surface.ref);
        CFRetain(surface.ref);
        return true;
    }

    ~Request() {
        for (IOSurfaceRef surface : surfaces)
            CFRelease(surface);
        if (objects)
            CFRelease(objects);
        if (indices)
            CFRelease(indices);
    }
};

std::string cache_key(const char *image_hash) {
    return std::string(image_hash) + KEY_SUFFIX;
}

}

/** @brief One owned image and its optionally loaded model. */
struct AneProgram {
    const AneEntry *entry;
    std::vector<uint8_t> image;
    id client = nullptr;
    id model = nullptr;
    bool loaded = false;
    CFDictionaryRef options = CFDictionaryCreate(
        nullptr, nullptr, nullptr, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);

    explicit AneProgram(const AneEntry *entry)
        : entry(entry), image(entry->image, entry->image + entry->image_len) {}

    int unload() noexcept {
        if (!loaded)
            return ANE_OK;
        loaded = false;
        CFErrorRef error = nullptr;
        const bool ok =
            send<BOOL>(client, "unloadModel:options:qos:error:", model, options,
                       QOS, &error);
        return ok ? ANE_OK : ANE_UNLOAD_FAILED;
    }

    ~AneProgram() {
        unload();
        if (options)
            CFRelease(options);
        if (model)
            CFRelease(model);
        if (client)
            CFRelease(client);
    }
};

namespace {
std::string image_hash(const AneProgram &program) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    size_t offset = 0;
    while (offset < program.image.size()) {
        const auto count = static_cast<CC_LONG>(std::min(
            program.image.size() - offset, static_cast<size_t>(UINT32_MAX)));
        CC_SHA256_Update(&context, program.image.data() + offset, count);
        offset += count;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (size_t i = 0; i < sizeof digest; ++i)
        std::snprintf(hex + 2 * i, 3, "%02X", digest[i]);
    return hex;
}
}

bool ane_check_tensor(const DLTensor *tensor, const int64_t *shape, int bits,
                      bool contiguous) {
    if (!tensor || !shape || !tensor->data || !tensor->shape ||
        tensor->ndim != 4 || tensor->device.device_type != kDLCPU ||
        tensor->device.device_id != 0 || tensor->dtype.code != kDLFloat ||
        tensor->dtype.bits != bits || tensor->dtype.lanes != 1 ||
        (bits != 16 && bits != 32))
        return false;
    uint64_t dense = 1, last = 0;
    constexpr uint64_t limit = INT64_MAX;
    for (int axis = 3; axis >= 0; --axis) {
        if (shape[axis] <= 0 || tensor->shape[axis] != shape[axis])
            return false;
        const int64_t stride = tensor->strides ? tensor->strides[axis] : dense;
        if (stride < 0 || (contiguous && shape[axis] > 1 &&
                           static_cast<uint64_t>(stride) != dense))
            return false;
        const uint64_t extent = shape[axis];
        if (dense > limit / extent ||
            (extent > 1 &&
             static_cast<uint64_t>(stride) > (limit - last) / (extent - 1)))
            return false;
        last += (extent - 1) * stride;
        dense *= extent;
    }
    const uint64_t width = bits / 8;
    if (last >= limit / width ||
        tensor->byte_offset > limit - (last + 1) * width)
        return false;
    return reinterpret_cast<uintptr_t>(tensor->data) <=
           UINTPTR_MAX - tensor->byte_offset - (last + 1) * width;
}

int ane_create(const AneEntry *entry, AneProgram **program) try {
    if (!program || *program || !entry || !entry->image || !entry->image_len ||
        !entry->io || entry->n_inputs < 0 || entry->n_outputs <= 0 ||
        entry->n_inputs > INT32_MAX - entry->n_outputs ||
        entry->n_weights < 0 ||
        (entry->n_weights &&
         (!entry->weights || entry->coefficient_offset >= entry->image_len)))
        return ANE_BAD_ARGUMENT;
    *program = new AneProgram(entry);
    return ANE_OK;
} catch (...) {
    return ANE_BAD_ARGUMENT;
}

int ane_set_weight(AneProgram *program, const char *name,
                   const DLTensor *tensor) try {
    if (!program || !name || program->loaded)
        return ANE_BAD_ARGUMENT;
    for (int i = 0; i < program->entry->n_weights; ++i) {
        const auto &weight = program->entry->weights[i];
        if (std::strcmp(name, weight.name) == 0)
            return weight.write(program->image.data() +
                                    program->entry->coefficient_offset,
                                tensor);
    }
    return ANE_BAD_ARGUMENT;
} catch (...) {
    return ANE_BAD_ARGUMENT;
}

int ane_image(const AneProgram *program, const uint8_t **bytes,
              size_t *length) try {
    if (!program || !bytes || !length)
        return ANE_BAD_ARGUMENT;
    *bytes = program->image.data();
    *length = program->image.size();
    return ANE_OK;
} catch (...) {
    return ANE_BAD_ARGUMENT;
}

int ane_cache_path(const AneProgram *program, char *path, size_t capacity) try {
    if (!program || !path)
        return ANE_BAD_ARGUMENT;
    const std::string identifier = signing_identifier(), build = os_build();
    if (identifier.empty() || build.empty())
        return ANE_UNAVAILABLE;
    const std::string full =
        std::string(CACHE_ROOT) + "/" + build + "/InMemoryModelCache/" +
        identifier + "_unsigned/" + cache_key(image_hash(*program).c_str()) +
        "/model.hwx";
    if (full.size() + 1 > capacity)
        return ANE_BAD_ARGUMENT;
    std::memcpy(path, full.c_str(), full.size() + 1);
    return ANE_OK;
} catch (...) {
    return ANE_BAD_ARGUMENT;
}

int ane_load(AneProgram *program) try {
    if (!program || !program->options || program->loaded)
        return ANE_BAD_ARGUMENT;
    if (!dlopen(FRAMEWORK, RTLD_NOW | RTLD_LOCAL))
        return ANE_UNAVAILABLE;
    Pool pool;
    Owned<CFStringRef> key(text(cache_key(image_hash(*program).c_str())));
    id client = send<id>(class_named("_ANEClient"), "sharedConnection");
    if (!client || !key.ref)
        return ANE_UNAVAILABLE;
    if (!send<BOOL>(client, "compiledModelExistsMatchingHash:", key.ref))
        return ANE_NOT_PROVISIONED;
    Owned<CFURLRef> url(CFURLCreateWithFileSystemPath(
        nullptr, key.ref, kCFURLPOSIXPathStyle, false));
    id model = send<id>(class_named("_ANEModel"), "modelAtURL:key:", url.ref,
                        CFSTR(""));
    const void *option_keys[] = {CFSTR("kANEFInMemoryModelIsCachedKey"),
                                 CFSTR("kANEFIsInMemoryModelTypeKey"),
                                 CFSTR("kANEFModelType")};
    const void *option_values[] = {kCFBooleanTrue, key.ref,
                                   CFSTR("kANEFModelMIL")};
    Owned<CFDictionaryRef> options(CFDictionaryCreate(
        nullptr, option_keys, option_values, 3, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks));
    if (!url.ref || !model || !options.ref)
        return ANE_LOAD_FAILED;
    CFErrorRef error = nullptr;
    if (!send<BOOL>(client, "loadModel:options:qos:error:", model, options.ref,
                    QOS, &error))
        return ANE_LOAD_FAILED;
    program->client = (id)CFRetain(client);
    program->model = (id)CFRetain(model);
    program->loaded = true;
    return ANE_OK;
} catch (...) {
    return ANE_LOAD_FAILED;
}

int ane_launch(AneProgram *program, DLTensor **inputs, int n_inputs,
               DLTensor **outputs, int n_outputs) try {
    if (!program || !program->loaded || n_inputs != program->entry->n_inputs ||
        n_outputs != program->entry->n_outputs || (n_inputs && !inputs) ||
        !outputs)
        return ANE_BAD_ARGUMENT;
    const auto *io = program->entry->io;
    for (int i = 0; i < n_inputs; ++i)
        if (!matches(io[i], inputs[i]))
            return ANE_BAD_ARGUMENT;
    for (int i = 0; i < n_outputs; ++i)
        if (!matches(io[n_inputs + i], outputs[i]))
            return ANE_BAD_ARGUMENT;
    Pool pool;
    Request sources, results;
    for (int i = 0; i < n_inputs; ++i)
        if (!sources.add(io[i]) ||
            !transfer(io[i], *inputs[i], sources.surfaces[i], true))
            return ANE_EVALUATE_FAILED;
    for (int i = 0; i < n_outputs; ++i)
        if (!results.add(io[n_inputs + i]))
            return ANE_EVALUATE_FAILED;
    Owned<CFNumberRef> procedure(number(0));
    id request =
        send<id>(class_named("_ANERequest"),
                 "requestWithInputs:inputIndices:outputs:outputIndices:"
                 "weightsBuffer:perfStats:procedureIndex:",
                 sources.objects, sources.indices, results.objects,
                 results.indices, (id) nullptr, (id) nullptr, procedure.ref);
    CFErrorRef error = nullptr;
    if (!request ||
        !send<BOOL>(program->client,
                    "evaluateWithModel:options:request:qos:error:",
                    program->model, program->options, request, QOS, &error))
        return ANE_EVALUATE_FAILED;
    for (int i = 0; i < n_outputs; ++i)
        if (!transfer(io[n_inputs + i], *outputs[i], results.surfaces[i],
                      false))
            return ANE_EVALUATE_FAILED;
    return ANE_OK;
} catch (...) {
    return ANE_EVALUATE_FAILED;
}

int ane_destroy(AneProgram **program) try {
    if (!program)
        return ANE_BAD_ARGUMENT;
    std::unique_ptr<AneProgram> owned(*program);
    *program = nullptr;
    return owned ? owned->unload() : ANE_OK;
} catch (...) {
    return ANE_UNLOAD_FAILED;
}
