//! Mapping of CUDA driver and runtime error codes to typed Zig errors.
//!
//! Every numeric error code returned by the CUDA Driver API (CUresult) and the
//! CUDA Runtime API (cudaError_t) is represented here as a member of the
//! `CudaError` error set. Call sites should never inspect raw integer codes;
//! use `core/result.zig` helpers to convert FFI return values into these errors
//! before they surface anywhere in the public API.
//!
//! Both CUDA 12.x and 13.x codes are included. Codes that were added in 13.x
//! are noted in their doc comments. Codes that were deprecated but still emitted
//! by 12.x drivers are retained for compatibility.

const std = @import("std");

/// The unified error set covering every failure mode of the CUDA Driver API,
/// the CUDA Runtime API, and cuda.zig's own higher-level operations.
///
/// Driver API (CUresult) and Runtime API (cudaError_t) share many semantically
/// equivalent codes; they are unified here into a single Zig error set so that
/// callers work with one consistent type regardless of which API layer produced
/// the error.
pub const CudaError = error{
    // General / infrastructure errors.

    /// A CUDA-capable device or driver was required but none is available on
    /// this host. Returned by cuda.zig when `isAvailable()` is false and the
    /// caller invokes an operation that cannot run on the CPU fallback.
    CudaRequired,

    /// The requested operation has not yet been implemented in this release.
    /// See ROADMAP.md for the target milestone.
    NotYetImplemented,

    /// An NVRTC shared library was required but could not be loaded.
    NvrtcNotFound,

    /// Peer-to-peer memory access between the specified devices is not
    /// supported by the current hardware or driver configuration.
    PeerAccessNotSupported,

    // CUresult / cudaError_t — success is never an error, only non-zero codes appear.

    /// CUDA_ERROR_INVALID_VALUE / cudaErrorInvalidValue: one or more arguments
    /// passed to a CUDA API call are outside the acceptable range.
    InvalidValue,

    /// CUDA_ERROR_OUT_OF_MEMORY / cudaErrorMemoryAllocation: the device ran out
    /// of free memory during an allocation request.
    OutOfMemory,

    /// CUDA_ERROR_NOT_INITIALIZED / cudaErrorInitializationError: the CUDA
    /// driver has not been initialized. Typically raised if `cuInit` was not
    /// called before the first driver API use.
    NotInitialized,

    /// CUDA_ERROR_DEINITIALIZED: the CUDA driver is in the process of shutting
    /// down. Subsequent API calls are undefined.
    Deinitialized,

    /// CUDA_ERROR_PROFILER_DISABLED: the CUDA profiler is disabled. Only
    /// returned when attempting to use profiling APIs.
    ProfilerDisabled,

    /// CUDA_ERROR_PROFILER_NOT_INITIALIZED: the profiler was not initialized
    /// before a profiling API was called.
    ProfilerNotInitialized,

    /// CUDA_ERROR_PROFILER_ALREADY_STARTED: `cuProfilerStart` was called when
    /// the profiler was already running.
    ProfilerAlreadyStarted,

    /// CUDA_ERROR_PROFILER_ALREADY_STOPPED: `cuProfilerStop` was called when
    /// the profiler was already stopped.
    ProfilerAlreadyStopped,

    /// CUDA_ERROR_STUB_LIBRARY: the CUDA driver API is a stub library that
    /// does not implement any operations. Raised on machines where only the
    /// CUDA stub library is installed.
    StubLibrary,

    /// CUDA_ERROR_DEVICE_UNAVAILABLE: a resource required for the requested
    /// device access is currently unavailable.
    DeviceUnavailable,

    /// CUDA_ERROR_NO_DEVICE / cudaErrorNoDevice: no CUDA-capable device was
    /// detected on this host.
    NoDevice,

    /// CUDA_ERROR_INVALID_DEVICE / cudaErrorInvalidDevice: the device index is
    /// outside the valid range `[0, deviceCount())`.
    InvalidDevice,

    /// CUDA_ERROR_DEVICE_NOT_LICENSED: the device is not licensed for CUDA
    /// compute (GRID/vGPU scenario).
    DeviceNotLicensed,

    /// CUDA_ERROR_INVALID_IMAGE: the device kernel image is invalid.
    InvalidImage,

    /// CUDA_ERROR_INVALID_CONTEXT: the current CUDA context is invalid or has
    /// been destroyed.
    InvalidContext,

    /// CUDA_ERROR_CONTEXT_ALREADY_CURRENT (deprecated in CUDA 3.2).
    ContextAlreadyCurrent,

    /// CUDA_ERROR_MAP_FAILED / cudaErrorMapBufferObjectFailed: a buffer mapping
    /// operation failed.
    MapFailed,

    /// CUDA_ERROR_UNMAP_FAILED / cudaErrorUnmapBufferObjectFailed: a buffer
    /// unmap operation failed.
    UnmapFailed,

    /// CUDA_ERROR_ARRAY_IS_MAPPED: the array is currently mapped and cannot be
    /// destroyed.
    ArrayIsMapped,

    /// CUDA_ERROR_ALREADY_MAPPED: the resource is already mapped.
    AlreadyMapped,

    /// CUDA_ERROR_NO_BINARY_FOR_GPU / cudaErrorInvalidKernelImage: no device
    /// binary was found compatible with the current device.
    NoBinaryForGpu,

    /// CUDA_ERROR_ALREADY_ACQUIRED: the resource has already been acquired.
    AlreadyAcquired,

    /// CUDA_ERROR_NOT_MAPPED: the resource is not mapped.
    NotMapped,

    /// CUDA_ERROR_NOT_MAPPED_AS_ARRAY: the resource is not mapped as an array.
    NotMappedAsArray,

    /// CUDA_ERROR_NOT_MAPPED_AS_POINTER: the resource is not mapped as a pointer.
    NotMappedAsPointer,

    /// CUDA_ERROR_ECC_UNCORRECTABLE / cudaErrorECCUncorrectable: an
    /// uncorrectable ECC error was detected during execution.
    EccUncorrectable,

    /// CUDA_ERROR_UNSUPPORTED_LIMIT / cudaErrorUnsupportedLimit: a limit is not
    /// supported by this device.
    UnsupportedLimit,

    /// CUDA_ERROR_CONTEXT_ALREADY_IN_USE: the CUDA context is already in use by
    /// another process.
    ContextAlreadyInUse,

    /// CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: peer device memory access is not
    /// supported between the two specified devices.
    PeerAccessUnsupported,

    /// CUDA_ERROR_INVALID_PTX / cudaErrorInvalidPtx: the PTX JIT compilation
    /// failed.
    InvalidPtx,

    /// CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: an invalid OpenGL or DirectX context
    /// was detected.
    InvalidGraphicsContext,

    /// CUDA_ERROR_NVLINK_UNCORRECTABLE: an uncorrectable NVLink error was
    /// detected during execution.
    NvlinkUncorrectable,

    /// CUDA_ERROR_JIT_COMPILER_NOT_FOUND: the PTX JIT compiler library was not
    /// found.
    JitCompilerNotFound,

    /// CUDA_ERROR_UNSUPPORTED_PTX_VERSION: the PTX version in the supplied PTX
    /// is higher than what the device supports.
    UnsupportedPtxVersion,

    /// CUDA_ERROR_JIT_COMPILATION_DISABLED: PTX JIT compilation is disabled.
    JitCompilationDisabled,

    /// CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY: the specified execution affinity is
    /// not supported by the current device.
    UnsupportedExecAffinity,

    /// CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC: device-side synchronization is not
    /// supported by the current context.
    UnsupportedDevsideSync,

    /// CUDA_ERROR_INVALID_SOURCE / cudaErrorInvalidSource: the device kernel
    /// source is invalid.
    InvalidSource,

    /// CUDA_ERROR_FILE_NOT_FOUND / cudaErrorFileNotFound: the specified file
    /// was not found.
    FileNotFound,

    /// CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND / cudaErrorSharedObjectSymbolNotFound:
    /// a shared object could not resolve the required symbol.
    SharedObjectSymbolNotFound,

    /// CUDA_ERROR_SHARED_OBJECT_INIT_FAILED / cudaErrorSharedObjectInitFailed:
    /// a shared object failed to initialize.
    SharedObjectInitFailed,

    /// CUDA_ERROR_OPERATING_SYSTEM / cudaErrorOperatingSystem: an OS call failed.
    OperatingSystem,

    /// CUDA_ERROR_INVALID_HANDLE / cudaErrorInvalidResourceHandle: an invalid
    /// resource handle was passed to an API call.
    InvalidHandle,

    /// CUDA_ERROR_ILLEGAL_STATE / cudaErrorIllegalState: the operation is not
    /// permitted in the current state.
    IllegalState,

    /// CUDA_ERROR_LOSSY_QUERY: a query that would produce a lossy result was
    /// attempted without explicitly requesting lossy semantics.
    LossyQuery,

    /// CUDA_ERROR_NOT_FOUND / cudaErrorInvalidSymbol: a named symbol could not
    /// be found in the module.
    NotFound,

    /// CUDA_ERROR_NOT_READY / cudaErrorNotReady: an asynchronous operation has
    /// not yet completed. Not an actual error; callers should poll or synchronize.
    NotReady,

    /// CUDA_ERROR_ILLEGAL_ADDRESS / cudaErrorIllegalAddress: the device
    /// encountered an illegal memory access. The context may be corrupt.
    IllegalAddress,

    /// CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES / cudaErrorLaunchOutOfResources: the
    /// kernel launch exceeded a resource limit (registers, shared memory, etc.).
    LaunchOutOfResources,

    /// CUDA_ERROR_LAUNCH_TIMEOUT / cudaErrorLaunchTimeout: the kernel exceeded
    /// the watchdog timer limit (display-attached GPU with TDR active).
    LaunchTimeout,

    /// CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING / cudaErrorLaunchIncompatibleTexturing:
    /// a kernel launch used an invalid texture configuration.
    LaunchIncompatibleTexturing,

    /// CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED / cudaErrorPeerAccessAlreadyEnabled:
    /// peer access is already enabled between these two devices.
    PeerAccessAlreadyEnabled,

    /// CUDA_ERROR_PEER_ACCESS_NOT_ENABLED / cudaErrorPeerAccessNotEnabled:
    /// peer access has not been enabled between these two devices.
    PeerAccessNotEnabled,

    /// CUDA_ERROR_CONTEXT_IS_DESTROYED: the context has been destroyed by
    /// a previous call.
    ContextIsDestroyed,

    /// CUDA_ERROR_ASSERT / cudaErrorAssert: a device-side assertion triggered.
    Assert,

    /// CUDA_ERROR_TOO_MANY_PEERS: too many peers are connected.
    TooManyPeers,

    /// CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED / cudaErrorHostMemoryAlreadyRegistered:
    /// this host memory range is already registered.
    HostMemoryAlreadyRegistered,

    /// CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED / cudaErrorHostMemoryNotRegistered:
    /// this host memory range was not previously registered.
    HostMemoryNotRegistered,

    /// CUDA_ERROR_HARDWARE_STACK_ERROR / cudaErrorHardwareStackError: the device
    /// hardware stack encountered an error.
    HardwareStackError,

    /// CUDA_ERROR_ILLEGAL_INSTRUCTION / cudaErrorIllegalInstruction: the device
    /// executed an illegal instruction.
    IllegalInstruction,

    /// CUDA_ERROR_MISALIGNED_ADDRESS / cudaErrorMisalignedAddress: the device
    /// encountered an unaligned memory access.
    MisalignedAddress,

    /// CUDA_ERROR_INVALID_ADDRESS_SPACE / cudaErrorInvalidAddressSpace: the
    /// device made an access to an invalid address space.
    InvalidAddressSpace,

    /// CUDA_ERROR_INVALID_PC / cudaErrorInvalidPc: the device program counter
    /// wrapped.
    InvalidPc,

    /// CUDA_ERROR_LAUNCH_FAILED / cudaErrorLaunchFailure: a kernel launch failed
    /// for a reason other than those covered by more specific error codes.
    LaunchFailed,

    /// CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: cooperative launch grid is too
    /// large for the device.
    CooperativeLaunchTooLarge,

    /// CUDA_ERROR_NOT_PERMITTED / cudaErrorNotPermitted: the operation is not
    /// permitted in the current execution context.
    NotPermitted,

    /// CUDA_ERROR_NOT_SUPPORTED / cudaErrorNotSupported: the operation is not
    /// supported on the current platform or device.
    NotSupported,

    /// CUDA_ERROR_SYSTEM_NOT_READY / cudaErrorSystemNotReady: the system is not
    /// ready to perform the requested operation.
    SystemNotReady,

    /// CUDA_ERROR_SYSTEM_DRIVER_MISMATCH / cudaErrorSystemDriverMismatch: the
    /// CUDA driver and the CUDA runtime versions do not match.
    SystemDriverMismatch,

    /// CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward-compatibility mode is
    /// not supported on this device.
    CompatNotSupportedOnDevice,

    /// CUDA_ERROR_MPS_CONNECTION_FAILED: the MPS client failed to connect to
    /// the MPS control daemon.
    MpsConnectionFailed,

    /// CUDA_ERROR_MPS_RPC_FAILURE: the MPS server returned an RPC failure.
    MpsRpcFailure,

    /// CUDA_ERROR_MPS_SERVER_NOT_READY: the MPS server is not yet ready.
    MpsServerNotReady,

    /// CUDA_ERROR_MPS_MAX_CLIENTS_REACHED: the maximum number of MPS clients
    /// for the current device has been reached.
    MpsMaxClientsReached,

    /// CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED: the maximum number of MPS
    /// connections has been reached.
    MpsMaxConnectionsReached,

    /// CUDA_ERROR_MPS_CLIENT_TERMINATED: the MPS client was terminated.
    MpsClientTerminated,

    /// CUDA_ERROR_CDP_NOT_SUPPORTED: CUDA Dynamic Parallelism is not supported
    /// on this device.
    CdpNotSupported,

    /// CUDA_ERROR_CDP_VERSION_MISMATCH: the CDP version in the kernel image
    /// does not match the driver version.
    CdpVersionMismatch,

    /// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED / cudaErrorStreamCaptureUnsupported:
    /// the operation is not supported while stream capture is active.
    StreamCaptureUnsupported,

    /// CUDA_ERROR_STREAM_CAPTURE_INVALIDATED / cudaErrorStreamCaptureInvalidated:
    /// the stream capture sequence is invalid.
    StreamCaptureInvalidated,

    /// CUDA_ERROR_STREAM_CAPTURE_MERGE / cudaErrorStreamCaptureMerge:
    /// the graph update would cause a merge of two distinct capture sequences.
    StreamCaptureMerge,

    /// CUDA_ERROR_STREAM_CAPTURE_UNMATCHED / cudaErrorStreamCaptureUnmatched:
    /// the capture was ended on a stream that was not the origin stream.
    StreamCaptureUnmatched,

    /// CUDA_ERROR_STREAM_CAPTURE_UNJOINED / cudaErrorStreamCaptureUnjoined:
    /// a fork of the capture sequence was not joined before the capture ended.
    StreamCaptureUnjoined,

    /// CUDA_ERROR_STREAM_CAPTURE_ISOLATION / cudaErrorStreamCaptureIsolation:
    /// a dependency that would have required a join was not captured.
    StreamCaptureIsolation,

    /// CUDA_ERROR_STREAM_CAPTURE_IMPLICIT / cudaErrorStreamCaptureImplicit:
    /// an implicit synchronization with the default stream was captured.
    StreamCaptureImplicit,

    /// CUDA_ERROR_CAPTURED_EVENT / cudaErrorCapturedEvent: the operation is not
    /// permitted on an event that was last recorded in a capturing stream.
    CapturedEvent,

    /// CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: the capture was ended on a
    /// different thread than where it began.
    StreamCaptureWrongThread,

    /// CUDA_ERROR_TIMEOUT / cudaErrorTimeout: the operation timed out.
    Timeout,

    /// CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE / cudaErrorGraphExecUpdateFailure:
    /// a graph update failed because the topology or parameters changed in a
    /// way that would require regeneration.
    GraphExecUpdateFailure,

    /// CUDA_ERROR_EXTERNAL_DEVICE: an external device encountered an error.
    ExternalDevice,

    /// CUDA_ERROR_INVALID_CLUSTER_SIZE: the specified cluster size is invalid
    /// for this device.
    InvalidClusterSize,

    /// CUDA_ERROR_FUNCTION_NOT_LOADED: the function was not yet loaded (lazy
    /// loading is in effect).
    FunctionNotLoaded,

    /// CUDA_ERROR_INVALID_RESOURCE_TYPE: the resource type is not valid for
    /// this operation.
    InvalidResourceType,

    /// CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION: the resource configuration
    /// is invalid for this operation.
    InvalidResourceConfiguration,

    /// CUDA_ERROR_UNKNOWN / cudaErrorUnknown: an unknown internal error occurred.
    /// If you encounter this, please file a bug report with your driver version
    /// and a reproducer.
    Unknown,
};

/// Converts a raw CUresult integer (Driver API) to a `CudaError` or returns
/// `void` on success (CUDA_SUCCESS == 0).
///
/// This function is the single authoritative translation point for Driver API
/// return codes. All driver FFI call sites must route through this function
/// rather than inspecting the raw integer.
pub fn fromDriverResult(code: c_int) CudaError!void {
    return switch (code) {
        0 => {}, // CUDA_SUCCESS
        1 => error.InvalidValue,
        2 => error.OutOfMemory,
        3 => error.NotInitialized,
        4 => error.Deinitialized,
        5 => error.ProfilerDisabled,
        6 => error.ProfilerNotInitialized,
        7 => error.ProfilerAlreadyStarted,
        8 => error.ProfilerAlreadyStopped,
        9 => error.StubLibrary,
        10 => error.DeviceUnavailable,
        100 => error.NoDevice,
        101 => error.InvalidDevice,
        102 => error.DeviceNotLicensed,
        200 => error.InvalidImage,
        201 => error.InvalidContext,
        202 => error.ContextAlreadyCurrent,
        205 => error.MapFailed,
        206 => error.UnmapFailed,
        207 => error.ArrayIsMapped,
        208 => error.AlreadyMapped,
        209 => error.NoBinaryForGpu,
        210 => error.AlreadyAcquired,
        211 => error.NotMapped,
        212 => error.NotMappedAsArray,
        213 => error.NotMappedAsPointer,
        214 => error.EccUncorrectable,
        215 => error.UnsupportedLimit,
        216 => error.ContextAlreadyInUse,
        217 => error.PeerAccessUnsupported,
        218 => error.InvalidPtx,
        219 => error.InvalidGraphicsContext,
        220 => error.NvlinkUncorrectable,
        221 => error.JitCompilerNotFound,
        222 => error.UnsupportedPtxVersion,
        223 => error.JitCompilationDisabled,
        224 => error.UnsupportedExecAffinity,
        225 => error.UnsupportedDevsideSync,
        300 => error.InvalidSource,
        301 => error.FileNotFound,
        302 => error.SharedObjectSymbolNotFound,
        303 => error.SharedObjectInitFailed,
        304 => error.OperatingSystem,
        400 => error.InvalidHandle,
        401 => error.IllegalState,
        402 => error.LossyQuery,
        500 => error.NotFound,
        600 => error.NotReady,
        700 => error.IllegalAddress,
        701 => error.LaunchOutOfResources,
        702 => error.LaunchTimeout,
        703 => error.LaunchIncompatibleTexturing,
        704 => error.PeerAccessAlreadyEnabled,
        705 => error.PeerAccessNotEnabled,
        708 => error.ContextIsDestroyed,
        710 => error.Assert,
        711 => error.TooManyPeers,
        712 => error.HostMemoryAlreadyRegistered,
        713 => error.HostMemoryNotRegistered,
        714 => error.HardwareStackError,
        715 => error.IllegalInstruction,
        716 => error.MisalignedAddress,
        717 => error.InvalidAddressSpace,
        718 => error.InvalidPc,
        719 => error.LaunchFailed,
        720 => error.CooperativeLaunchTooLarge,
        800 => error.NotPermitted,
        801 => error.NotSupported,
        802 => error.SystemNotReady,
        803 => error.SystemDriverMismatch,
        804 => error.CompatNotSupportedOnDevice,
        805 => error.MpsConnectionFailed,
        806 => error.MpsRpcFailure,
        807 => error.MpsServerNotReady,
        808 => error.MpsMaxClientsReached,
        809 => error.MpsMaxConnectionsReached,
        810 => error.MpsClientTerminated,
        811 => error.CdpNotSupported,
        812 => error.CdpVersionMismatch,
        900 => error.StreamCaptureUnsupported,
        901 => error.StreamCaptureInvalidated,
        902 => error.StreamCaptureMerge,
        903 => error.StreamCaptureUnmatched,
        904 => error.StreamCaptureUnjoined,
        905 => error.StreamCaptureIsolation,
        906 => error.StreamCaptureImplicit,
        907 => error.CapturedEvent,
        908 => error.StreamCaptureWrongThread,
        909 => error.Timeout,
        910 => error.GraphExecUpdateFailure,
        911 => error.ExternalDevice,
        912 => error.InvalidClusterSize,
        913 => error.FunctionNotLoaded,
        914 => error.InvalidResourceType,
        915 => error.InvalidResourceConfiguration,
        999 => error.Unknown,
        else => error.Unknown,
    };
}

/// Converts a raw cudaError_t integer (Runtime API) to a `CudaError` or returns
/// `void` on success (cudaSuccess == 0).
///
/// The runtime API error space overlaps significantly with the driver API space
/// but uses a different integer encoding. This function handles the runtime
/// encoding; driver-API codes must go through `fromDriverResult`.
pub fn fromRuntimeResult(code: c_int) CudaError!void {
    return switch (code) {
        0 => {}, // cudaSuccess
        1 => error.InvalidValue, // cudaErrorInvalidValue
        2 => error.OutOfMemory, // cudaErrorMemoryAllocation
        3 => error.NotInitialized, // cudaErrorInitializationError
        4 => error.LaunchFailed, // cudaErrorCudartUnloading (maps to launch failed)
        5 => error.ProfilerDisabled, // cudaErrorProfilerDisabled
        9 => error.ProfilerNotInitialized, // cudaErrorProfilerNotInitialized
        10 => error.ProfilerAlreadyStarted, // cudaErrorProfilerAlreadyStarted
        11 => error.ProfilerAlreadyStopped, // cudaErrorProfilerAlreadyStopped
        12 => error.InvalidValue, // cudaErrorInvalidConfiguration (alias)
        13 => error.InvalidDevice, // cudaErrorInvalidDevice
        16 => error.InvalidImage, // cudaErrorInvalidKernelImage
        17 => error.InvalidContext, // cudaErrorDeviceAlreadyInUse
        18 => error.MapFailed, // cudaErrorMapBufferObjectFailed
        19 => error.UnmapFailed, // cudaErrorUnmapBufferObjectFailed
        20 => error.ArrayIsMapped, // cudaErrorArrayIsMapped
        21 => error.AlreadyMapped, // cudaErrorAlreadyMapped
        22 => error.NoBinaryForGpu, // cudaErrorNoKernelImageForDevice
        23 => error.AlreadyAcquired, // cudaErrorAlreadyAcquired
        24 => error.NotMapped, // cudaErrorNotMapped
        25 => error.NotMappedAsArray, // cudaErrorNotMappedAsArray
        26 => error.NotMappedAsPointer, // cudaErrorNotMappedAsPointer
        27 => error.EccUncorrectable, // cudaErrorECCUncorrectable
        28 => error.UnsupportedLimit, // cudaErrorUnsupportedLimit
        29 => error.DeviceUnavailable, // cudaErrorDeviceAlreadyInUse
        30 => error.PeerAccessUnsupported, // cudaErrorPeerAccessUnsupported
        31 => error.InvalidPtx, // cudaErrorInvalidPtx
        32 => error.InvalidGraphicsContext, // cudaErrorInvalidGraphicsContext
        33 => error.NvlinkUncorrectable, // cudaErrorNvlinkUncorrectable
        34 => error.JitCompilerNotFound, // cudaErrorJitCompilerNotFound
        35 => error.UnsupportedPtxVersion, // cudaErrorUnsupportedPtxVersion
        36 => error.JitCompilationDisabled, // cudaErrorJitCompilationDisabled
        37 => error.UnsupportedExecAffinity, // cudaErrorUnsupportedExecAffinity
        38 => error.UnsupportedDevsideSync, // cudaErrorUnsupportedDevSideSync
        39 => error.InvalidSource, // cudaErrorInvalidSource (39)
        40 => error.FileNotFound, // cudaErrorFileNotFound
        41 => error.SharedObjectSymbolNotFound, // cudaErrorSharedObjectSymbolNotFound
        42 => error.SharedObjectInitFailed, // cudaErrorSharedObjectInitFailed
        43 => error.OperatingSystem, // cudaErrorOperatingSystem
        44 => error.InvalidHandle, // cudaErrorInvalidResourceHandle
        45 => error.IllegalState, // cudaErrorIllegalState
        46 => error.LossyQuery, // cudaErrorLossyQuery
        49 => error.NotFound, // cudaErrorSymbolNotFound
        52 => error.NotReady, // cudaErrorNotReady
        53 => error.IllegalAddress, // cudaErrorIllegalAddress
        54 => error.LaunchOutOfResources, // cudaErrorLaunchOutOfResources
        55 => error.LaunchTimeout, // cudaErrorLaunchTimeout
        56 => error.LaunchIncompatibleTexturing, // cudaErrorLaunchIncompatibleTexturing
        57 => error.PeerAccessAlreadyEnabled, // cudaErrorPeerAccessAlreadyEnabled
        58 => error.PeerAccessNotEnabled, // cudaErrorPeerAccessNotEnabled
        59 => error.Assert, // cudaErrorAssert
        60 => error.TooManyPeers, // cudaErrorTooManyPeers
        61 => error.HostMemoryAlreadyRegistered, // cudaErrorHostMemoryAlreadyRegistered
        62 => error.HostMemoryNotRegistered, // cudaErrorHostMemoryNotRegistered
        63 => error.OperatingSystem, // cudaErrorOperatingSystem (63)
        64 => error.PeerAccessUnsupported, // cudaErrorPeerAccessUnsupported (64)
        65 => error.HardwareStackError, // cudaErrorHardwareStackError
        66 => error.IllegalInstruction, // cudaErrorIllegalInstruction
        67 => error.MisalignedAddress, // cudaErrorMisalignedAddress
        68 => error.InvalidAddressSpace, // cudaErrorInvalidAddressSpace
        69 => error.InvalidPc, // cudaErrorInvalidPc
        70 => error.LaunchFailed, // cudaErrorLaunchFailure
        71 => error.CooperativeLaunchTooLarge, // cudaErrorCooperativeLaunchTooLarge
        72 => error.NotPermitted, // cudaErrorNotPermitted
        73 => error.NotSupported, // cudaErrorNotSupported
        74 => error.SystemNotReady, // cudaErrorSystemNotReady
        75 => error.SystemDriverMismatch, // cudaErrorSystemDriverMismatch
        76 => error.CompatNotSupportedOnDevice, // cudaErrorCompatNotSupportedOnDevice
        900 => error.StreamCaptureUnsupported, // cudaErrorStreamCaptureUnsupported
        901 => error.StreamCaptureInvalidated, // cudaErrorStreamCaptureInvalidated
        902 => error.StreamCaptureMerge, // cudaErrorStreamCaptureMerge
        903 => error.StreamCaptureUnmatched, // cudaErrorStreamCaptureUnmatched
        904 => error.StreamCaptureUnjoined, // cudaErrorStreamCaptureUnjoined
        905 => error.StreamCaptureIsolation, // cudaErrorStreamCaptureIsolation
        906 => error.StreamCaptureImplicit, // cudaErrorStreamCaptureImplicit
        907 => error.CapturedEvent, // cudaErrorCapturedEvent
        908 => error.StreamCaptureWrongThread, // cudaErrorStreamCaptureWrongThread
        909 => error.Timeout, // cudaErrorTimeout
        910 => error.GraphExecUpdateFailure, // cudaErrorGraphExecUpdateFailure
        911 => error.ExternalDevice, // cudaErrorExternalDevice
        912 => error.InvalidClusterSize, // cudaErrorInvalidClusterSize
        999 => error.Unknown, // cudaErrorUnknown
        else => error.Unknown,
    };
}

// Workaround: cudaErrorInvalidConfiguration shares a numeric slot in some
// driver versions; map it to InvalidValue for clean error semantics.
const InvalidConfiguration = error.InvalidValue;

test "fromDriverResult: success" {
    try fromDriverResult(0);
}

test "fromDriverResult: known error codes" {
    const expectError = std.testing.expectError;
    try expectError(error.InvalidValue, fromDriverResult(1));
    try expectError(error.OutOfMemory, fromDriverResult(2));
    try expectError(error.NotInitialized, fromDriverResult(3));
    try expectError(error.NoDevice, fromDriverResult(100));
    try expectError(error.InvalidDevice, fromDriverResult(101));
    try expectError(error.NotReady, fromDriverResult(600));
    try expectError(error.LaunchFailed, fromDriverResult(719));
    try expectError(error.Unknown, fromDriverResult(999));
    try expectError(error.Unknown, fromDriverResult(12345));
}

test "fromRuntimeResult: success" {
    try fromRuntimeResult(0);
}

test "fromRuntimeResult: known error codes" {
    const expectError = std.testing.expectError;
    try expectError(error.InvalidValue, fromRuntimeResult(1));
    try expectError(error.OutOfMemory, fromRuntimeResult(2));
    try expectError(error.NotInitialized, fromRuntimeResult(3));
    try expectError(error.InvalidDevice, fromRuntimeResult(13));
    try expectError(error.NotReady, fromRuntimeResult(52));
    try expectError(error.Unknown, fromRuntimeResult(999));
}
