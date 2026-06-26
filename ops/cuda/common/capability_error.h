#pragma once

// E04-01 stable cross-language capability taxonomy.  Numeric values are part
// of the replay contract; append new values, never renumber existing ones.
namespace hqsb {

enum class CapabilityStageCode : int {
  kDiscovery = 1,
  kImport = 2,
  kVersion = 3,
  kDevice = 4,
  kCompiler = 5,
  kCompile = 6,
  kLoad = 7,
  kExecute = 8,
  kResource = 9,
  kPolicy = 10,
};

enum class CapabilityReasonCode : int {
  kAvailable = 0,
  kPackageNotInstalled = 1,
  kSharedLibraryNotFound = 2,
  kVersionIncompatible = 3,
  kAbiMismatch = 4,
  kDeviceUnavailable = 5,
  kArchUnsupported = 6,
  kCompilerUnavailable = 7,
  kCompileFailed = 8,
  kModuleLoadFailed = 9,
  kSymbolMissing = 10,
  kRuntimeFailed = 11,
  kOutOfMemory = 12,
  kPermissionDenied = 13,
  kTimeout = 14,
  kDisabledByPolicy = 15,
  kProbeInternalError = 16,
};

}  // namespace hqsb
