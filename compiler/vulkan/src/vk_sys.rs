//! Raw Vulkan C bindings — replaces ash crate.
//!
//! Direct `extern "system"` declarations against:
//!   Windows: vulkan-1.lib / vulkan-1.dll
//!   Linux:   libvulkan.so
//!
//! Only declares the ~35 Vulkan functions used by OctoFlow's compute path.
//! Struct layouts match the Vulkan 1.0 specification exactly.

#![allow(non_camel_case_types, non_snake_case, dead_code, clippy::all)]

use std::ffi::{c_char, c_void};

// ── Primitive typedefs ────────────────────────────────────────────────────────

pub type VkBool32 = u32;
pub type VkDeviceSize = u64;
pub type VkFlags = u32;
pub type VkSampleCountFlags = VkFlags;
pub type VkResult = i32;

// ── Result codes ─────────────────────────────────────────────────────────────

pub const VK_SUCCESS: VkResult = 0;
pub const VK_NOT_READY: VkResult = 1;
pub const VK_TIMEOUT: VkResult = 2;

// Error result codes (Vulkan 1.0 specification)
pub const VK_ERROR_OUT_OF_HOST_MEMORY: VkResult = -1;
pub const VK_ERROR_OUT_OF_DEVICE_MEMORY: VkResult = -2;
pub const VK_ERROR_INITIALIZATION_FAILED: VkResult = -3;
pub const VK_ERROR_DEVICE_LOST: VkResult = -4;
pub const VK_ERROR_MEMORY_MAP_FAILED: VkResult = -5;
pub const VK_ERROR_LAYER_NOT_PRESENT: VkResult = -6;
pub const VK_ERROR_EXTENSION_NOT_PRESENT: VkResult = -7;
pub const VK_ERROR_FEATURE_NOT_PRESENT: VkResult = -8;
pub const VK_ERROR_INCOMPATIBLE_DRIVER: VkResult = -9;
pub const VK_ERROR_TOO_MANY_OBJECTS: VkResult = -10;
pub const VK_ERROR_FORMAT_NOT_SUPPORTED: VkResult = -11;

/// Human-readable name for a VkResult code.
pub fn vk_result_name(result: VkResult) -> &'static str {
    match result {
        VK_SUCCESS => "VK_SUCCESS",
        VK_NOT_READY => "VK_NOT_READY",
        VK_TIMEOUT => "VK_TIMEOUT",
        VK_ERROR_OUT_OF_HOST_MEMORY => "VK_ERROR_OUT_OF_HOST_MEMORY",
        VK_ERROR_OUT_OF_DEVICE_MEMORY => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        VK_ERROR_INITIALIZATION_FAILED => "VK_ERROR_INITIALIZATION_FAILED",
        VK_ERROR_DEVICE_LOST => "VK_ERROR_DEVICE_LOST",
        VK_ERROR_MEMORY_MAP_FAILED => "VK_ERROR_MEMORY_MAP_FAILED",
        VK_ERROR_LAYER_NOT_PRESENT => "VK_ERROR_LAYER_NOT_PRESENT",
        VK_ERROR_EXTENSION_NOT_PRESENT => "VK_ERROR_EXTENSION_NOT_PRESENT",
        VK_ERROR_FEATURE_NOT_PRESENT => "VK_ERROR_FEATURE_NOT_PRESENT",
        VK_ERROR_INCOMPATIBLE_DRIVER => "VK_ERROR_INCOMPATIBLE_DRIVER",
        VK_ERROR_TOO_MANY_OBJECTS => "VK_ERROR_TOO_MANY_OBJECTS",
        VK_ERROR_FORMAT_NOT_SUPPORTED => "VK_ERROR_FORMAT_NOT_SUPPORTED",
        _ => "VK_ERROR_UNKNOWN",
    }
}

// ── Limits constants ─────────────────────────────────────────────────────────

pub const VK_WHOLE_SIZE: VkDeviceSize = u64::MAX;
pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub const VK_UUID_SIZE: usize = 16;
pub const VK_MAX_MEMORY_TYPES: usize = 32;
pub const VK_MAX_MEMORY_HEAPS: usize = 16;

// ── VkStructureType values ────────────────────────────────────────────────────

pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO: u32 = 9;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: u32 = 51;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: u32 = 1000059000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: u32 = 1000082000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: u32 = 1000083000;

// ── Flag bits ─────────────────────────────────────────────────────────────────

pub const VK_QUEUE_COMPUTE_BIT: VkFlags                     = 0x00000002;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkFlags      = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkFlags      = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkFlags     = 0x00000004;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: VkFlags       = 0x00000008;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkFlags         = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkFlags         = 0x00000002;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkFlags       = 0x00000020;
pub const VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: VkFlags      = 0x00000100;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkFlags = 0x00000001;
pub const VK_ACCESS_TRANSFER_WRITE_BIT: VkFlags             = 0x00001000;
pub const VK_ACCESS_SHADER_READ_BIT: VkFlags                = 0x00000020;
pub const VK_ACCESS_SHADER_WRITE_BIT: VkFlags               = 0x00000040;
pub const VK_ACCESS_TRANSFER_READ_BIT: VkFlags              = 0x00000800;
pub const VK_PIPELINE_STAGE_TRANSFER_BIT: VkFlags           = 0x00001000;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: VkFlags     = 0x00000800;
pub const VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT: VkFlags      = 0x00000002;
pub const VK_ACCESS_INDIRECT_COMMAND_READ_BIT: VkFlags      = 0x00000001;

// ── Enum values ───────────────────────────────────────────────────────────────

pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32     = 2;
pub const VK_SHARING_MODE_EXCLUSIVE: u32                 = 0;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32           = 0;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32            = 1;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32         = 7;
pub const VK_SHADER_STAGE_COMPUTE_BIT: VkFlags           = 0x00000020;

// ── Handles ───────────────────────────────────────────────────────────────────

// Dispatchable handles — pointer to opaque struct
pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;

// Non-dispatchable handles — u64 on 64-bit platforms
pub type VkBuffer = u64;
pub type VkSemaphore = u64;
pub type VkFence = u64;
pub type VkDeviceMemory = u64;
pub type VkCommandPool = u64;
pub type VkShaderModule = u64;
pub type VkPipelineLayout = u64;
pub type VkPipeline = u64;
pub type VkDescriptorSetLayout = u64;
pub type VkDescriptorSet = u64;
pub type VkDescriptorPool = u64;

pub const VK_NULL_HANDLE: u64 = 0;

// ── VkExtent3D ────────────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

// ── VkPhysicalDeviceLimits ─────────────────────────────────────────────────────
// All fields match vulkan_core.h layout exactly.

#[repr(C)]
pub struct VkPhysicalDeviceLimits {
    pub maxImageDimension1D: u32,
    pub maxImageDimension2D: u32,
    pub maxImageDimension3D: u32,
    pub maxImageDimensionCube: u32,
    pub maxImageArrayLayers: u32,
    pub maxTexelBufferElements: u32,
    pub maxUniformBufferRange: u32,
    pub maxStorageBufferRange: u32,
    pub maxPushConstantsSize: u32,
    pub maxMemoryAllocationCount: u32,
    pub maxSamplerAllocationCount: u32,
    pub bufferImageGranularity: VkDeviceSize,
    pub sparseAddressSpaceSize: VkDeviceSize,
    pub maxBoundDescriptorSets: u32,
    pub maxPerStageDescriptorSamplers: u32,
    pub maxPerStageDescriptorUniformBuffers: u32,
    pub maxPerStageDescriptorStorageBuffers: u32,
    pub maxPerStageDescriptorSampledImages: u32,
    pub maxPerStageDescriptorStorageImages: u32,
    pub maxPerStageDescriptorInputAttachments: u32,
    pub maxPerStageResources: u32,
    pub maxDescriptorSetSamplers: u32,
    pub maxDescriptorSetUniformBuffers: u32,
    pub maxDescriptorSetUniformBuffersDynamic: u32,
    pub maxDescriptorSetStorageBuffers: u32,
    pub maxDescriptorSetStorageBuffersDynamic: u32,
    pub maxDescriptorSetSampledImages: u32,
    pub maxDescriptorSetStorageImages: u32,
    pub maxDescriptorSetInputAttachments: u32,
    pub maxVertexInputAttributes: u32,
    pub maxVertexInputBindings: u32,
    pub maxVertexInputAttributeOffset: u32,
    pub maxVertexInputBindingStride: u32,
    pub maxVertexOutputComponents: u32,
    pub maxTessellationGenerationLevel: u32,
    pub maxTessellationPatchSize: u32,
    pub maxTessellationControlPerVertexInputComponents: u32,
    pub maxTessellationControlPerVertexOutputComponents: u32,
    pub maxTessellationControlPerPatchOutputComponents: u32,
    pub maxTessellationControlTotalOutputComponents: u32,
    pub maxTessellationEvaluationInputComponents: u32,
    pub maxTessellationEvaluationOutputComponents: u32,
    pub maxGeometryShaderInvocations: u32,
    pub maxGeometryInputComponents: u32,
    pub maxGeometryOutputComponents: u32,
    pub maxGeometryOutputVertices: u32,
    pub maxGeometryTotalOutputComponents: u32,
    pub maxFragmentInputComponents: u32,
    pub maxFragmentOutputAttachments: u32,
    pub maxFragmentDualSrcAttachments: u32,
    pub maxFragmentCombinedOutputResources: u32,
    pub maxComputeSharedMemorySize: u32,
    pub maxComputeWorkGroupCount: [u32; 3],
    pub maxComputeWorkGroupInvocations: u32,
    pub maxComputeWorkGroupSize: [u32; 3],
    pub subPixelPrecisionBits: u32,
    pub subTexelPrecisionBits: u32,
    pub mipmapPrecisionBits: u32,
    pub maxDrawIndexedIndexValue: u32,
    pub maxDrawIndirectCount: u32,
    pub maxSamplerLodBias: f32,
    pub maxSamplerAnisotropy: f32,
    pub maxViewports: u32,
    pub maxViewportDimensions: [u32; 2],
    pub viewportBoundsRange: [f32; 2],
    pub viewportSubPixelBits: u32,
    pub minMemoryMapAlignment: usize,
    pub minTexelBufferOffsetAlignment: VkDeviceSize,
    pub minUniformBufferOffsetAlignment: VkDeviceSize,
    pub minStorageBufferOffsetAlignment: VkDeviceSize,
    pub minTexelOffset: i32,
    pub maxTexelOffset: u32,
    pub minTexelGatherOffset: i32,
    pub maxTexelGatherOffset: u32,
    pub minInterpolationOffset: f32,
    pub maxInterpolationOffset: f32,
    pub subPixelInterpolationOffsetBits: u32,
    pub maxFramebufferWidth: u32,
    pub maxFramebufferHeight: u32,
    pub maxFramebufferLayers: u32,
    pub framebufferColorSampleCounts: VkSampleCountFlags,
    pub framebufferDepthSampleCounts: VkSampleCountFlags,
    pub framebufferStencilSampleCounts: VkSampleCountFlags,
    pub framebufferNoAttachmentsSampleCounts: VkSampleCountFlags,
    pub maxColorAttachments: u32,
    pub sampledImageColorSampleCounts: VkSampleCountFlags,
    pub sampledImageIntegerSampleCounts: VkSampleCountFlags,
    pub sampledImageDepthSampleCounts: VkSampleCountFlags,
    pub sampledImageStencilSampleCounts: VkSampleCountFlags,
    pub storageImageSampleCounts: VkSampleCountFlags,
    pub maxSampleMaskWords: u32,
    pub timestampComputeAndGraphics: VkBool32,
    pub timestampPeriod: f32,
    pub maxClipDistances: u32,
    pub maxCullDistances: u32,
    pub maxCombinedClipAndCullDistances: u32,
    pub discreteQueuePriorities: u32,
    pub pointSizeRange: [f32; 2],
    pub lineWidthRange: [f32; 2],
    pub pointSizeGranularity: f32,
    pub lineWidthGranularity: f32,
    pub strictLines: VkBool32,
    pub standardSampleLocations: VkBool32,
    pub optimalBufferCopyOffsetAlignment: VkDeviceSize,
    pub optimalBufferCopyRowPitchAlignment: VkDeviceSize,
    pub nonCoherentAtomSize: VkDeviceSize,
}

// ── VkPhysicalDeviceSparseProperties ─────────────────────────────────────────

#[repr(C)]
pub struct VkPhysicalDeviceSparseProperties {
    pub residencyStandard2DBlockShape: VkBool32,
    pub residencyStandard2DMultisampleBlockShape: VkBool32,
    pub residencyStandard3DBlockShape: VkBool32,
    pub residencyAlignedMipSize: VkBool32,
    pub residencyNonResidentStrict: VkBool32,
}

// ── VkPhysicalDeviceProperties ────────────────────────────────────────────────

#[repr(C)]
pub struct VkPhysicalDeviceProperties {
    pub apiVersion: u32,
    pub driverVersion: u32,
    pub vendorID: u32,
    pub deviceID: u32,
    pub deviceType: u32,
    pub deviceName: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pub pipelineCacheUUID: [u8; VK_UUID_SIZE],
    pub limits: VkPhysicalDeviceLimits,
    pub sparseProperties: VkPhysicalDeviceSparseProperties,
}

// ── VkQueueFamilyProperties ───────────────────────────────────────────────────

#[repr(C)]
pub struct VkQueueFamilyProperties {
    pub queueFlags: VkFlags,
    pub queueCount: u32,
    pub timestampValidBits: u32,
    pub minImageTransferGranularity: VkExtent3D,
}

// ── VkMemoryType ─────────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkMemoryType {
    pub propertyFlags: VkFlags,
    pub heapIndex: u32,
}

// ── VkMemoryHeap ─────────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: VkFlags,
}

// ── VkPhysicalDeviceMemoryProperties ─────────────────────────────────────────

#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memoryTypeCount: u32,
    pub memoryTypes: [VkMemoryType; VK_MAX_MEMORY_TYPES],
    pub memoryHeapCount: u32,
    pub memoryHeaps: [VkMemoryHeap; VK_MAX_MEMORY_HEAPS],
}

// ── VkApplicationInfo ─────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkApplicationInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub pApplicationName: *const c_char,
    pub applicationVersion: u32,
    pub pEngineName: *const c_char,
    pub engineVersion: u32,
    pub apiVersion: u32,
}

// ── VkInstanceCreateInfo ──────────────────────────────────────────────────────

#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub pApplicationInfo: *const VkApplicationInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
}

// ── VkDeviceQueueCreateInfo ───────────────────────────────────────────────────

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueFamilyIndex: u32,
    pub queueCount: u32,
    pub pQueuePriorities: *const f32,
}

// ── VkDeviceCreateInfo ────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueCreateInfoCount: u32,
    pub pQueueCreateInfos: *const VkDeviceQueueCreateInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
    pub pEnabledFeatures: *const c_void,
}

// ── VkPhysicalDeviceFeatures2 (Vulkan 1.1+) ──────────────────────────────────

/// VkPhysicalDeviceFeatures — core 1.0 feature flags (55 bools).
/// Used as part of VkPhysicalDeviceFeatures2 to query device capabilities.
#[repr(C)]
pub struct VkPhysicalDeviceFeatures {
    pub robustBufferAccess: VkBool32,
    pub fullDrawIndexUint32: VkBool32,
    pub imageCubeArray: VkBool32,
    pub independentBlend: VkBool32,
    pub geometryShader: VkBool32,
    pub tessellationShader: VkBool32,
    pub sampleRateShading: VkBool32,
    pub dualSrcBlend: VkBool32,
    pub logicOp: VkBool32,
    pub multiDrawIndirect: VkBool32,
    pub drawIndirectFirstInstance: VkBool32,
    pub depthClamp: VkBool32,
    pub depthBiasClamp: VkBool32,
    pub fillModeNonSolid: VkBool32,
    pub depthBounds: VkBool32,
    pub wideLines: VkBool32,
    pub largePoints: VkBool32,
    pub alphaToOne: VkBool32,
    pub multiViewport: VkBool32,
    pub samplerAnisotropy: VkBool32,
    pub textureCompressionETC2: VkBool32,
    pub textureCompressionASTC_LDR: VkBool32,
    pub textureCompressionBC: VkBool32,
    pub occlusionQueryPrecise: VkBool32,
    pub pipelineStatisticsQuery: VkBool32,
    pub vertexPipelineStoresAndAtomics: VkBool32,
    pub fragmentStoresAndAtomics: VkBool32,
    pub shaderTessellationAndGeometryPointSize: VkBool32,
    pub shaderImageGatherExtended: VkBool32,
    pub shaderStorageImageExtendedFormats: VkBool32,
    pub shaderStorageImageMultisample: VkBool32,
    pub shaderStorageImageReadWithoutFormat: VkBool32,
    pub shaderStorageImageWriteWithoutFormat: VkBool32,
    pub shaderUniformBufferArrayDynamicIndexing: VkBool32,
    pub shaderSampledImageArrayDynamicIndexing: VkBool32,
    pub shaderStorageBufferArrayDynamicIndexing: VkBool32,
    pub shaderStorageImageArrayDynamicIndexing: VkBool32,
    pub shaderClipDistance: VkBool32,
    pub shaderCullDistance: VkBool32,
    pub shaderFloat64: VkBool32,
    pub shaderInt64: VkBool32,
    pub shaderInt16: VkBool32,
    pub shaderResourceResidency: VkBool32,
    pub shaderResourceMinLod: VkBool32,
    pub sparseBinding: VkBool32,
    pub sparseResidencyBuffer: VkBool32,
    pub sparseResidencyImage2D: VkBool32,
    pub sparseResidencyImage3D: VkBool32,
    pub sparseResidency2Samples: VkBool32,
    pub sparseResidency4Samples: VkBool32,
    pub sparseResidency8Samples: VkBool32,
    pub sparseResidency16Samples: VkBool32,
    pub sparseResidencyAliased: VkBool32,
    pub variableMultisampleRate: VkBool32,
    pub inheritedQueries: VkBool32,
}

/// VkPhysicalDeviceFeatures2 — chain head for feature queries (Vulkan 1.1+).
#[repr(C)]
pub struct VkPhysicalDeviceFeatures2 {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub features: VkPhysicalDeviceFeatures,
}

/// VkPhysicalDeviceShaderFloat16Int8Features — f16/int8 arithmetic in shaders.
#[repr(C)]
pub struct VkPhysicalDeviceShaderFloat16Int8Features {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub shaderFloat16: VkBool32,
    pub shaderInt8: VkBool32,
}

/// VkPhysicalDevice16BitStorageFeatures — f16 storage buffer load/store.
#[repr(C)]
pub struct VkPhysicalDevice16BitStorageFeatures {
    pub sType: u32,
    pub pNext: *mut c_void,
    pub storageBuffer16BitAccess: VkBool32,
    pub uniformAndStorageBuffer16BitAccess: VkBool32,
    pub storagePushConstant16: VkBool32,
    pub storageInputOutput16: VkBool32,
}

// ── VkMemoryRequirements ──────────────────────────────────────────────────────

#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memoryTypeBits: u32,
}

// ── VkMemoryAllocateInfo ──────────────────────────────────────────────────────

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub allocationSize: VkDeviceSize,
    pub memoryTypeIndex: u32,
}

// ── VkBufferCreateInfo ────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub size: VkDeviceSize,
    pub usage: VkFlags,
    pub sharingMode: u32,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
}

// ── VkBufferCopy ─────────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkBufferCopy {
    pub srcOffset: VkDeviceSize,
    pub dstOffset: VkDeviceSize,
    pub size: VkDeviceSize,
}

// ── VkMemoryBarrier ──────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkMemoryBarrier {
    pub sType: u32,
    pub pNext: *const c_void,
    pub srcAccessMask: VkFlags,
    pub dstAccessMask: VkFlags,
}

pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER: u32 = 46;

// ── VkBufferMemoryBarrier ────────────────────────────────────────────────────

#[repr(C)]
pub struct VkBufferMemoryBarrier {
    pub sType: u32,
    pub pNext: *const c_void,
    pub srcAccessMask: VkFlags,
    pub dstAccessMask: VkFlags,
    pub srcQueueFamilyIndex: u32,
    pub dstQueueFamilyIndex: u32,
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER: u32 = 44;
pub const VK_QUEUE_FAMILY_IGNORED: u32 = !0u32;

// ── VkShaderModuleCreateInfo ──────────────────────────────────────────────────

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub codeSize: usize,
    pub pCode: *const u32,
}

// ── VkDescriptorSetLayoutBinding ──────────────────────────────────────────────

#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptorType: u32,
    pub descriptorCount: u32,
    pub stageFlags: VkFlags,
    pub pImmutableSamplers: *const u64,  // VkSampler*
}

// ── VkDescriptorSetLayoutCreateInfo ──────────────────────────────────────────

#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub bindingCount: u32,
    pub pBindings: *const VkDescriptorSetLayoutBinding,
}

// ── VkPushConstantRange ──────────────────────────────────────────────────────

#[repr(C)]
pub struct VkPushConstantRange {
    pub stageFlags: VkFlags,
    pub offset: u32,
    pub size: u32,
}

// ── VkPipelineLayoutCreateInfo ────────────────────────────────────────────────

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub setLayoutCount: u32,
    pub pSetLayouts: *const VkDescriptorSetLayout,
    pub pushConstantRangeCount: u32,
    pub pPushConstantRanges: *const VkPushConstantRange,
}

// ── VkPipelineShaderStageCreateInfo ──────────────────────────────────────────

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stage: VkFlags,         // VkShaderStageFlagBits
    pub module: VkShaderModule,
    pub pName: *const c_char,
    pub pSpecializationInfo: *const c_void,
}

// ── VkComputePipelineCreateInfo ───────────────────────────────────────────────

#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub basePipelineHandle: VkPipeline,
    pub basePipelineIndex: i32,
}

// ── VkDescriptorPoolSize ──────────────────────────────────────────────────────

#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub ty: u32,                // VkDescriptorType
    pub descriptorCount: u32,
}

// ── VkDescriptorPoolCreateInfo ────────────────────────────────────────────────

#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub maxSets: u32,
    pub poolSizeCount: u32,
    pub pPoolSizes: *const VkDescriptorPoolSize,
}

// ── VkDescriptorSetAllocateInfo ───────────────────────────────────────────────

#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub descriptorPool: VkDescriptorPool,
    pub descriptorSetCount: u32,
    pub pSetLayouts: *const VkDescriptorSetLayout,
}

// ── VkDescriptorBufferInfo ────────────────────────────────────────────────────

#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

// ── VkWriteDescriptorSet ──────────────────────────────────────────────────────

#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub sType: u32,
    pub pNext: *const c_void,
    pub dstSet: VkDescriptorSet,
    pub dstBinding: u32,
    pub dstArrayElement: u32,
    pub descriptorCount: u32,
    pub descriptorType: u32,
    pub pImageInfo: *const c_void,
    pub pBufferInfo: *const VkDescriptorBufferInfo,
    pub pTexelBufferView: *const u64,  // VkBufferView*
}

// ── VkCommandPoolCreateInfo ───────────────────────────────────────────────────

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueFamilyIndex: u32,
}

// ── VkCommandBufferAllocateInfo ───────────────────────────────────────────────

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub commandPool: VkCommandPool,
    pub level: u32,
    pub commandBufferCount: u32,
}

// ── VkCommandBufferBeginInfo ──────────────────────────────────────────────────

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub pInheritanceInfo: *const c_void,
}

// ── VkFenceCreateInfo ─────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkFenceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
}

// ── VkSubmitInfo ──────────────────────────────────────────────────────────────

#[repr(C)]
pub struct VkSubmitInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const VkSemaphore,
    pub pWaitDstStageMask: *const VkFlags,
    pub commandBufferCount: u32,
    pub pCommandBuffers: *const VkCommandBuffer,
    pub signalSemaphoreCount: u32,
    pub pSignalSemaphores: *const VkSemaphore,
}

// ── VkMakeApiVersion macro equivalent ────────────────────────────────────────

#[inline(always)]
pub const fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

// ── Vulkan C function declarations ────────────────────────────────────────────
//
// On Windows: links against vulkan-1.lib (via build.rs)
// On Linux:   links against libvulkan.so

#[cfg_attr(target_os = "windows", link(name = "vulkan-1"))]
#[cfg_attr(not(target_os = "windows"), link(name = "vulkan"))]
extern "system" {
    // ── Instance creation / destruction ──────────────────────────────────────
    pub fn vkCreateInstance(
        pCreateInfo: *const VkInstanceCreateInfo,
        pAllocator: *const c_void,
        pInstance: *mut VkInstance,
    ) -> VkResult;

    pub fn vkDestroyInstance(instance: VkInstance, pAllocator: *const c_void);

    // ── Physical device enumeration ───────────────────────────────────────────
    pub fn vkEnumeratePhysicalDevices(
        instance: VkInstance,
        pPhysicalDeviceCount: *mut u32,
        pPhysicalDevices: *mut VkPhysicalDevice,
    ) -> VkResult;

    pub fn vkGetPhysicalDeviceProperties(
        physicalDevice: VkPhysicalDevice,
        pProperties: *mut VkPhysicalDeviceProperties,
    );

    pub fn vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice: VkPhysicalDevice,
        pQueueFamilyPropertyCount: *mut u32,
        pQueueFamilyProperties: *mut VkQueueFamilyProperties,
    );

    pub fn vkGetPhysicalDeviceMemoryProperties(
        physicalDevice: VkPhysicalDevice,
        pMemoryProperties: *mut VkPhysicalDeviceMemoryProperties,
    );

    /// Vulkan 1.1+ — query device features with extension chain (pNext).
    pub fn vkGetPhysicalDeviceFeatures2(
        physicalDevice: VkPhysicalDevice,
        pFeatures: *mut VkPhysicalDeviceFeatures2,
    );

    // ── Logical device ────────────────────────────────────────────────────────
    pub fn vkCreateDevice(
        physicalDevice: VkPhysicalDevice,
        pCreateInfo: *const VkDeviceCreateInfo,
        pAllocator: *const c_void,
        pDevice: *mut VkDevice,
    ) -> VkResult;

    pub fn vkDestroyDevice(device: VkDevice, pAllocator: *const c_void);

    pub fn vkGetDeviceQueue(
        device: VkDevice,
        queueFamilyIndex: u32,
        queueIndex: u32,
        pQueue: *mut VkQueue,
    );

    // ── Memory ────────────────────────────────────────────────────────────────
    pub fn vkAllocateMemory(
        device: VkDevice,
        pAllocateInfo: *const VkMemoryAllocateInfo,
        pAllocator: *const c_void,
        pMemory: *mut VkDeviceMemory,
    ) -> VkResult;

    pub fn vkFreeMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        pAllocator: *const c_void,
    );

    pub fn vkMapMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        offset: VkDeviceSize,
        size: VkDeviceSize,
        flags: VkFlags,
        ppData: *mut *mut c_void,
    ) -> VkResult;

    pub fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory);

    // ── Buffers ───────────────────────────────────────────────────────────────
    pub fn vkCreateBuffer(
        device: VkDevice,
        pCreateInfo: *const VkBufferCreateInfo,
        pAllocator: *const c_void,
        pBuffer: *mut VkBuffer,
    ) -> VkResult;

    pub fn vkDestroyBuffer(
        device: VkDevice,
        buffer: VkBuffer,
        pAllocator: *const c_void,
    );

    pub fn vkGetBufferMemoryRequirements(
        device: VkDevice,
        buffer: VkBuffer,
        pMemoryRequirements: *mut VkMemoryRequirements,
    );

    pub fn vkBindBufferMemory(
        device: VkDevice,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        memoryOffset: VkDeviceSize,
    ) -> VkResult;

    // ── Shaders ───────────────────────────────────────────────────────────────
    pub fn vkCreateShaderModule(
        device: VkDevice,
        pCreateInfo: *const VkShaderModuleCreateInfo,
        pAllocator: *const c_void,
        pShaderModule: *mut VkShaderModule,
    ) -> VkResult;

    pub fn vkDestroyShaderModule(
        device: VkDevice,
        shaderModule: VkShaderModule,
        pAllocator: *const c_void,
    );

    // ── Descriptor set layout ─────────────────────────────────────────────────
    pub fn vkCreateDescriptorSetLayout(
        device: VkDevice,
        pCreateInfo: *const VkDescriptorSetLayoutCreateInfo,
        pAllocator: *const c_void,
        pSetLayout: *mut VkDescriptorSetLayout,
    ) -> VkResult;

    pub fn vkDestroyDescriptorSetLayout(
        device: VkDevice,
        descriptorSetLayout: VkDescriptorSetLayout,
        pAllocator: *const c_void,
    );

    // ── Pipeline layout ───────────────────────────────────────────────────────
    pub fn vkCreatePipelineLayout(
        device: VkDevice,
        pCreateInfo: *const VkPipelineLayoutCreateInfo,
        pAllocator: *const c_void,
        pPipelineLayout: *mut VkPipelineLayout,
    ) -> VkResult;

    pub fn vkDestroyPipelineLayout(
        device: VkDevice,
        pipelineLayout: VkPipelineLayout,
        pAllocator: *const c_void,
    );

    // ── Compute pipeline ──────────────────────────────────────────────────────
    pub fn vkCreateComputePipelines(
        device: VkDevice,
        pipelineCache: u64,     // VkPipelineCache (NULL)
        createInfoCount: u32,
        pCreateInfos: *const VkComputePipelineCreateInfo,
        pAllocator: *const c_void,
        pPipelines: *mut VkPipeline,
    ) -> VkResult;

    pub fn vkDestroyPipeline(
        device: VkDevice,
        pipeline: VkPipeline,
        pAllocator: *const c_void,
    );

    // ── Descriptor pool & sets ────────────────────────────────────────────────
    pub fn vkCreateDescriptorPool(
        device: VkDevice,
        pCreateInfo: *const VkDescriptorPoolCreateInfo,
        pAllocator: *const c_void,
        pDescriptorPool: *mut VkDescriptorPool,
    ) -> VkResult;

    pub fn vkDestroyDescriptorPool(
        device: VkDevice,
        descriptorPool: VkDescriptorPool,
        pAllocator: *const c_void,
    );

    pub fn vkAllocateDescriptorSets(
        device: VkDevice,
        pAllocateInfo: *const VkDescriptorSetAllocateInfo,
        pDescriptorSets: *mut VkDescriptorSet,
    ) -> VkResult;

    pub fn vkUpdateDescriptorSets(
        device: VkDevice,
        descriptorWriteCount: u32,
        pDescriptorWrites: *const VkWriteDescriptorSet,
        descriptorCopyCount: u32,
        pDescriptorCopies: *const c_void,
    );

    // ── Command pool & buffers ────────────────────────────────────────────────
    pub fn vkCreateCommandPool(
        device: VkDevice,
        pCreateInfo: *const VkCommandPoolCreateInfo,
        pAllocator: *const c_void,
        pCommandPool: *mut VkCommandPool,
    ) -> VkResult;

    pub fn vkDestroyCommandPool(
        device: VkDevice,
        commandPool: VkCommandPool,
        pAllocator: *const c_void,
    );

    pub fn vkAllocateCommandBuffers(
        device: VkDevice,
        pAllocateInfo: *const VkCommandBufferAllocateInfo,
        pCommandBuffers: *mut VkCommandBuffer,
    ) -> VkResult;

    pub fn vkBeginCommandBuffer(
        commandBuffer: VkCommandBuffer,
        pBeginInfo: *const VkCommandBufferBeginInfo,
    ) -> VkResult;

    pub fn vkEndCommandBuffer(commandBuffer: VkCommandBuffer) -> VkResult;

    pub fn vkCmdBindPipeline(
        commandBuffer: VkCommandBuffer,
        pipelineBindPoint: u32,
        pipeline: VkPipeline,
    );

    pub fn vkCmdBindDescriptorSets(
        commandBuffer: VkCommandBuffer,
        pipelineBindPoint: u32,
        layout: VkPipelineLayout,
        firstSet: u32,
        descriptorSetCount: u32,
        pDescriptorSets: *const VkDescriptorSet,
        dynamicOffsetCount: u32,
        pDynamicOffsets: *const u32,
    );

    pub fn vkCmdCopyBuffer(
        commandBuffer: VkCommandBuffer,
        srcBuffer: VkBuffer,
        dstBuffer: VkBuffer,
        regionCount: u32,
        pRegions: *const VkBufferCopy,
    );

    pub fn vkCmdPipelineBarrier(
        commandBuffer: VkCommandBuffer,
        srcStageMask: VkFlags,
        dstStageMask: VkFlags,
        dependencyFlags: VkFlags,
        memoryBarrierCount: u32,
        pMemoryBarriers: *const c_void,
        bufferMemoryBarrierCount: u32,
        pBufferMemoryBarriers: *const VkBufferMemoryBarrier,
        imageMemoryBarrierCount: u32,
        pImageMemoryBarriers: *const c_void,
    );

    pub fn vkCmdDispatch(
        commandBuffer: VkCommandBuffer,
        groupCountX: u32,
        groupCountY: u32,
        groupCountZ: u32,
    );

    pub fn vkCmdDispatchIndirect(
        commandBuffer: VkCommandBuffer,
        buffer: VkBuffer,
        offset: VkDeviceSize,
    );

    pub fn vkCmdFillBuffer(
        commandBuffer: VkCommandBuffer,
        dstBuffer: VkBuffer,
        dstOffset: VkDeviceSize,
        size: VkDeviceSize,
        data: u32,
    );

    pub fn vkCmdPushConstants(
        commandBuffer: VkCommandBuffer,
        layout: VkPipelineLayout,
        stageFlags: VkFlags,
        offset: u32,
        size: u32,
        pValues: *const c_void,
    );

    // ── Queue submission ──────────────────────────────────────────────────────
    pub fn vkQueueSubmit(
        queue: VkQueue,
        submitCount: u32,
        pSubmits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult;

    // ── Fences ────────────────────────────────────────────────────────────────
    pub fn vkCreateFence(
        device: VkDevice,
        pCreateInfo: *const VkFenceCreateInfo,
        pAllocator: *const c_void,
        pFence: *mut VkFence,
    ) -> VkResult;

    pub fn vkDestroyFence(
        device: VkDevice,
        fence: VkFence,
        pAllocator: *const c_void,
    );

    pub fn vkWaitForFences(
        device: VkDevice,
        fenceCount: u32,
        pFences: *const VkFence,
        waitAll: VkBool32,
        timeout: u64,
    ) -> VkResult;

    pub fn vkResetFences(
        device: VkDevice,
        fenceCount: u32,
        pFences: *const VkFence,
    ) -> VkResult;

    pub fn vkGetFenceStatus(
        device: VkDevice,
        fence: VkFence,
    ) -> VkResult;

    pub fn vkResetCommandPool(
        device: VkDevice,
        commandPool: VkCommandPool,
        flags: VkFlags,
    ) -> VkResult;

    pub fn vkResetDescriptorPool(
        device: VkDevice,
        descriptorPool: VkDescriptorPool,
        flags: VkFlags,
    ) -> VkResult;

    // ── Semaphores (Vulkan 1.2 timeline semaphore support) ─────────────────
    pub fn vkCreateSemaphore(
        device: VkDevice,
        pCreateInfo: *const c_void,
        pAllocator: *const c_void,
        pSemaphore: *mut VkSemaphore,
    ) -> VkResult;

    pub fn vkDestroySemaphore(
        device: VkDevice,
        semaphore: VkSemaphore,
        pAllocator: *const c_void,
    );

    pub fn vkSignalSemaphore(
        device: VkDevice,
        pSignalInfo: *const c_void,
    ) -> VkResult;

    pub fn vkWaitSemaphores(
        device: VkDevice,
        pWaitInfo: *const c_void,
        timeout: u64,
    ) -> VkResult;

    pub fn vkGetSemaphoreCounterValue(
        device: VkDevice,
        semaphore: VkSemaphore,
        pValue: *mut u64,
    ) -> VkResult;
}

// ── Helper: check VkResult and convert to Err ────────────────────────────────

#[inline]
pub fn vk_check(result: VkResult) -> Result<(), VkResult> {
    if result == VK_SUCCESS { Ok(()) } else { Err(result) }
}
