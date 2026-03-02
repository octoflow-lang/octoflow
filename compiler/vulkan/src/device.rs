//! Vulkan instance creation, physical device selection, and logical device setup.

use std::ffi::{c_void, CStr};
use std::fmt;

use crate::vk_sys::*;

/// Error type for Vulkan operations.
#[derive(Debug)]
pub enum VulkanError {
    /// Vulkan API returned an error code.
    Vk(VkResult),
    /// No Vulkan-capable GPU found.
    NoGpu,
    /// No compute queue family found.
    NoComputeQueue,
    /// No suitable memory type found.
    NoMemoryType,
    /// Buffer mapping failed.
    MapFailed,
    /// General error with description.
    Other(String),
}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VulkanError::Vk(code) => {
                let name = vk_result_name(*code);
                match *code {
                    VK_ERROR_DEVICE_LOST => write!(f, "Vulkan error: {} ({}) — GPU device lost, restart may be required", name, code),
                    VK_ERROR_OUT_OF_DEVICE_MEMORY => write!(f, "Vulkan error: {} ({}) — GPU out of memory, reduce allocation size or use --gpu-max-mb", name, code),
                    VK_ERROR_OUT_OF_HOST_MEMORY => write!(f, "Vulkan error: {} ({}) — host out of memory", name, code),
                    _ => write!(f, "Vulkan error: {} ({})", name, code),
                }
            }
            VulkanError::NoGpu        => write!(f, "No Vulkan-capable GPU found"),
            VulkanError::NoComputeQueue => write!(f, "No compute queue family found"),
            VulkanError::NoMemoryType => write!(f, "No suitable memory type"),
            VulkanError::MapFailed    => write!(f, "Failed to map device memory"),
            VulkanError::Other(msg)   => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for VulkanError {}

impl From<VkResult> for VulkanError {
    fn from(result: VkResult) -> Self {
        VulkanError::Vk(result)
    }
}

/// Manages a Vulkan instance, physical device, logical device, and compute queue.
pub struct VulkanCompute {
    pub(crate) instance: VkInstance,
    pub(crate) physical_device: VkPhysicalDevice,
    pub(crate) device: VkDevice,
    pub(crate) compute_queue: VkQueue,
    pub(crate) queue_family_index: u32,
    pub(crate) memory_properties: VkPhysicalDeviceMemoryProperties,
    /// True if the GPU supports f16 arithmetic (shaderFloat16) and 16-bit storage buffers.
    pub supports_f16: bool,
    /// True if the GPU supports 64-bit integer arithmetic (shaderInt64).
    pub supports_int64: bool,
    /// Mutex for thread-safe vkQueueSubmit access.
    pub(crate) queue_mutex: std::sync::Mutex<()>,
}

impl VulkanCompute {
    /// Thread-safe queue submission. Holds queue_mutex during vkQueueSubmit.
    pub(crate) unsafe fn queue_submit(
        &self,
        submit_count: u32,
        submits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> Result<(), VulkanError> {
        let _lock = self.queue_mutex.lock().unwrap();
        vk_check(vkQueueSubmit(self.compute_queue, submit_count, submits, fence))
            .map_err(VulkanError::Vk)
    }

    /// Initialize Vulkan: create instance, pick GPU, create device.
    pub fn new() -> Result<Self, VulkanError> {
        let app_name = b"OctoFlow\0";
        let engine_name = b"OctoFlow\0";

        let app_info = VkApplicationInfo {
            sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: std::ptr::null(),
            pApplicationName: app_name.as_ptr() as *const _,
            applicationVersion: vk_make_api_version(0, 0, 1, 0),
            pEngineName: engine_name.as_ptr() as *const _,
            engineVersion: vk_make_api_version(0, 0, 1, 0),
            apiVersion: vk_make_api_version(0, 1, 1, 0),
        };

        let create_info = VkInstanceCreateInfo {
            sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            pApplicationInfo: &app_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: std::ptr::null(),
            enabledExtensionCount: 0,
            ppEnabledExtensionNames: std::ptr::null(),
        };

        // Create Vulkan instance
        let mut instance: VkInstance = std::ptr::null_mut();
        let result = unsafe {
            vkCreateInstance(&create_info, std::ptr::null(), &mut instance)
        };
        if result != VK_SUCCESS {
            return Err(VulkanError::NoGpu);
        }

        // Enumerate physical devices
        let mut count = 0u32;
        unsafe { vkEnumeratePhysicalDevices(instance, &mut count, std::ptr::null_mut()); }
        if count == 0 {
            unsafe { vkDestroyInstance(instance, std::ptr::null()); }
            return Err(VulkanError::NoGpu);
        }
        let mut phys_devs: Vec<VkPhysicalDevice> = vec![std::ptr::null_mut(); count as usize];
        unsafe { vkEnumeratePhysicalDevices(instance, &mut count, phys_devs.as_mut_ptr()); }

        // Prefer discrete GPU, fall back to first device
        let mut chosen = phys_devs[0];
        for &pd in &phys_devs {
            let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
            unsafe { vkGetPhysicalDeviceProperties(pd, &mut props); }
            if props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU {
                chosen = pd;
                break;
            }
        }
        let physical_device = chosen;

        // Find compute queue family
        let mut fam_count = 0u32;
        unsafe { vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut fam_count, std::ptr::null_mut()); }
        let families: Vec<VkQueueFamilyProperties> = unsafe {
            let mut v: Vec<VkQueueFamilyProperties> = Vec::with_capacity(fam_count as usize);
            // Zero-initialize before Vulkan fills them (avoids UB from uninitialized read)
            std::ptr::write_bytes(v.as_mut_ptr(), 0, fam_count as usize);
            v.set_len(fam_count as usize);
            vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut fam_count, v.as_mut_ptr());
            v
        };
        let _ = families.len(); // suppress warning

        let queue_family_index = families.iter().enumerate()
            .find(|(_, f)| f.queueFlags & VK_QUEUE_COMPUTE_BIT != 0)
            .map(|(i, _)| i as u32)
            .ok_or_else(|| {
                unsafe { vkDestroyInstance(instance, std::ptr::null()); }
                VulkanError::NoComputeQueue
            })?;

        // Query f16 capability via Vulkan 1.1 feature chain
        let mut storage_16bit: VkPhysicalDevice16BitStorageFeatures = unsafe { std::mem::zeroed() };
        storage_16bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

        let mut float16_int8: VkPhysicalDeviceShaderFloat16Int8Features = unsafe { std::mem::zeroed() };
        float16_int8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        float16_int8.pNext = &mut storage_16bit as *mut _ as *mut c_void;

        let mut features2: VkPhysicalDeviceFeatures2 = unsafe { std::mem::zeroed() };
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &mut float16_int8 as *mut _ as *mut c_void;

        unsafe { vkGetPhysicalDeviceFeatures2(physical_device, &mut features2); }

        let supports_f16 = float16_int8.shaderFloat16 != 0
            && storage_16bit.storageBuffer16BitAccess != 0;
        let supports_int64 = features2.features.shaderInt64 != 0;

        // Create logical device — enable f16 + int64 features if GPU supports them
        let priority = 1.0f32;
        let queue_info = VkDeviceQueueCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            queueFamilyIndex: queue_family_index,
            queueCount: 1,
            pQueuePriorities: &priority,
        };

        // Build pNext chain for device features if f16 supported
        let mut enable_storage_16bit: VkPhysicalDevice16BitStorageFeatures = unsafe { std::mem::zeroed() };
        enable_storage_16bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        enable_storage_16bit.storageBuffer16BitAccess = if supports_f16 { 1 } else { 0 };

        let mut enable_float16: VkPhysicalDeviceShaderFloat16Int8Features = unsafe { std::mem::zeroed() };
        enable_float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        enable_float16.shaderFloat16 = if supports_f16 { 1 } else { 0 };
        enable_float16.pNext = &mut enable_storage_16bit as *mut _ as *mut c_void;

        let device_pnext: *const c_void = if supports_f16 {
            &mut enable_float16 as *mut _ as *const c_void
        } else {
            std::ptr::null()
        };

        // Enable core device features (shaderInt64)
        let mut enabled_features: VkPhysicalDeviceFeatures = unsafe { std::mem::zeroed() };
        if supports_int64 {
            enabled_features.shaderInt64 = 1;
        }

        let device_info = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: device_pnext,
            flags: 0,
            queueCreateInfoCount: 1,
            pQueueCreateInfos: &queue_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: std::ptr::null(),
            enabledExtensionCount: 0,
            ppEnabledExtensionNames: std::ptr::null(),
            pEnabledFeatures: &enabled_features as *const _ as *const c_void,
        };

        let mut device: VkDevice = std::ptr::null_mut();
        let result = unsafe {
            vkCreateDevice(physical_device, &device_info, std::ptr::null(), &mut device)
        };
        if result != VK_SUCCESS {
            unsafe { vkDestroyInstance(instance, std::ptr::null()); }
            return Err(VulkanError::Vk(result));
        }

        // Get compute queue handle
        let mut compute_queue: VkQueue = std::ptr::null_mut();
        unsafe { vkGetDeviceQueue(device, queue_family_index, 0, &mut compute_queue); }

        // Get memory properties
        let mut memory_properties: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
        unsafe { vkGetPhysicalDeviceMemoryProperties(physical_device, &mut memory_properties); }

        Ok(Self {
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family_index,
            memory_properties,
            supports_f16,
            supports_int64,
            queue_mutex: std::sync::Mutex::new(()),
        })
    }

    /// Return the GPU device name.
    pub fn device_name(&self) -> String {
        let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe { vkGetPhysicalDeviceProperties(self.physical_device, &mut props); }
        let name = unsafe { CStr::from_ptr(props.deviceName.as_ptr()) };
        name.to_string_lossy().into_owned()
    }

    /// Return GPU device properties for gpu_info() builtin.
    pub fn gpu_properties(&self) -> GpuInfo {
        let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe { vkGetPhysicalDeviceProperties(self.physical_device, &mut props); }
        let name = unsafe { CStr::from_ptr(props.deviceName.as_ptr()) };
        let device_type = match props.deviceType {
            0 => "other",
            1 => "integrated",
            2 => "discrete",
            3 => "virtual",
            4 => "cpu",
            _ => "unknown",
        };
        GpuInfo {
            name: name.to_string_lossy().into_owned(),
            device_type: device_type.to_string(),
            api_version_major: (props.apiVersion >> 22) & 0x3FF,
            api_version_minor: (props.apiVersion >> 12) & 0x3FF,
            vendor_id: props.vendorID,
            device_id: props.deviceID,
            max_compute_workgroup_size_x: props.limits.maxComputeWorkGroupSize[0],
            max_compute_workgroup_size_y: props.limits.maxComputeWorkGroupSize[1],
            max_compute_workgroup_size_z: props.limits.maxComputeWorkGroupSize[2],
            max_compute_workgroup_invocations: props.limits.maxComputeWorkGroupInvocations,
            max_storage_buffer_range: props.limits.maxStorageBufferRange,
            max_compute_shared_memory: props.limits.maxComputeSharedMemorySize,
            supports_f16: self.supports_f16,
            supports_int64: self.supports_int64,
        }
    }
}

/// GPU device information returned by gpu_info().
pub struct GpuInfo {
    pub name: String,
    pub device_type: String,
    pub api_version_major: u32,
    pub api_version_minor: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroup_invocations: u32,
    pub max_storage_buffer_range: u32,
    pub max_compute_shared_memory: u32,
    pub supports_f16: bool,
    pub supports_int64: bool,
}

impl Drop for VulkanCompute {
    fn drop(&mut self) {
        unsafe {
            vkDestroyDevice(self.device, std::ptr::null());
            vkDestroyInstance(self.instance, std::ptr::null());
        }
    }
}

// Suppress unused import warning on c_void used in vk_sys
fn _unused(_: *const c_void) {}
