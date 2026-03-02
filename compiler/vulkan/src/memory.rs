//! Buffer allocation, memory type selection, and host ↔ device transfers.

use std::ffi::c_void;

use crate::device::{VulkanCompute, VulkanError};
use crate::vk_sys::*;

impl VulkanCompute {
    /// Find a memory type index satisfying the type filter and required property flags.
    pub fn find_memory_type_index(
        &self,
        type_filter: u32,
        required_flags: VkFlags,
    ) -> Result<u32, VulkanError> {
        let count = self.memory_properties.memoryTypeCount as usize;
        for i in 0..count {
            let has_type  = (type_filter & (1 << i)) != 0;
            let has_flags = self.memory_properties.memoryTypes[i].propertyFlags & required_flags == required_flags;
            if has_type && has_flags {
                return Ok(i as u32);
            }
        }
        Err(VulkanError::NoMemoryType)
    }

    /// Create a buffer with dedicated device memory.
    ///
    /// Returns `(buffer, device_memory)`. Caller is responsible for cleanup.
    pub fn create_buffer(
        &self,
        size: VkDeviceSize,
        usage: VkFlags,
        memory_flags: VkFlags,
    ) -> Result<(VkBuffer, VkDeviceMemory), VulkanError> {
        let buffer_info = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            size,
            usage,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: std::ptr::null(),
        };

        let mut buffer: VkBuffer = VK_NULL_HANDLE;
        let result = unsafe {
            vkCreateBuffer(self.device, &buffer_info, std::ptr::null(), &mut buffer)
        };
        vk_check(result).map_err(VulkanError::Vk)?;

        let mut mem_req: VkMemoryRequirements = unsafe { std::mem::zeroed() };
        unsafe { vkGetBufferMemoryRequirements(self.device, buffer, &mut mem_req); }

        let memory_type_index =
            self.find_memory_type_index(mem_req.memoryTypeBits, memory_flags)?;

        let alloc_info = VkMemoryAllocateInfo {
            sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            allocationSize: mem_req.size,
            memoryTypeIndex: memory_type_index,
        };

        let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
        let result = unsafe {
            vkAllocateMemory(self.device, &alloc_info, std::ptr::null(), &mut memory)
        };
        vk_check(result).map_err(VulkanError::Vk)?;

        let result = unsafe { vkBindBufferMemory(self.device, buffer, memory, 0) };
        vk_check(result).map_err(VulkanError::Vk)?;

        Ok((buffer, memory))
    }

    /// Upload f32 data to a HOST_VISIBLE device memory allocation.
    pub fn upload_f32(
        &self,
        memory: VkDeviceMemory,
        data: &[f32],
    ) -> Result<(), VulkanError> {
        let size = (data.len() * std::mem::size_of::<f32>()) as VkDeviceSize;
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let result = vkMapMemory(self.device, memory, 0, size, 0, &mut ptr);
            vk_check(result).map_err(|_| VulkanError::MapFailed)?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr as *mut u8,
                size as usize,
            );
            vkUnmapMemory(self.device, memory);
        }
        Ok(())
    }

    /// Download f32 data from a HOST_VISIBLE device memory allocation.
    pub fn download_f32(
        &self,
        memory: VkDeviceMemory,
        count: usize,
    ) -> Result<Vec<f32>, VulkanError> {
        let size = (count * std::mem::size_of::<f32>()) as VkDeviceSize;
        let mut result_vec = vec![0.0f32; count];
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let result = vkMapMemory(self.device, memory, 0, size, 0, &mut ptr);
            vk_check(result).map_err(|_| VulkanError::MapFailed)?;
            std::ptr::copy_nonoverlapping(
                ptr as *const u8,
                result_vec.as_mut_ptr() as *mut u8,
                size as usize,
            );
            vkUnmapMemory(self.device, memory);
        }
        Ok(result_vec)
    }
}
