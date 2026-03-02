//! GPU VM — general-purpose virtual machine with SSBO-backed memory.
//!
//! Each VM instance has a register file (32 registers x reg_size floats) in DEVICE_LOCAL VRAM.
//! Metrics (binding 1) and Control (binding 3) are HOST_VISIBLE for zero-copy CPU polling.
//! Registers and Globals stay DEVICE_LOCAL for GPU compute performance.
//! GPU broadcasts status via Metrics; CPU provisions via Control writes.

use std::ffi::c_void;

use crate::device::{VulkanCompute, VulkanError};
use crate::dispatch::{
    GpuBuffer, upload_buffer, upload_buffer_with_usage,
    upload_buffer_vram,
    acquire_buffer, release_buffer, acquire_command_pool, acquire_fence,
    spirv_hash,
};
use crate::vk_sys::*;

// ── VM pipeline cache ─────────────────────────────────────────────────────────
// Pipelines are expensive to create (~10-15ms each, requires GPU driver JIT).
// OctoUI uses only 3 unique .spv files but dispatches them 50-200x per frame.
// Cache key: (device_ptr, spirv_hash, num_bindings, pc_size_bytes).
// Cache lives for process lifetime — pipelines are never explicitly destroyed.
// Safe: device outlives thread (GPU context is cleaned up after all programs).

#[derive(Clone, Copy)]
struct VmCachedPipeline {
    shader_module: VkShaderModule,
    ds_layout: VkDescriptorSetLayout,
    pipeline_layout: VkPipelineLayout,
    pipeline: VkPipeline,
}

thread_local! {
    static VM_PIPELINE_CACHE: std::cell::RefCell<
        std::collections::HashMap<(u64, u64, u32, u32), VmCachedPipeline>
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

// ── Per-frame Vulkan resource pool ─────────────────────────────────────────
// VkCommandPool + VkFence + VkDescriptorPool created/destroyed every frame
// adds ~2-5ms overhead even with pipeline caching. Fix: pool these objects
// and reset them O(1) each frame instead of destroy+create.
//
// vkResetCommandPool  → command buffer returns to initial state (no realloc)
// vkResetFences       → fence returns to unsignaled
// vkResetDescriptorPool → allocation watermark reset (all sets freed instantly)
//
// First frame: allocates. All subsequent frames: O(1) resets only.

struct VmFrameResources {
    device: VkDevice,
    command_pool: VkCommandPool,
    command_buffer: VkCommandBuffer,
    fence: VkFence,
    desc_pool: VkDescriptorPool,
    desc_pool_max_sets: u32,
    desc_pool_max_descriptors: u32,
}

impl VmFrameResources {
    /// Destroy all owned Vulkan objects. Called when pool entry is evicted.
    unsafe fn destroy(&self) {
        vkDestroyDescriptorPool(self.device, self.desc_pool, std::ptr::null());
        vkDestroyFence(self.device, self.fence, std::ptr::null());
        vkDestroyCommandPool(self.device, self.command_pool, std::ptr::null());
    }
}

thread_local! {
    // Idle frame resources per device. Pop to acquire, push to release.
    static VM_FRAME_POOL: std::cell::RefCell<
        std::collections::HashMap<u64, Vec<VmFrameResources>>
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Acquire reusable frame resources. Resets existing or allocates fresh.
fn acquire_frame_resources(
    gpu: &VulkanCompute,
    num_ops: u32,
    num_bindings: u32,
) -> Result<VmFrameResources, VulkanError> {
    let device = gpu.device;
    let device_key = device as usize as u64;
    let needed_sets = num_ops;
    let needed_descriptors = num_ops * num_bindings;

    // Try to reuse cached resources
    let maybe_cached = VM_FRAME_POOL.with(|pool| {
        pool.borrow_mut().entry(device_key).or_default().pop()
    });

    if let Some(mut res) = maybe_cached {
        // Reset command pool — command buffer returns to initial state
        vk_check(unsafe { vkResetCommandPool(device, res.command_pool, 0) })
            .map_err(VulkanError::Vk)?;

        // Reset fence to unsignaled (was signaled by previous vm_execute)
        vk_check(unsafe { vkResetFences(device, 1, &res.fence) })
            .map_err(VulkanError::Vk)?;

        // Descriptor pool: resize if needed, else O(1) reset
        if needed_sets > res.desc_pool_max_sets
            || needed_descriptors > res.desc_pool_max_descriptors
        {
            unsafe { vkDestroyDescriptorPool(device, res.desc_pool, std::ptr::null()); }
            let new_max_sets = (needed_sets * 2).max(64);
            let new_max_desc = (needed_descriptors * 2).max(256);
            let pool_size = VkDescriptorPoolSize {
                ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: new_max_desc,
            };
            let dp_info = VkDescriptorPoolCreateInfo {
                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                maxSets: new_max_sets,
                poolSizeCount: 1,
                pPoolSizes: &pool_size,
            };
            let mut new_pool: VkDescriptorPool = VK_NULL_HANDLE;
            vk_check(unsafe {
                vkCreateDescriptorPool(device, &dp_info, std::ptr::null(), &mut new_pool)
            }).map_err(VulkanError::Vk)?;
            res.desc_pool = new_pool;
            res.desc_pool_max_sets = new_max_sets;
            res.desc_pool_max_descriptors = new_max_desc;
        } else {
            // Fast reset: O(1) — allocation watermark reset, all sets freed
            vk_check(unsafe { vkResetDescriptorPool(device, res.desc_pool, 0) })
                .map_err(VulkanError::Vk)?;
        }
        return Ok(res);
    }

    // No cached resources — allocate fresh with headroom
    let pool_info = VkCommandPoolCreateInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        queueFamilyIndex: gpu.queue_family_index,
    };
    let mut cmd_pool: VkCommandPool = VK_NULL_HANDLE;
    vk_check(unsafe {
        vkCreateCommandPool(device, &pool_info, std::ptr::null(), &mut cmd_pool)
    }).map_err(VulkanError::Vk)?;

    let cmd_alloc = VkCommandBufferAllocateInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        pNext: std::ptr::null(),
        commandPool: cmd_pool,
        level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount: 1,
    };
    let mut cmd: VkCommandBuffer = std::ptr::null_mut();
    vk_check(unsafe { vkAllocateCommandBuffers(device, &cmd_alloc, &mut cmd) })
        .map_err(VulkanError::Vk)?;

    let fence_info = VkFenceCreateInfo {
        sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
    };
    let mut fence: VkFence = VK_NULL_HANDLE;
    vk_check(unsafe { vkCreateFence(device, &fence_info, std::ptr::null(), &mut fence) })
        .map_err(VulkanError::Vk)?;

    let max_sets = (needed_sets * 2).max(64);
    let max_descriptors = (needed_descriptors * 2).max(256);
    let pool_size = VkDescriptorPoolSize {
        ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount: max_descriptors,
    };
    let dp_info = VkDescriptorPoolCreateInfo {
        sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        maxSets: max_sets,
        poolSizeCount: 1,
        pPoolSizes: &pool_size,
    };
    let mut desc_pool: VkDescriptorPool = VK_NULL_HANDLE;
    vk_check(unsafe {
        vkCreateDescriptorPool(device, &dp_info, std::ptr::null(), &mut desc_pool)
    }).map_err(VulkanError::Vk)?;

    Ok(VmFrameResources {
        device,
        command_pool: cmd_pool,
        command_buffer: cmd,
        fence,
        desc_pool,
        desc_pool_max_sets: max_sets,
        desc_pool_max_descriptors: max_descriptors,
    })
}

/// Return frame resources to pool for reuse. Destroys if pool is full.
fn release_frame_resources(res: VmFrameResources) {
    let device_key = res.device as usize as u64;
    VM_FRAME_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let entries = pool.entry(device_key).or_default();
        if entries.len() < 4 {
            entries.push(res);
        } else {
            // Pool full — destroy to prevent unbounded growth
            unsafe { res.destroy(); }
        }
    });
}

/// GPU VM handle. Owns SSBOs for registers, metrics, globals, control.
/// Metrics + Control are HOST_VISIBLE (zero-copy CPU polling).
/// Registers + Globals are DEVICE_LOCAL (GPU compute performance).
pub struct VmHandle {
    pub n_instances: u32,
    pub reg_size: u32,
    pub n_registers: u32,
    pub registers: GpuBuffer,   // Binding 0: DEVICE_LOCAL, n_instances * 32 * reg_size floats
    pub metrics: GpuBuffer,     // Binding 1: HOST_VISIBLE, n_instances * METRICS_STRIDE floats
    pub globals: GpuBuffer,     // Binding 2: DEVICE_LOCAL, user-specified size (shared mutable state)
    pub control: GpuBuffer,     // Binding 3: HOST_VISIBLE, n_instances * CONTROL_STRIDE floats (regulator + indirect dispatch)
    pub heap: Option<GpuBuffer>, // Binding 4: DEVICE_LOCAL, large immutable data (added separately)
}

const METRICS_STRIDE: u32 = 8;
const CONTROL_STRIDE: u32 = 8;

/// Create a new GPU VM with n_instances, each having 32 registers of reg_size floats.
pub fn vm_create(
    gpu: &VulkanCompute,
    n_instances: u32,
    reg_size: u32,
    globals_size: u32,
) -> Result<VmHandle, VulkanError> {
    // Ensure all buffer sizes are at least 1 float (4 bytes) to avoid
    // 0-byte Vulkan allocations which are undefined on some drivers.
    let total_reg_floats = ((n_instances * 32 * reg_size) as usize).max(1);
    let total_metrics_floats = ((n_instances * METRICS_STRIDE) as usize).max(1);
    let total_globals_floats = (globals_size as usize).max(1);
    let total_control_floats = ((n_instances * CONTROL_STRIDE) as usize).max(1);

    let reg_data = vec![0.0f32; total_reg_floats];
    let metrics_data = vec![0.0f32; total_metrics_floats];
    let globals_data = vec![0.0f32; total_globals_floats];
    let control_data = vec![0.0f32; total_control_floats];

    let registers = upload_buffer_vram(gpu, &reg_data)?;
    // Metrics: HOST_VISIBLE for zero-copy CPU polling (GPU broadcasts status here)
    let metrics = upload_buffer(gpu, &metrics_data)?;
    let globals = upload_buffer_vram(gpu, &globals_data)?;
    // Control: HOST_VISIBLE + INDIRECT_BUFFER_BIT (CPU writes dispatch counts, GPU reads via vkCmdDispatchIndirect)
    let control = upload_buffer_with_usage(gpu, &control_data, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)?;

    Ok(VmHandle {
        n_instances,
        reg_size,
        n_registers: 32,
        registers,
        metrics,
        globals,
        control,
        heap: None,
    })
}

/// Write data to a specific register of a specific VM instance via staging DMA.
pub fn vm_write_reg(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    instance: u32,
    reg: u32,
    data: &[f32],
) -> Result<(), VulkanError> {
    if instance >= vm.n_instances {
        return Err(VulkanError::Other(format!(
            "vm_write_reg: instance {} >= n_instances {}", instance, vm.n_instances
        )));
    }
    if reg >= vm.n_registers {
        return Err(VulkanError::Other(format!(
            "vm_write_reg: register {} >= n_registers {}", reg, vm.n_registers
        )));
    }
    let offset_floats = (instance * vm.n_registers * vm.reg_size + reg * vm.reg_size) as usize;
    let byte_offset = (offset_floats * 4) as u64;
    let write_len = data.len().min(vm.reg_size as usize);

    vm_write_region(gpu, &vm.registers, byte_offset, &data[..write_len])
}

/// Write data to the globals SSBO at a float offset.
pub fn vm_write_globals(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    offset_floats: u32,
    data: &[f32],
) -> Result<(), VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    vm_write_region(gpu, &vm.globals, byte_offset, data)
}

/// Write data to the metrics SSBO at a float offset.
pub fn vm_write_metrics(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    offset_floats: u32,
    data: &[f32],
) -> Result<(), VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    vm_write_region(gpu, &vm.metrics, byte_offset, data)
}

/// Read data from the metrics SSBO at a float offset.
pub fn vm_read_metrics(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    offset_floats: u32,
    count: u32,
) -> Result<Vec<f32>, VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    vm_read_region(gpu, &vm.metrics, byte_offset, count as usize)
}

/// Write data to the control SSBO at a float offset.
pub fn vm_write_control(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    offset_floats: u32,
    data: &[f32],
) -> Result<(), VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    vm_write_region(gpu, &vm.control, byte_offset, data)
}

/// Read data from the control SSBO at a float offset.
pub fn vm_read_control(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    offset_floats: u32,
    count: u32,
) -> Result<Vec<f32>, VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    vm_read_region(gpu, &vm.control, byte_offset, count as usize)
}

/// Read data from the globals SSBO at a float offset.
pub fn vm_read_globals(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    offset_floats: u32,
    count: u32,
) -> Result<Vec<f32>, VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    vm_read_region(gpu, &vm.globals, byte_offset, count as usize)
}

/// Download all 3 UI framebuffer channels (R, G, B) in ONE staging submission.
///
/// Globals layout for OctoUI: [R_0..R_n, G_0..G_n, B_0..B_n] (3*total floats, offset 0).
/// Returns `(r, g, b)` each with `total` floats. 1 fence wait vs 3 with vm_read_globals×3.
pub fn vm_read_fb(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    total: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), VulkanError> {
    if total == 0 {
        return Ok((vec![], vec![], vec![]));
    }
    let count = total * 3;
    let byte_len = (count * 4) as u64;

    // Fast path: HOST_VISIBLE (integrated GPU, unified memory) — direct map, no staging
    if (vm.globals.mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 {
        let mut flat = vec![0.0f32; count];
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            vk_check(vkMapMemory(vm.globals.device, vm.globals.memory, 0, byte_len, 0, &mut ptr))
                .map_err(VulkanError::Vk)?;
            std::ptr::copy_nonoverlapping(
                ptr as *const u8, flat.as_mut_ptr() as *mut u8, byte_len as usize,
            );
            vkUnmapMemory(vm.globals.device, vm.globals.memory);
        }
        let r = flat[..total].to_vec();
        let g = flat[total..total * 2].to_vec();
        let b = flat[total * 2..].to_vec();
        return Ok((r, g, b));
    }

    // DEVICE_LOCAL: one staging buffer, one command buffer, one fence wait for all 3 channels
    let stg_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    let stg_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let (stg_buf, stg_memory) = acquire_buffer(gpu, byte_len, stg_usage, stg_mem)?;

    let command_pool = acquire_command_pool(gpu)?;
    let fence = acquire_fence(gpu)?;
    unsafe {
        let alloc_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cmd: VkCommandBuffer = std::ptr::null_mut();
        vk_check(vkAllocateCommandBuffers(gpu.device, &alloc_info, &mut cmd))
            .map_err(VulkanError::Vk)?;

        let begin_info = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: std::ptr::null(),
        };
        vk_check(vkBeginCommandBuffer(cmd, &begin_info)).map_err(VulkanError::Vk)?;

        // Single copy: globals[0..3*total] → staging (contiguous, one DMA)
        let region = VkBufferCopy { srcOffset: 0, dstOffset: 0, size: byte_len };
        vkCmdCopyBuffer(cmd, vm.globals.buffer, stg_buf, 1, &region);

        vk_check(vkEndCommandBuffer(cmd)).map_err(VulkanError::Vk)?;

        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1, pCommandBuffers: &cmd,
            signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
        };
        gpu.queue_submit(1, &submit, fence)?;
        vk_check(vkWaitForFences(gpu.device, 1, &fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }

    let flat = gpu.download_f32(stg_memory, count)?;
    release_buffer(gpu.device, byte_len, stg_mem, stg_buf, stg_memory);

    let r = flat[..total].to_vec();
    let g = flat[total..total * 2].to_vec();
    let b = flat[total * 2..].to_vec();
    Ok((r, g, b))
}

/// Pending framebuffer read state for async present (double-buffer pattern).
pub struct PendingFbRead {
    pub device: VkDevice,
    pub fence: VkFence,
    pub stg_buf: VkBuffer,
    pub stg_memory: VkDeviceMemory,
    pub byte_len: u64,
    pub stg_mem_flags: VkFlags,
    pub total: usize,
}

/// Submit GPU→staging copy for framebuffer without waiting.
/// Returns PendingFbRead to be completed later, or None if HOST_VISIBLE (already done).
pub fn vm_submit_fb_copy(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    total: usize,
) -> Result<(Option<PendingFbRead>, Option<(Vec<f32>, Vec<f32>, Vec<f32>)>), VulkanError> {
    if total == 0 {
        return Ok((None, Some((vec![], vec![], vec![]))));
    }
    let count = total * 3;
    let byte_len = (count * 4) as u64;

    // Fast path: HOST_VISIBLE — direct map, return immediately
    if (vm.globals.mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 {
        let mut flat = vec![0.0f32; count];
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            vk_check(vkMapMemory(vm.globals.device, vm.globals.memory, 0, byte_len, 0, &mut ptr))
                .map_err(VulkanError::Vk)?;
            std::ptr::copy_nonoverlapping(
                ptr as *const u8, flat.as_mut_ptr() as *mut u8, byte_len as usize,
            );
            vkUnmapMemory(vm.globals.device, vm.globals.memory);
        }
        let r = flat[..total].to_vec();
        let g = flat[total..total * 2].to_vec();
        let b = flat[total * 2..].to_vec();
        return Ok((None, Some((r, g, b))));
    }

    // DEVICE_LOCAL: submit copy, return pending handle
    let stg_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    let stg_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let (stg_buf, stg_memory) = acquire_buffer(gpu, byte_len, stg_usage, stg_mem)?;

    let command_pool = acquire_command_pool(gpu)?;
    let fence = acquire_fence(gpu)?;
    unsafe {
        let alloc_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cmd: VkCommandBuffer = std::ptr::null_mut();
        vk_check(vkAllocateCommandBuffers(gpu.device, &alloc_info, &mut cmd))
            .map_err(VulkanError::Vk)?;

        let begin_info = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: std::ptr::null(),
        };
        vk_check(vkBeginCommandBuffer(cmd, &begin_info)).map_err(VulkanError::Vk)?;

        let region = VkBufferCopy { srcOffset: 0, dstOffset: 0, size: byte_len };
        vkCmdCopyBuffer(cmd, vm.globals.buffer, stg_buf, 1, &region);

        vk_check(vkEndCommandBuffer(cmd)).map_err(VulkanError::Vk)?;

        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1, pCommandBuffers: &cmd,
            signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
        };
        gpu.queue_submit(1, &submit, fence)?;
    }

    Ok((Some(PendingFbRead {
        device: gpu.device,
        fence,
        stg_buf,
        stg_memory,
        byte_len,
        stg_mem_flags: stg_mem,
        total,
    }), None))
}

/// Complete a pending framebuffer read: wait for fence, download, release staging.
pub fn vm_finish_fb_read(
    gpu: &VulkanCompute,
    pending: PendingFbRead,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), VulkanError> {
    let count = pending.total * 3;
    unsafe {
        // Fast path: check if already done
        let status = vkGetFenceStatus(pending.device, pending.fence);
        if status != VK_SUCCESS {
            vk_check(vkWaitForFences(pending.device, 1, &pending.fence, 1, u64::MAX))
                .map_err(VulkanError::Vk)?;
        }
    }
    let flat = gpu.download_f32(pending.stg_memory, count)?;
    release_buffer(pending.device, pending.byte_len, pending.stg_mem_flags, pending.stg_buf, pending.stg_memory);

    let r = flat[..pending.total].to_vec();
    let g = flat[pending.total..pending.total * 2].to_vec();
    let b = flat[pending.total * 2..].to_vec();
    Ok((r, g, b))
}

/// A staged upload: CPU data is in a staging buffer, ready for vkCmdCopyBuffer.
pub struct PendingUpload {
    pub staging_buffer: VkBuffer,
    pub staging_memory: VkDeviceMemory,
    pub staging_mem_flags: VkFlags,
    pub dst_buffer: VkBuffer,
    pub offset: u64,
    pub byte_len: u64,
}

/// Stage data for a deferred DMA copy (CPU→staging memcpy only, no GPU submit).
/// Returns a PendingUpload to be consumed by vm_build_program.
/// For HOST_VISIBLE buffers, writes directly and returns None.
pub fn vm_stage_write(
    gpu: &VulkanCompute,
    dst: &GpuBuffer,
    byte_offset: u64,
    data: &[f32],
) -> Result<Option<PendingUpload>, VulkanError> {
    let byte_len = (data.len() * 4) as u64;
    if byte_len == 0 { return Ok(None); }

    // HOST_VISIBLE: direct write, no staging needed
    if (dst.mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            vk_check(vkMapMemory(dst.device, dst.memory, byte_offset, byte_len, 0, &mut ptr))
                .map_err(VulkanError::Vk)?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8, ptr as *mut u8, byte_len as usize,
            );
            vkUnmapMemory(dst.device, dst.memory);
        }
        return Ok(None);
    }

    // DEVICE_LOCAL: alloc staging + memcpy, defer DMA
    let stg_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    let stg_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let (stg_buf, stg_memory) = acquire_buffer(gpu, byte_len, stg_usage, stg_mem_flags)?;
    gpu.upload_f32(stg_memory, data)?;

    Ok(Some(PendingUpload {
        staging_buffer: stg_buf,
        staging_memory: stg_memory,
        staging_mem_flags: stg_mem_flags,
        dst_buffer: dst.buffer,
        offset: byte_offset,
        byte_len,
    }))
}

/// Copy data from an external VkBuffer into the globals SSBO (GPU-to-GPU, no CPU roundtrip).
/// Used to load weight matrices from gguf_load_tensor GPU arrays into the VM.
pub fn vm_gpu_to_globals(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    dst_offset_floats: u32,
    src_buffer: VkBuffer,
    src_count: u32,
) -> Result<(), VulkanError> {
    let dst_byte_offset = (dst_offset_floats as u64) * 4;
    let byte_len = (src_count as u64) * 4;
    if byte_len == 0 { return Ok(()); }

    let command_pool = acquire_command_pool(gpu)?;
    let fence = acquire_fence(gpu)?;
    unsafe {
        let alloc_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cmd: VkCommandBuffer = std::ptr::null_mut();
        vk_check(vkAllocateCommandBuffers(gpu.device, &alloc_info, &mut cmd))
            .map_err(VulkanError::Vk)?;

        let begin_info = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: std::ptr::null(),
        };
        vk_check(vkBeginCommandBuffer(cmd, &begin_info)).map_err(VulkanError::Vk)?;

        let region = VkBufferCopy { srcOffset: 0, dstOffset: dst_byte_offset, size: byte_len };
        vkCmdCopyBuffer(cmd, src_buffer, vm.globals.buffer, 1, &region);

        vk_check(vkEndCommandBuffer(cmd)).map_err(VulkanError::Vk)?;

        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1, pCommandBuffers: &cmd,
            signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
        };
        gpu.queue_submit(1, &submit, fence)?;
        vk_check(vkWaitForFences(gpu.device, 1, &fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }
    Ok(())
}

/// GPU→GPU copy between two VMs' globals buffers. No CPU roundtrip.
/// Copies `count` floats from src_vm.globals[src_offset..] to dst_vm.globals[dst_offset..].
pub fn vm_copy_globals(
    gpu: &VulkanCompute,
    src_vm: &VmHandle,
    src_offset_floats: u32,
    dst_vm: &VmHandle,
    dst_offset_floats: u32,
    count: u32,
) -> Result<(), VulkanError> {
    let src_byte_offset = (src_offset_floats as u64) * 4;
    let dst_byte_offset = (dst_offset_floats as u64) * 4;
    let byte_len = (count as u64) * 4;
    if byte_len == 0 { return Ok(()); }

    let command_pool = acquire_command_pool(gpu)?;
    let fence = acquire_fence(gpu)?;
    unsafe {
        let alloc_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cmd: VkCommandBuffer = std::ptr::null_mut();
        vk_check(vkAllocateCommandBuffers(gpu.device, &alloc_info, &mut cmd))
            .map_err(VulkanError::Vk)?;

        let begin_info = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: std::ptr::null(),
        };
        vk_check(vkBeginCommandBuffer(cmd, &begin_info)).map_err(VulkanError::Vk)?;

        let region = VkBufferCopy { srcOffset: src_byte_offset, dstOffset: dst_byte_offset, size: byte_len };
        vkCmdCopyBuffer(cmd, src_vm.globals.buffer, dst_vm.globals.buffer, 1, &region);

        vk_check(vkEndCommandBuffer(cmd)).map_err(VulkanError::Vk)?;

        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1, pCommandBuffers: &cmd,
            signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
        };
        gpu.queue_submit(1, &submit, fence)?;
        vk_check(vkWaitForFences(gpu.device, 1, &fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }
    Ok(())
}

/// Set the VM's heap buffer (binding 4) with the given data.
/// The heap is immutable after this call — used for quantized weights, embeddings, etc.
pub fn vm_set_heap(
    gpu: &VulkanCompute,
    vm: &mut VmHandle,
    data: &[f32],
) -> Result<(), VulkanError> {
    let buf = upload_buffer_vram(gpu, data)?;
    vm.heap = Some(buf);
    Ok(())
}

/// Read data from a specific register of a specific VM instance.
pub fn vm_read_reg(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    instance: u32,
    reg: u32,
    count: u32,
) -> Result<Vec<f32>, VulkanError> {
    if instance >= vm.n_instances || reg >= vm.n_registers {
        return Err(VulkanError::Other("vm_read_reg: out of bounds".into()));
    }
    let offset_floats = (instance * vm.n_registers * vm.reg_size + reg * vm.reg_size) as usize;
    let byte_offset = (offset_floats * 4) as u64;
    let read_len = (count as usize).min(vm.reg_size as usize);

    vm_read_region(gpu, &vm.registers, byte_offset, read_len)
}

/// Write a subregion of a buffer. HOST_VISIBLE: direct memcpy. DEVICE_LOCAL: staging DMA.
fn vm_write_region(
    gpu: &VulkanCompute,
    dst: &GpuBuffer,
    byte_offset: u64,
    data: &[f32],
) -> Result<(), VulkanError> {
    let byte_len = (data.len() * 4) as u64;
    if byte_len == 0 { return Ok(()); }

    // Fast path: HOST_VISIBLE buffer — direct vkMapMemory write (no staging, no fence)
    if (dst.mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            vk_check(vkMapMemory(dst.device, dst.memory, byte_offset, byte_len, 0, &mut ptr))
                .map_err(VulkanError::Vk)?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8, ptr as *mut u8, byte_len as usize,
            );
            vkUnmapMemory(dst.device, dst.memory);
        }
        return Ok(());
    }

    // Slow path: DEVICE_LOCAL — staging DMA copy
    let stg_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    let stg_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let (stg_buf, stg_memory) = acquire_buffer(gpu, byte_len, stg_usage, stg_mem_flags)?;

    gpu.upload_f32(stg_memory, data)?;

    let command_pool = acquire_command_pool(gpu)?;
    let fence = acquire_fence(gpu)?;
    unsafe {
        let alloc_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cmd: VkCommandBuffer = std::ptr::null_mut();
        vk_check(vkAllocateCommandBuffers(gpu.device, &alloc_info, &mut cmd))
            .map_err(VulkanError::Vk)?;

        let begin_info = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: std::ptr::null(),
        };
        vk_check(vkBeginCommandBuffer(cmd, &begin_info)).map_err(VulkanError::Vk)?;

        let region = VkBufferCopy { srcOffset: 0, dstOffset: byte_offset, size: byte_len };
        vkCmdCopyBuffer(cmd, stg_buf, dst.buffer, 1, &region);

        vk_check(vkEndCommandBuffer(cmd)).map_err(VulkanError::Vk)?;

        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1, pCommandBuffers: &cmd,
            signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
        };
        gpu.queue_submit(1, &submit, fence)?;
        vk_check(vkWaitForFences(gpu.device, 1, &fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }

    release_buffer(gpu.device, byte_len, stg_mem_flags, stg_buf, stg_memory);
    Ok(())
}

/// Read a subregion of a buffer. HOST_VISIBLE: direct memcpy. DEVICE_LOCAL: staging DMA.
fn vm_read_region(
    gpu: &VulkanCompute,
    src: &GpuBuffer,
    byte_offset: u64,
    count: usize,
) -> Result<Vec<f32>, VulkanError> {
    let byte_len = (count * 4) as u64;
    if count == 0 { return Ok(vec![]); }

    // Fast path: HOST_VISIBLE buffer — direct vkMapMemory read (no staging, no fence)
    if (src.mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 {
        let mut result = vec![0.0f32; count];
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            vk_check(vkMapMemory(src.device, src.memory, byte_offset, byte_len, 0, &mut ptr))
                .map_err(VulkanError::Vk)?;
            std::ptr::copy_nonoverlapping(
                ptr as *const u8, result.as_mut_ptr() as *mut u8, byte_len as usize,
            );
            vkUnmapMemory(src.device, src.memory);
        }
        return Ok(result);
    }

    // Slow path: DEVICE_LOCAL — staging DMA copy
    let stg_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    let stg_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let (stg_buf, stg_memory) = acquire_buffer(gpu, byte_len, stg_usage, stg_mem_flags)?;

    let command_pool = acquire_command_pool(gpu)?;
    let fence = acquire_fence(gpu)?;
    unsafe {
        let alloc_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cmd: VkCommandBuffer = std::ptr::null_mut();
        vk_check(vkAllocateCommandBuffers(gpu.device, &alloc_info, &mut cmd))
            .map_err(VulkanError::Vk)?;

        let begin_info = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: std::ptr::null(),
        };
        vk_check(vkBeginCommandBuffer(cmd, &begin_info)).map_err(VulkanError::Vk)?;

        let region = VkBufferCopy { srcOffset: byte_offset, dstOffset: 0, size: byte_len };
        vkCmdCopyBuffer(cmd, src.buffer, stg_buf, 1, &region);

        vk_check(vkEndCommandBuffer(cmd)).map_err(VulkanError::Vk)?;

        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1, pCommandBuffers: &cmd,
            signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
        };
        gpu.queue_submit(1, &submit, fence)?;
        vk_check(vkWaitForFences(gpu.device, 1, &fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }

    let result = gpu.download_f32(stg_memory, count)?;
    release_buffer(gpu.device, byte_len, stg_mem_flags, stg_buf, stg_memory);
    Ok(result)
}

// ── CPU Polling (zero-copy, HOST_VISIBLE) ─────────────────────────────────

/// Status codes for GPU→CPU broadcast (Metrics[instance * 8 + 0]).
pub const VM_STATUS_OK: f32 = 0.0;
pub const VM_STATUS_ANOMALY: f32 = 1.0;
pub const VM_STATUS_ATTENTION: f32 = 2.0;
pub const VM_STATUS_NEED_VM: f32 = 3.0;
pub const VM_STATUS_NEED_IO: f32 = 4.0;
pub const VM_STATUS_NEED_DATA: f32 = 5.0;
pub const VM_STATUS_EXCESS: f32 = 6.0;

/// Poll a VM instance's status from Metrics (single float, zero-copy).
/// Returns Metrics[instance * METRICS_STRIDE + 0] via direct vkMapMemory.
/// Non-blocking, ~1μs. Caller must ensure GPU has completed relevant work (fence/poll).
pub fn vm_poll_status(
    vm: &VmHandle,
    instance: u32,
) -> Result<f32, VulkanError> {
    if instance >= vm.n_instances {
        return Err(VulkanError::Other(format!(
            "vm_poll_status: instance {} >= n_instances {}", instance, vm.n_instances
        )));
    }
    let byte_offset = (instance * METRICS_STRIDE * 4) as u64;
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        vk_check(vkMapMemory(vm.metrics.device, vm.metrics.memory, byte_offset, 4, 0, &mut ptr))
            .map_err(VulkanError::Vk)?;
        let val = *(ptr as *const f32);
        vkUnmapMemory(vm.metrics.device, vm.metrics.memory);
        Ok(val)
    }
}

/// Write data directly to Control SSBO (zero-copy, HOST_VISIBLE).
/// Used by CPU poll loop to activate dormant VMs or write dispatch parameters.
/// No staging, no fence, no command buffer — just vkMapMemory + memcpy.
pub fn vm_write_control_live(
    vm: &VmHandle,
    offset_floats: u32,
    data: &[f32],
) -> Result<(), VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    let byte_len = (data.len() * 4) as u64;
    if byte_len == 0 { return Ok(()); }
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        vk_check(vkMapMemory(vm.control.device, vm.control.memory, byte_offset, byte_len, 0, &mut ptr))
            .map_err(VulkanError::Vk)?;
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8, ptr as *mut u8, byte_len as usize,
        );
        vkUnmapMemory(vm.control.device, vm.control.memory);
    }
    Ok(())
}

/// Write uint32 values directly to Control SSBO (zero-copy, HOST_VISIBLE).
/// Float args are truncated to u32 and written as raw uint32 bytes.
/// Used by CPU poll loop to write dispatch workgroup counts (vkCmdDispatchIndirect reads uint32).
pub fn vm_write_control_u32_live(
    vm: &VmHandle,
    offset_floats: u32,
    data: &[f32],
) -> Result<(), VulkanError> {
    let byte_offset = (offset_floats as u64) * 4;
    let byte_len = (data.len() * 4) as u64;
    if byte_len == 0 { return Ok(()); }
    // Convert floats to uint32 values (truncate)
    let u32_data: Vec<u32> = data.iter().map(|f| *f as u32).collect();
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        vk_check(vkMapMemory(vm.control.device, vm.control.memory, byte_offset, byte_len, 0, &mut ptr))
            .map_err(VulkanError::Vk)?;
        std::ptr::copy_nonoverlapping(
            u32_data.as_ptr() as *const u8, ptr as *mut u8, byte_len as usize,
        );
        vkUnmapMemory(vm.control.device, vm.control.memory);
    }
    Ok(())
}

// ── Dispatch Chain Builder ─────────────────────────────────────────────────

/// A compiled VM program — a VkCommandBuffer with pre-recorded dispatches.
/// Frame resources (command pool, fence, descriptor pool) are returned to
/// VM_FRAME_POOL on Drop instead of being destroyed — O(1) reuse next frame.
pub struct VmProgram {
    pub command_buffer: VkCommandBuffer,
    pub(crate) fence: VkFence,  // copy from frame_res for fast execute/wait/poll access
    device: VkDevice,           // copy from frame_res
    frame_res: Option<VmFrameResources>,  // returned to pool on Drop
    /// Staging buffers from batched uploads — released after program execution.
    staging_buffers: Vec<(VkBuffer, VkDeviceMemory, VkFlags, u64)>,
}

impl Drop for VmProgram {
    fn drop(&mut self) {
        // Release staging buffers back to the buffer pool.
        for &(buf, mem, flags, size) in &self.staging_buffers {
            release_buffer(self.device, size, flags, buf, mem);
        }
        // Return frame resources to pool — no Vulkan destroy calls here.
        // acquire_frame_resources resets them on next use.
        if let Some(res) = self.frame_res.take() {
            release_frame_resources(res);
        }
    }
}

/// One dispatch operation in a VM program.
pub struct VmOp {
    pub spirv: Vec<u8>,
    pub push_constants: Vec<f32>,
    pub workgroups: (u32, u32, u32),
    /// If Some(byte_offset), use vkCmdDispatchIndirect reading from control buffer.
    /// The control buffer at byte_offset must contain 3 consecutive uint32: {x, y, z}.
    pub indirect_offset: Option<u64>,
}

/// Build a VM program: a single VkCommandBuffer with sequential dispatches + barriers.
/// All dispatches share the same SSBOs (registers, metrics, globals, control).
/// Binding layout: 0=registers, 1=metrics, 2=globals, 3=control (4=heap if present).
pub fn vm_build_program(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    ops: &[VmOp],
) -> Result<VmProgram, VulkanError> {
    vm_build_program_with_uploads(gpu, vm, ops, Vec::new())
}

/// Build a VM program with optional batched uploads prepended.
pub fn vm_build_program_with_uploads(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    ops: &[VmOp],
    pending_uploads: Vec<PendingUpload>,
) -> Result<VmProgram, VulkanError> {
    if ops.is_empty() {
        return Err(VulkanError::Other("vm_build_program: no ops".into()));
    }

    let has_heap = vm.heap.is_some();
    let num_bindings: u32 = if has_heap { 5 } else { 4 };
    let device = gpu.device;

    // Collect SSBO buffer handles for descriptor binding
    let mut ssbo_buffers: Vec<VkBuffer> = vec![
        vm.registers.buffer,
        vm.metrics.buffer,
        vm.globals.buffer,
        vm.control.buffer,
    ];
    let mut ssbo_sizes: Vec<u64> = vec![
        vm.registers.size_bytes,
        vm.metrics.size_bytes,
        vm.globals.size_bytes,
        vm.control.size_bytes,
    ];
    if let Some(ref heap) = vm.heap {
        ssbo_buffers.push(heap.buffer);
        ssbo_sizes.push(heap.size_bytes);
    }

    // Acquire reusable frame resources (O(1) reset on all frames after first).
    // First frame: allocates VkCommandPool + VkFence + VkDescriptorPool.
    // Subsequent frames: vkResetCommandPool + vkResetFences + vkResetDescriptorPool.
    let frame_res = acquire_frame_resources(gpu, ops.len() as u32, num_bindings)?;
    let cmd = frame_res.command_buffer;
    let fence = frame_res.fence;
    let desc_pool = frame_res.desc_pool;

    // Begin command buffer
    let begin_info = VkCommandBufferBeginInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        pNext: std::ptr::null(),
        flags: 0, // NOT one-time — this command buffer is reusable
        pInheritanceInfo: std::ptr::null(),
    };
    vk_check(unsafe { vkBeginCommandBuffer(cmd, &begin_info) }).map_err(VulkanError::Vk)?;

    // Record batched uploads: vkCmdCopyBuffer for each pending upload, then memory barrier.
    if !pending_uploads.is_empty() {
        for pu in &pending_uploads {
            let region = VkBufferCopy {
                srcOffset: 0,
                dstOffset: pu.offset,
                size: pu.byte_len,
            };
            unsafe {
                vkCmdCopyBuffer(cmd, pu.staging_buffer, pu.dst_buffer, 1, &region);
            }
        }
        // Memory barrier: transfer writes must complete before compute shader reads.
        let barrier = VkMemoryBarrier {
            sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            pNext: std::ptr::null(),
            srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask: VK_ACCESS_SHADER_READ_BIT,
        };
        unsafe {
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // flags
                1, &barrier as *const _ as *const c_void,
                0, std::ptr::null(),
                0, std::ptr::null(),
            );
        }
    }

    for (op_idx, op) in ops.iter().enumerate() {
        let pc_bytes = (op.push_constants.len() * 4) as u32;

        // 1-4. Get or create pipeline (cached by spirv_hash + binding count + pc size).
        // Cache hit: ~0μs. Cache miss (first use): ~10-15ms per unique SPIR-V.
        let cache_key = (device as usize as u64, spirv_hash(&op.spirv), num_bindings, pc_bytes);
        let cached = VM_PIPELINE_CACHE.with(|c| c.borrow().get(&cache_key).copied());
        let (_shader_module, ds_layout, pipeline_layout, pipeline) = if let Some(cp) = cached {
            (cp.shader_module, cp.ds_layout, cp.pipeline_layout, cp.pipeline)
        } else {
            // Create shader module
            let spirv_words: Vec<u32> = op.spirv
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let shader_info = VkShaderModuleCreateInfo {
                sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                codeSize: spirv_words.len() * 4,
                pCode: spirv_words.as_ptr(),
            };
            let mut sm: VkShaderModule = VK_NULL_HANDLE;
            vk_check(unsafe {
                vkCreateShaderModule(device, &shader_info, std::ptr::null(), &mut sm)
            }).map_err(VulkanError::Vk)?;

            // Create descriptor set layout
            let binding_descs: Vec<VkDescriptorSetLayoutBinding> = (0..num_bindings)
                .map(|i| VkDescriptorSetLayoutBinding {
                    binding: i,
                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount: 1,
                    stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                    pImmutableSamplers: std::ptr::null(),
                })
                .collect();
            let ds_layout_info = VkDescriptorSetLayoutCreateInfo {
                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                bindingCount: num_bindings,
                pBindings: binding_descs.as_ptr(),
            };
            let mut dsl: VkDescriptorSetLayout = VK_NULL_HANDLE;
            vk_check(unsafe {
                vkCreateDescriptorSetLayout(device, &ds_layout_info, std::ptr::null(), &mut dsl)
            }).map_err(VulkanError::Vk)?;

            // Create pipeline layout (with push constants if any)
            let pc_range = VkPushConstantRange {
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                offset: 0,
                size: pc_bytes,
            };
            let dsl_arr = [dsl];
            let pipeline_layout_info = VkPipelineLayoutCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                setLayoutCount: 1,
                pSetLayouts: dsl_arr.as_ptr(),
                pushConstantRangeCount: if pc_bytes > 0 { 1 } else { 0 },
                pPushConstantRanges: if pc_bytes > 0 { &pc_range } else { std::ptr::null() },
            };
            let mut pl: VkPipelineLayout = VK_NULL_HANDLE;
            vk_check(unsafe {
                vkCreatePipelineLayout(device, &pipeline_layout_info, std::ptr::null(), &mut pl)
            }).map_err(VulkanError::Vk)?;

            // Create compute pipeline
            let entry_name = b"main\0";
            let stage = VkPipelineShaderStageCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                module: sm,
                pName: entry_name.as_ptr() as *const _,
                pSpecializationInfo: std::ptr::null(),
            };
            let pipeline_info = VkComputePipelineCreateInfo {
                sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                stage,
                layout: pl,
                basePipelineHandle: VK_NULL_HANDLE,
                basePipelineIndex: -1,
            };
            let mut p: VkPipeline = VK_NULL_HANDLE;
            vk_check(unsafe {
                vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, std::ptr::null(), &mut p)
            }).map_err(VulkanError::Vk)?;

            // Insert into cache (lives for process lifetime — no explicit cleanup needed)
            VM_PIPELINE_CACHE.with(|c| c.borrow_mut().insert(cache_key, VmCachedPipeline {
                shader_module: sm, ds_layout: dsl, pipeline_layout: pl, pipeline: p,
            }));
            (sm, dsl, pl, p)
        };

        // 5. Allocate descriptor set
        let ds_alloc = VkDescriptorSetAllocateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext: std::ptr::null(),
            descriptorPool: desc_pool,
            descriptorSetCount: 1,
            pSetLayouts: &ds_layout,
        };
        let mut descriptor_set: VkDescriptorSet = VK_NULL_HANDLE;
        vk_check(unsafe {
            vkAllocateDescriptorSets(device, &ds_alloc, &mut descriptor_set)
        }).map_err(VulkanError::Vk)?;

        // 6. Bind SSBOs to descriptor set
        let buf_infos: Vec<VkDescriptorBufferInfo> = (0..num_bindings as usize)
            .map(|i| VkDescriptorBufferInfo {
                buffer: ssbo_buffers[i],
                offset: 0,
                range: ssbo_sizes[i],
            })
            .collect();
        let writes: Vec<VkWriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| VkWriteDescriptorSet {
                sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: std::ptr::null(),
                dstSet: descriptor_set,
                dstBinding: i as u32,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: std::ptr::null(),
                pBufferInfo: info,
                pTexelBufferView: std::ptr::null(),
            })
            .collect();
        unsafe {
            vkUpdateDescriptorSets(device, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null());
        }

        // 7. Record: bind pipeline, descriptors, push constants, dispatch
        unsafe {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                0, 1, &descriptor_set, 0, std::ptr::null(),
            );
            if pc_bytes > 0 {
                vkCmdPushConstants(
                    cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, pc_bytes, op.push_constants.as_ptr() as *const c_void,
                );
            }
            if let Some(indirect_byte_offset) = op.indirect_offset {
                // Indirect dispatch: GPU reads workgroup counts from control buffer
                vkCmdDispatchIndirect(cmd, vm.control.buffer, indirect_byte_offset);
            } else {
                vkCmdDispatch(cmd, op.workgroups.0, op.workgroups.1, op.workgroups.2);
            }
        }

        // 8. Pipeline barrier between dispatches
        if op_idx < ops.len() - 1 {
            // Check if NEXT op is indirect — need DRAW_INDIRECT stage + INDIRECT_COMMAND_READ access
            let next_is_indirect = ops[op_idx + 1].indirect_offset.is_some();
            let dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                | if next_is_indirect { VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT } else { 0 };
            let dst_access = VK_ACCESS_SHADER_READ_BIT
                | if next_is_indirect { VK_ACCESS_INDIRECT_COMMAND_READ_BIT } else { 0 };
            let barrier = VkMemoryBarrier {
                sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                pNext: std::ptr::null(),
                srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask: dst_access,
            };
            unsafe {
                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    dst_stage,
                    0,
                    1, &barrier as *const VkMemoryBarrier as *const c_void,
                    0, std::ptr::null(),
                    0, std::ptr::null(),
                );
            }
        }
    }

    vk_check(unsafe { vkEndCommandBuffer(cmd) }).map_err(VulkanError::Vk)?;

    // Collect staging buffer info for deferred release (after program execution).
    let staging_buffers: Vec<(VkBuffer, VkDeviceMemory, VkFlags, u64)> = pending_uploads
        .into_iter()
        .map(|pu| (pu.staging_buffer, pu.staging_memory, pu.staging_mem_flags, pu.byte_len))
        .collect();

    Ok(VmProgram {
        command_buffer: cmd,
        fence,
        device,
        frame_res: Some(frame_res),
        staging_buffers,
    })
}

/// Execute a VM program (submit command buffer + wait for fence).
pub fn vm_execute_program(
    gpu: &VulkanCompute,
    prog: &VmProgram,
) -> Result<(), VulkanError> {
    unsafe {
        vk_check(vkResetFences(gpu.device, 1, &prog.fence)).map_err(VulkanError::Vk)?;
    }
    let submit = VkSubmitInfo {
        sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
        pNext: std::ptr::null(),
        waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
        pWaitDstStageMask: std::ptr::null(),
        commandBufferCount: 1, pCommandBuffers: &prog.command_buffer,
        signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
    };
    unsafe {
        gpu.queue_submit(1, &submit, prog.fence)?;
        vk_check(vkWaitForFences(gpu.device, 1, &prog.fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }
    Ok(())
}

/// Submit a VM program without waiting — GPU runs autonomously.
/// CPU can poll with vm_poll_program() or block with vm_wait_program().
pub fn vm_execute_async(
    gpu: &VulkanCompute,
    prog: &VmProgram,
) -> Result<(), VulkanError> {
    unsafe {
        vk_check(vkResetFences(gpu.device, 1, &prog.fence)).map_err(VulkanError::Vk)?;
    }
    let submit = VkSubmitInfo {
        sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
        pNext: std::ptr::null(),
        waitSemaphoreCount: 0, pWaitSemaphores: std::ptr::null(),
        pWaitDstStageMask: std::ptr::null(),
        commandBufferCount: 1, pCommandBuffers: &prog.command_buffer,
        signalSemaphoreCount: 0, pSignalSemaphores: std::ptr::null(),
    };
    unsafe {
        gpu.queue_submit(1, &submit, prog.fence)?;
    }
    Ok(())
}

/// Poll a VM program's fence — returns true if GPU work is complete.
/// Non-blocking: returns immediately.
pub fn vm_poll_program(prog: &VmProgram) -> bool {
    let status = unsafe { vkGetFenceStatus(prog.device, prog.fence) };
    status == VK_SUCCESS
}

/// Wait for a VM program to complete.
/// Fast path: if fence is already signaled, returns immediately (~1us).
/// Slow path: blocks until GPU completes.
pub fn vm_wait_program(prog: &VmProgram) -> Result<(), VulkanError> {
    unsafe {
        // Fast path: check if already signaled (non-blocking)
        let status = vkGetFenceStatus(prog.device, prog.fence);
        if status == VK_SUCCESS {
            return Ok(()); // GPU already done — skip blocking wait
        }
        // Slow path: GPU still running — block
        vk_check(vkWaitForFences(prog.device, 1, &prog.fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::VulkanCompute;

    fn get_gpu() -> Option<VulkanCompute> {
        match VulkanCompute::new() {
            Ok(g) => Some(g),
            Err(_) => { eprintln!("Skipping VM test — no Vulkan device"); None }
        }
    }

    /// vm_add.spv: reads registers at flat offsets a_off, b_off, writes sum to dst_off.
    /// Push constants: [a_off, b_off, dst_off, count].
    /// Binding layout: 0=registers, 1=metrics, 2=globals, 3=control.
    static VM_ADD_SPV: &[u8] = include_bytes!("../../../stdlib/loom/kernels/ops/vm_add.spv");

    #[test]
    fn test_multi_vm_concurrent_launch() {
        let gpu = match get_gpu() { Some(g) => g, None => return };

        // Boot 2 VMs: 1 instance, reg_size=8, globals=16
        let vm1 = vm_create(&gpu, 1, 8, 16).expect("vm1 create");
        let vm2 = vm_create(&gpu, 1, 8, 16).expect("vm2 create");

        // VM1: write [1,2,3,4] to reg0, [10,20,30,40] to reg1
        vm_write_reg(&gpu, &vm1, 0, 0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        vm_write_reg(&gpu, &vm1, 0, 1, &[10.0, 20.0, 30.0, 40.0]).unwrap();

        // VM2: write [100,200,300,400] to reg0, [5,6,7,8] to reg1
        vm_write_reg(&gpu, &vm2, 0, 0, &[100.0, 200.0, 300.0, 400.0]).unwrap();
        vm_write_reg(&gpu, &vm2, 0, 1, &[5.0, 6.0, 7.0, 8.0]).unwrap();

        // Push constants: a_off=0, b_off=8, dst_off=16, count=4
        let pc = vec![0.0f32, 8.0, 16.0, 4.0];

        // Build programs for both VMs
        let op1 = VmOp { spirv: VM_ADD_SPV.to_vec(), push_constants: pc.clone(), workgroups: (1, 1, 1), indirect_offset: None };
        let op2 = VmOp { spirv: VM_ADD_SPV.to_vec(), push_constants: pc.clone(), workgroups: (1, 1, 1), indirect_offset: None };
        let prog1 = vm_build_program(&gpu, &vm1, &[op1]).expect("build prog1");
        let prog2 = vm_build_program(&gpu, &vm2, &[op2]).expect("build prog2");

        // Launch both async (concurrent GPU execution)
        vm_execute_async(&gpu, &prog1).expect("launch prog1");
        vm_execute_async(&gpu, &prog2).expect("launch prog2");

        // Wait both
        vm_wait_program(&prog1).expect("wait prog1");
        vm_wait_program(&prog2).expect("wait prog2");

        // Read results: reg2 (flat offset 16) should have the sums
        let r1 = vm_read_reg(&gpu, &vm1, 0, 2, 4).expect("read vm1 reg2");
        let r2 = vm_read_reg(&gpu, &vm2, 0, 2, 4).expect("read vm2 reg2");

        assert_eq!(r1, vec![11.0, 22.0, 33.0, 44.0], "VM1: 1+10=11, 2+20=22, ...");
        assert_eq!(r2, vec![105.0, 206.0, 307.0, 408.0], "VM2: 100+5=105, 200+6=206, ...");
    }

    #[test]
    fn test_multi_vm_sequential_launch_wait() {
        let gpu = match get_gpu() { Some(g) => g, None => return };

        let vm1 = vm_create(&gpu, 1, 8, 16).expect("vm1 create");
        let vm2 = vm_create(&gpu, 1, 8, 16).expect("vm2 create");

        // VM1: [2,4,6,8] + [1,1,1,1]
        vm_write_reg(&gpu, &vm1, 0, 0, &[2.0, 4.0, 6.0, 8.0]).unwrap();
        vm_write_reg(&gpu, &vm1, 0, 1, &[1.0, 1.0, 1.0, 1.0]).unwrap();

        // VM2: [10,10,10,10] + [5,5,5,5]
        vm_write_reg(&gpu, &vm2, 0, 0, &[10.0, 10.0, 10.0, 10.0]).unwrap();
        vm_write_reg(&gpu, &vm2, 0, 1, &[5.0, 5.0, 5.0, 5.0]).unwrap();

        let pc = vec![0.0f32, 8.0, 16.0, 4.0];

        // Build and execute VM1 first (sequential)
        let op1 = VmOp { spirv: VM_ADD_SPV.to_vec(), push_constants: pc.clone(), workgroups: (1, 1, 1), indirect_offset: None };
        let prog1 = vm_build_program(&gpu, &vm1, &[op1]).expect("build prog1");
        vm_execute_program(&gpu, &prog1).expect("exec prog1");

        let r1 = vm_read_reg(&gpu, &vm1, 0, 2, 4).expect("read vm1");
        assert_eq!(r1, vec![3.0, 5.0, 7.0, 9.0]);

        // Then VM2 (sequential, reusing queue after VM1 finished)
        let op2 = VmOp { spirv: VM_ADD_SPV.to_vec(), push_constants: pc.clone(), workgroups: (1, 1, 1), indirect_offset: None };
        let prog2 = vm_build_program(&gpu, &vm2, &[op2]).expect("build prog2");
        vm_execute_program(&gpu, &prog2).expect("exec prog2");

        let r2 = vm_read_reg(&gpu, &vm2, 0, 2, 4).expect("read vm2");
        assert_eq!(r2, vec![15.0, 15.0, 15.0, 15.0]);
    }

    #[test]
    fn test_multi_vm_poll_loop() {
        let gpu = match get_gpu() { Some(g) => g, None => return };

        // Boot 4 VMs
        let vms: Vec<VmHandle> = (0..4).map(|i| {
            let vm = vm_create(&gpu, 1, 8, 16).expect(&format!("vm{} create", i));
            // Each VM: [i*10, i*10+1, i*10+2, i*10+3] + [1,1,1,1]
            let base = (i as f32) * 10.0;
            vm_write_reg(&gpu, &vm, 0, 0, &[base, base + 1.0, base + 2.0, base + 3.0]).unwrap();
            vm_write_reg(&gpu, &vm, 0, 1, &[1.0, 1.0, 1.0, 1.0]).unwrap();
            vm
        }).collect();

        let pc = vec![0.0f32, 8.0, 16.0, 4.0];

        // Build and launch all 4 async
        let progs: Vec<VmProgram> = vms.iter().map(|vm| {
            let op = VmOp { spirv: VM_ADD_SPV.to_vec(), push_constants: pc.clone(), workgroups: (1, 1, 1), indirect_offset: None };
            vm_build_program(&gpu, vm, &[op]).expect("build")
        }).collect();

        for prog in &progs {
            vm_execute_async(&gpu, prog).expect("launch");
        }

        // Poll loop until all complete
        let mut done = vec![false; 4];
        for _ in 0..10_000 {
            let mut all_done = true;
            for (i, prog) in progs.iter().enumerate() {
                if !done[i] {
                    done[i] = vm_poll_program(prog);
                }
                if !done[i] { all_done = false; }
            }
            if all_done { break; }
        }
        assert!(done.iter().all(|&d| d), "all 4 VMs should complete within poll limit");

        // Verify results
        for (i, vm) in vms.iter().enumerate() {
            let r = vm_read_reg(&gpu, vm, 0, 2, 4).expect(&format!("read vm{}", i));
            let base = (i as f32) * 10.0;
            let expected = vec![base + 1.0, base + 2.0, base + 3.0, base + 4.0];
            assert_eq!(r, expected, "VM{} result mismatch", i);
        }
    }

    #[test]
    fn test_loom_copy_between_vms() {
        let gpu = match get_gpu() { Some(g) => g, None => return };

        // VM1: globals_size=8, VM2: globals_size=8
        let vm1 = vm_create(&gpu, 1, 4, 8).expect("vm1 create");
        let vm2 = vm_create(&gpu, 1, 4, 8).expect("vm2 create");

        // Write data to VM1 globals
        let data = [42.0f32, 43.0, 44.0, 45.0];
        vm_write_globals(&gpu, &vm1, 0, &data).unwrap();

        // GPU→GPU copy: VM1 globals[0..4] → VM2 globals[2..6]
        vm_copy_globals(&gpu, &vm1, 0, &vm2, 2, 4).unwrap();

        // Read VM2 globals — first 2 should be 0.0 (untouched), then our data
        let result = vm_read_globals(&gpu, &vm2, 0, 8).unwrap();
        assert_eq!(result[0], 0.0, "VM2 globals[0] untouched");
        assert_eq!(result[1], 0.0, "VM2 globals[1] untouched");
        assert_eq!(result[2], 42.0, "VM2 globals[2] = copied");
        assert_eq!(result[3], 43.0, "VM2 globals[3] = copied");
        assert_eq!(result[4], 44.0, "VM2 globals[4] = copied");
        assert_eq!(result[5], 45.0, "VM2 globals[5] = copied");
    }
}
