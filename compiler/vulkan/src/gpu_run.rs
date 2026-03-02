//! Universal GPU compute dispatch for `gpu_run()` builtin.
//!
//! Loads a pre-compiled .spv kernel, creates/caches the Vulkan pipeline,
//! uploads input buffers, dispatches, and reads back the output buffer.
//!
//! Pipeline objects (shader module, descriptor set layout, pipeline layout,
//! compute pipeline) are cached per unique (device, spirv_hash, binding_count,
//! push_constant_size) tuple and reused across calls.

use crate::device::{VulkanCompute, VulkanError};
use crate::vk_sys::*;

/// Cached compiled Vulkan pipeline — reused across calls with the same SPIR-V.
#[derive(Copy, Clone)]
struct CachedPipeline {
    shader_module: VkShaderModule,
    ds_layout: VkDescriptorSetLayout,
    pipeline_layout: VkPipelineLayout,
    pipeline: VkPipeline,
}

thread_local! {
    /// Pipeline cache keyed by `(device_ptr, spirv_hash, binding_count, push_constant_bytes)`.
    static GPU_RUN_CACHE: std::cell::RefCell<
        std::collections::HashMap<(u64, u64, u32, u32), CachedPipeline>
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// FNV-1a 64-bit hash of SPIR-V bytes.
fn spirv_hash(spirv: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in spirv {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Universal GPU compute dispatch.
///
/// # Arguments
/// - `gpu` — Vulkan device context
/// - `spirv` — Pre-compiled SPIR-V compute shader bytes
/// - `inputs` — Input arrays (each becomes a storage buffer at binding 0..N-1)
/// - `push_constants` — Optional scalar parameters passed via push constants
/// - `output_size` — Number of f32 elements in the output buffer
/// - `workgroups` — Optional (x, y, z) workgroup counts. None = auto 1D (ceil(output_size/256))
///
/// The output buffer is always the last binding (binding N).
/// Push constants, if any, are bound as a single contiguous f32 block.
pub fn gpu_run_dispatch(
    gpu: &VulkanCompute,
    spirv: &[u8],
    inputs: &[Vec<f32>],
    push_constants: &[f32],
    output_size: usize,
    workgroups: Option<(u32, u32, u32)>,
) -> Result<Vec<f32>, VulkanError> {
    let device = gpu.device;
    let num_bindings = inputs.len() as u32 + 1; // inputs + 1 output
    let pc_bytes = (push_constants.len() * 4) as u32;

    // ── Pipeline (cached) ────────────────────────────────────────────────────
    assert!(spirv.len() % 4 == 0, "SPIR-V binary must be 4-byte aligned");
    let cache_key = (device as usize as u64, spirv_hash(spirv), num_bindings, pc_bytes);
    let cached = GPU_RUN_CACHE.with(|c| c.borrow().get(&cache_key).copied());
    let (_shader_module, ds_layout, pipeline_layout, pipeline) = if let Some(c) = cached {
        (c.shader_module, c.ds_layout, c.pipeline_layout, c.pipeline)
    } else {
        let spirv_words: Vec<u32> = spirv
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
        let mut shader_module: VkShaderModule = VK_NULL_HANDLE;
        vk_check(unsafe {
            vkCreateShaderModule(device, &shader_info, std::ptr::null(), &mut shader_module)
        }).map_err(VulkanError::Vk)?;

        // Descriptor set layout: one storage buffer per binding
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
        let mut ds_layout: VkDescriptorSetLayout = VK_NULL_HANDLE;
        vk_check(unsafe {
            vkCreateDescriptorSetLayout(device, &ds_layout_info, std::ptr::null(), &mut ds_layout)
        }).map_err(VulkanError::Vk)?;

        // Pipeline layout (with optional push constant range)
        let pc_range = VkPushConstantRange {
            stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: pc_bytes,
        };
        let dsl_arr = [ds_layout];
        let pipeline_layout_info = VkPipelineLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            setLayoutCount: 1,
            pSetLayouts: dsl_arr.as_ptr(),
            pushConstantRangeCount: if pc_bytes > 0 { 1 } else { 0 },
            pPushConstantRanges: if pc_bytes > 0 { &pc_range } else { std::ptr::null() },
        };
        let mut pipeline_layout: VkPipelineLayout = VK_NULL_HANDLE;
        vk_check(unsafe {
            vkCreatePipelineLayout(device, &pipeline_layout_info, std::ptr::null(), &mut pipeline_layout)
        }).map_err(VulkanError::Vk)?;

        // Compute pipeline
        let entry_name = b"main\0";
        let stage = VkPipelineShaderStageCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            pName: entry_name.as_ptr() as *const _,
            pSpecializationInfo: std::ptr::null(),
        };
        let pipeline_info = VkComputePipelineCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            basePipelineHandle: VK_NULL_HANDLE,
            basePipelineIndex: -1,
        };
        let mut pipeline: VkPipeline = VK_NULL_HANDLE;
        vk_check(unsafe {
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, std::ptr::null(), &mut pipeline)
        }).map_err(VulkanError::Vk)?;

        GPU_RUN_CACHE.with(|c| c.borrow_mut().insert(cache_key, CachedPipeline {
            shader_module, ds_layout, pipeline_layout, pipeline,
        }));
        (shader_module, ds_layout, pipeline_layout, pipeline)
    };
    let layouts = [ds_layout];

    // ── Buffers ──────────────────────────────────────────────────────────────
    let mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    let mut vk_buffers: Vec<VkBuffer> = Vec::with_capacity(num_bindings as usize);
    let mut vk_memories: Vec<VkDeviceMemory> = Vec::with_capacity(num_bindings as usize);

    // Input buffers (bindings 0..N-1)
    for input in inputs {
        let buf_size = (input.len() * 4) as VkDeviceSize;
        let (buf, mem) = gpu.create_buffer(buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem_flags)?;
        gpu.upload_f32(mem, input)?;
        vk_buffers.push(buf);
        vk_memories.push(mem);
    }

    // Output buffer (binding N)
    let out_size = (output_size * 4) as VkDeviceSize;
    let (out_buf, out_mem) = gpu.create_buffer(out_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem_flags)?;
    // Zero-initialize output
    let zeros = vec![0.0f32; output_size];
    gpu.upload_f32(out_mem, &zeros)?;
    vk_buffers.push(out_buf);
    vk_memories.push(out_mem);

    // ── Descriptor pool & set ────────────────────────────────────────────────
    let pool_size = VkDescriptorPoolSize {
        ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount: num_bindings,
    };
    let pool_info = VkDescriptorPoolCreateInfo {
        sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        maxSets: 1,
        poolSizeCount: 1,
        pPoolSizes: &pool_size,
    };
    let mut descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;
    vk_check(unsafe {
        vkCreateDescriptorPool(device, &pool_info, std::ptr::null(), &mut descriptor_pool)
    }).map_err(VulkanError::Vk)?;

    let alloc_info = VkDescriptorSetAllocateInfo {
        sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        pNext: std::ptr::null(),
        descriptorPool: descriptor_pool,
        descriptorSetCount: 1,
        pSetLayouts: layouts.as_ptr(),
    };
    let mut descriptor_set: VkDescriptorSet = VK_NULL_HANDLE;
    vk_check(unsafe {
        vkAllocateDescriptorSets(device, &alloc_info, &mut descriptor_set)
    }).map_err(VulkanError::Vk)?;

    // Bind buffers to descriptor set
    let buf_infos: Vec<VkDescriptorBufferInfo> = vk_buffers
        .iter()
        .enumerate()
        .map(|(i, &buf)| {
            let size = if i < inputs.len() {
                (inputs[i].len() * 4) as VkDeviceSize
            } else {
                out_size
            };
            VkDescriptorBufferInfo { buffer: buf, offset: 0, range: size }
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

    unsafe { vkUpdateDescriptorSets(device, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null()); }

    // ── Command buffer ───────────────────────────────────────────────────────
    let cmd_pool_info = VkCommandPoolCreateInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        queueFamilyIndex: gpu.queue_family_index,
    };
    let mut command_pool: VkCommandPool = VK_NULL_HANDLE;
    vk_check(unsafe {
        vkCreateCommandPool(device, &cmd_pool_info, std::ptr::null(), &mut command_pool)
    }).map_err(VulkanError::Vk)?;

    let cmd_alloc_info = VkCommandBufferAllocateInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        pNext: std::ptr::null(),
        commandPool: command_pool,
        level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount: 1,
    };
    let mut cmd: VkCommandBuffer = std::ptr::null_mut();
    vk_check(unsafe {
        vkAllocateCommandBuffers(device, &cmd_alloc_info, &mut cmd)
    }).map_err(VulkanError::Vk)?;

    let begin_info = VkCommandBufferBeginInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        pNext: std::ptr::null(),
        flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        pInheritanceInfo: std::ptr::null(),
    };

    // Compute workgroup counts
    let (wg_x, wg_y, wg_z) = workgroups.unwrap_or_else(|| {
        ((output_size as u32).div_ceil(256), 1, 1)
    });

    unsafe {
        vk_check(vkBeginCommandBuffer(cmd, &begin_info)).map_err(VulkanError::Vk)?;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
            0, 1, &descriptor_set, 0, std::ptr::null(),
        );
        // Push constants (if any)
        if pc_bytes > 0 {
            vkCmdPushConstants(
                cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, pc_bytes, push_constants.as_ptr() as *const std::ffi::c_void,
            );
        }
        vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
        vk_check(vkEndCommandBuffer(cmd)).map_err(VulkanError::Vk)?;
    }

    // ── Submit & wait ────────────────────────────────────────────────────────
    let fence_info = VkFenceCreateInfo {
        sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
    };
    let mut fence: VkFence = VK_NULL_HANDLE;
    vk_check(unsafe {
        vkCreateFence(device, &fence_info, std::ptr::null(), &mut fence)
    }).map_err(VulkanError::Vk)?;

    let submit_info = VkSubmitInfo {
        sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
        pNext: std::ptr::null(),
        waitSemaphoreCount: 0,
        pWaitSemaphores: std::ptr::null(),
        pWaitDstStageMask: std::ptr::null(),
        commandBufferCount: 1,
        pCommandBuffers: &cmd,
        signalSemaphoreCount: 0,
        pSignalSemaphores: std::ptr::null(),
    };

    unsafe {
        gpu.queue_submit(1, &submit_info, fence)?;
        vk_check(vkWaitForFences(device, 1, &fence, 1, u64::MAX))
            .map_err(VulkanError::Vk)?;
    }

    // ── Readback ─────────────────────────────────────────────────────────────
    let output = gpu.download_f32(out_mem, output_size)?;

    // ── Cleanup (per-dispatch resources only — pipeline stays cached) ────────
    unsafe {
        vkDestroyFence(device, fence, std::ptr::null());
        vkDestroyCommandPool(device, command_pool, std::ptr::null());
        vkDestroyDescriptorPool(device, descriptor_pool, std::ptr::null());
        for &mem in &vk_memories {
            vkFreeMemory(device, mem, std::ptr::null());
        }
        for &buf in &vk_buffers {
            vkDestroyBuffer(device, buf, std::ptr::null());
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::VulkanCompute;

    fn make_double_spirv() -> &'static [u8] {
        include_bytes!("../../../stdlib/loom/kernels/math/double.spv")
    }

    fn make_add_spirv() -> &'static [u8] {
        include_bytes!("../../../stdlib/loom/kernels/math/add.spv")
    }

    #[test]
    fn test_gpu_run_double() {
        let gpu = match VulkanCompute::new() {
            Ok(g) => g,
            Err(_) => return, // skip if no GPU
        };
        let spirv = make_double_spirv();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = gpu_run_dispatch(&gpu, &spirv, &[input], &[], 4, None).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_gpu_run_binop_add() {
        let gpu = match VulkanCompute::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let spirv = make_add_spirv();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let result = gpu_run_dispatch(&gpu, &spirv, &[a, b], &[], 4, None).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_gpu_run_pipeline_caching() {
        let gpu = match VulkanCompute::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let spirv = make_double_spirv();
        // First call — creates pipeline
        let r1 = gpu_run_dispatch(&gpu, &spirv, &[vec![1.0, 2.0]], &[], 2, None).unwrap();
        // Second call — should reuse cached pipeline
        let r2 = gpu_run_dispatch(&gpu, &spirv, &[vec![5.0, 10.0]], &[], 2, None).unwrap();
        assert_eq!(r1, vec![2.0, 4.0]);
        assert_eq!(r2, vec![10.0, 20.0]);
    }
}
