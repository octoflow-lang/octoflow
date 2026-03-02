//! OctoFlow Vulkan compute runtime.
//!
//! Provides [`VulkanCompute`] for GPU device management and
//! [`dispatch_compute`] for running SPIR-V compute shaders on the GPU.
//!
//! Phase 49: Replaced ash crate with raw vk_sys bindings (extern "system" + vulkan-1.lib).
//! Phase 118: Replaced SPIR-V crate with pre-built .spv kernels from ir.flow.
//! Phase 119: Consolidated gpu_ops + matmul + types into dispatch.rs.

pub mod device;
pub mod dispatch;
pub mod gpu_run;
pub mod memory;
pub mod vm;
pub(crate) mod vk_sys;

pub use device::{VulkanCompute, VulkanError, GpuInfo};
pub use dispatch::{
    // Core dispatch
    dispatch_compute, dispatch_compute_pc, dispatch_with_buffers, dispatch_reduce,
    BufferSpec, GpuBuffer, GpuBufferRef,
    upload_buffer, upload_buffer_f16, download_buffer, download_buffer_fast, read_buffer_element,
    f32_to_f16, f16_to_f32,
    // GPU-resident dispatch
    dispatch_resident, dispatch_resident_pc, dispatch_resident_deferred, flush_pending, has_pending,
    // High-level operations (formerly gpu_ops.rs)
    dispatch_binop, dispatch_select, dispatch_map_op, dispatch_reduce_op,
    dispatch_binop_resident, dispatch_map_resident, dispatch_select_resident,
    dispatch_binop_deferred, dispatch_map_deferred, dispatch_select_deferred,
    // Matrix multiply (formerly matmul.rs)
    dispatch_matmul,
    // Operation enums (formerly types.rs)
    MapOp, BinaryOp, ReduceOp, FusedOp, TemporalOp,
};
pub use gpu_run::gpu_run_dispatch;
