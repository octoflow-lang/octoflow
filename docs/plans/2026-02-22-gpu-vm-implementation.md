# GPU VM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a general-purpose GPU VM with SSBO-backed registers, dispatch-chain execution, and message passing between VM instances. Prove it works with a matvec kernel before mapping LLM inference onto it.

**Architecture:** Each VM instance has a register file (32 registers x 4096 floats) in a DEVICE_LOCAL SSBO. All instances share a heap SSBO (weights), globals SSBO (KV cache), and control SSBO (homeostasis parameters). The CPU (BIOS) boots the VM, writes inputs, submits one command buffer containing the full dispatch chain, and reads outputs. The GPU executes autonomously.

**Tech Stack:** Rust (flowgpu-vulkan crate), SPIR-V kernels, OctoFlow .flow scripts

**Design Doc:** `docs/plans/2026-02-22-gpu-vm-design.md`

---

## Task 1: VM State Module in flowgpu-vulkan

Create the Rust-side VM state management. This is the foundation everything else builds on.

**Files:**
- Create: `arms/flowgpu-vulkan/src/vm.rs`
- Modify: `arms/flowgpu-vulkan/src/lib.rs` (add `pub mod vm;`)

**Step 1: Create vm.rs with VmHandle struct and allocation**

```rust
// arms/flowgpu-vulkan/src/vm.rs
use crate::{VulkanCompute, GpuBuffer, VulkanError};
use crate::dispatch::{upload_buffer_vram, download_buffer_fast, acquire_buffer};
use crate::vk_sys::*;

/// GPU VM handle. Owns DEVICE_LOCAL SSBOs for registers, metrics, heap, globals, control.
pub struct VmHandle {
    pub n_instances: u32,          // Number of VM instances (e.g., 28 layers)
    pub reg_size: u32,             // Floats per register (e.g., 4096)
    pub n_registers: u32,          // Registers per instance (always 32)
    pub registers: GpuBuffer,      // Binding 0: n_instances * 32 * reg_size floats
    pub metrics: GpuBuffer,        // Binding 1: n_instances * 8 floats
    pub globals: GpuBuffer,        // Binding 3: user-specified size
    pub control: GpuBuffer,        // Binding 4: n_instances * 8 floats
    // Binding 2 (heap) is added separately via vm_heap_alloc
    pub heap: Option<GpuBuffer>,
}

const METRICS_STRIDE: u32 = 8;    // floats per instance in metrics SSBO
const CONTROL_STRIDE: u32 = 8;    // floats per instance in control SSBO

/// Create a new GPU VM with n_instances, each having 32 registers of reg_size floats.
pub fn vm_create(
    gpu: &VulkanCompute,
    n_instances: u32,
    reg_size: u32,
    globals_size: u32,
) -> Result<VmHandle, VulkanError> {
    let total_reg_floats = (n_instances * 32 * reg_size) as usize;
    let total_metrics_floats = (n_instances * METRICS_STRIDE) as usize;
    let total_globals_floats = globals_size as usize;
    let total_control_floats = (n_instances * CONTROL_STRIDE) as usize;

    // Allocate DEVICE_LOCAL SSBOs (zero-initialized via staging upload)
    let reg_data = vec![0.0f32; total_reg_floats];
    let metrics_data = vec![0.0f32; total_metrics_floats];
    let globals_data = vec![0.0f32; total_globals_floats.max(1)];
    let control_data = vec![0.0f32; total_control_floats];

    let registers = upload_buffer_vram(gpu, &reg_data)?;
    let metrics = upload_buffer_vram(gpu, &metrics_data)?;
    let globals = upload_buffer_vram(gpu, &globals_data)?;
    let control = upload_buffer_vram(gpu, &control_data)?;

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

/// Write data to a specific register of a specific VM instance.
/// Uses staging buffer to DMA into DEVICE_LOCAL register SSBO.
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
    let byte_offset = offset_floats * 4;
    let write_len = data.len().min(vm.reg_size as usize);

    // Stage data into HOST_VISIBLE buffer, then DMA copy to register SSBO region
    vm_write_region(gpu, &vm.registers, byte_offset as u64, &data[..write_len])
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
    let byte_offset = offset_floats * 4;
    let read_len = (count as usize).min(vm.reg_size as usize);

    vm_read_region(gpu, &vm.registers, byte_offset as u64, read_len)
}

/// Write a subregion of a DEVICE_LOCAL buffer via staging + DMA copy.
fn vm_write_region(
    gpu: &VulkanCompute,
    dst: &GpuBuffer,
    byte_offset: u64,
    data: &[f32],
) -> Result<(), VulkanError> {
    let byte_len = (data.len() * 4) as u64;
    // 1. Create HOST_VISIBLE staging buffer with data
    let staging_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    let (staging_buf, staging_mem) = acquire_buffer(gpu, byte_len, usage, staging_flags)?;

    // 2. Map staging, write data
    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        vkMapMemory(gpu.device, staging_mem, 0, byte_len, 0, &mut mapped);
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, mapped as *mut u8, byte_len as usize);
        vkUnmapMemory(gpu.device, staging_mem);
    }

    // 3. DMA copy staging -> dst at offset
    let cmd = crate::dispatch::begin_single_command(gpu)?;
    let region = VkBufferCopy { srcOffset: 0, dstOffset: byte_offset, size: byte_len };
    unsafe { vkCmdCopyBuffer(cmd, staging_buf, dst.buffer, 1, &region); }
    crate::dispatch::end_single_command(gpu, cmd)?;

    // 4. Free staging
    unsafe {
        vkDestroyBuffer(gpu.device, staging_buf, std::ptr::null());
        vkFreeMemory(gpu.device, staging_mem, std::ptr::null());
    }
    Ok(())
}

/// Read a subregion of a DEVICE_LOCAL buffer via staging + DMA copy.
fn vm_read_region(
    gpu: &VulkanCompute,
    src: &GpuBuffer,
    byte_offset: u64,
    count: usize,
) -> Result<Vec<f32>, VulkanError> {
    let byte_len = (count * 4) as u64;
    let staging_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    let (staging_buf, staging_mem) = acquire_buffer(gpu, byte_len, usage, staging_flags)?;

    // DMA copy src region -> staging
    let cmd = crate::dispatch::begin_single_command(gpu)?;
    let region = VkBufferCopy { srcOffset: byte_offset, dstOffset: 0, size: byte_len };
    unsafe { vkCmdCopyBuffer(cmd, src.buffer, staging_buf, 1, &region); }
    crate::dispatch::end_single_command(gpu, cmd)?;

    // Map staging, read data
    let mut result = vec![0.0f32; count];
    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        vkMapMemory(gpu.device, staging_mem, 0, byte_len, 0, &mut mapped);
        std::ptr::copy_nonoverlapping(mapped as *const u8, result.as_mut_ptr() as *mut u8, byte_len as usize);
        vkUnmapMemory(gpu.device, staging_mem);
    }

    unsafe {
        vkDestroyBuffer(gpu.device, staging_buf, std::ptr::null());
        vkFreeMemory(gpu.device, staging_mem, std::ptr::null());
    }
    Ok(result)
}

/// Destroy a VM, freeing all GPU resources.
pub fn vm_destroy(vm: VmHandle) {
    // GpuBuffer Drop impl handles VkBuffer/VkDeviceMemory cleanup
    drop(vm);
}
```

**Step 2: Add `pub mod vm;` to lib.rs**

In `arms/flowgpu-vulkan/src/lib.rs`, add:
```rust
pub mod vm;
```

**Step 3: Verify build**

Run: `cargo test -p flowgpu-vulkan 2>&1 | tail -5`
Expected: Compilation succeeds (no tests yet for vm.rs, but no errors)

**Step 4: Commit**

```bash
git add arms/flowgpu-vulkan/src/vm.rs arms/flowgpu-vulkan/src/lib.rs
git commit -m "feat(gpu-vm): add vm.rs with VmHandle, register SSBO allocation"
```

---

## Task 2: Expose Helper Functions from dispatch.rs

The VM module needs `begin_single_command`, `end_single_command`, and `acquire_buffer` from dispatch.rs. These may be `pub(crate)` or private. Make them accessible.

**Files:**
- Modify: `arms/flowgpu-vulkan/src/dispatch.rs`

**Step 1: Check visibility of needed functions**

Search dispatch.rs for:
- `fn begin_single_command` — if private, add `pub(crate)`
- `fn end_single_command` — if private, add `pub(crate)`
- `fn acquire_buffer` — if private, add `pub(crate)`
- `pub fn upload_buffer_vram` — should already be pub
- `pub fn download_buffer_fast` — should already be pub

**Step 2: Make functions pub(crate) if needed**

Only change visibility, no logic changes.

**Step 3: Verify build**

Run: `cargo test -p flowgpu-vulkan 2>&1 | tail -5`

**Step 4: Also check that VkBufferCopy and vkCmdCopyBuffer exist in vk_sys.rs**

If `VkBufferCopy` struct and `vkCmdCopyBuffer` FFI binding don't exist, add them:

```rust
// In vk_sys.rs
#[repr(C)]
pub struct VkBufferCopy {
    pub srcOffset: u64,
    pub dstOffset: u64,
    pub size: u64,
}

extern "system" {
    pub fn vkCmdCopyBuffer(
        commandBuffer: VkCommandBuffer,
        srcBuffer: VkBuffer,
        dstBuffer: VkBuffer,
        regionCount: u32,
        pRegions: *const VkBufferCopy,
    );
}
```

**Step 5: Commit**

```bash
git add arms/flowgpu-vulkan/src/dispatch.rs arms/flowgpu-vulkan/src/vk_sys.rs
git commit -m "feat(gpu-vm): expose dispatch helpers for VM module"
```

---

## Task 3: VM Builtins in compiler.rs

Add the .flow-callable builtins: `vm_boot`, `vm_write_register`, `vm_read_register`, `vm_shutdown`.

**Files:**
- Modify: `arms/flowgpu-cli/src/compiler.rs` (lines ~98 for thread-local, ~5935 for builtins)

**Step 1: Add VM thread-local state**

After line 97 (after GPU_CACHE_BYTES), add:

```rust
    /// GPU VM instances. Key = VM ID, Value = VmHandle.
    static GPU_VMS: RefCell<HashMap<u32, flowgpu_vulkan::vm::VmHandle>> = RefCell::new(HashMap::new());
    static VM_NEXT_ID: Cell<u32> = Cell::new(1);
```

**Step 2: Add vm_boot builtin**

After the `vm_layer_estimate` handler (~line 5935), add:

```rust
// vm_boot(n_instances, reg_size, globals_size) → VM handle ID
// Allocates DEVICE_LOCAL SSBOs for register file, metrics, globals, control.
if name == "vm_boot" && args.len() == 3 {
    let n_instances = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let reg_size = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let globals_size = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    if device_ptr == 0 {
        return Err(CliError::Runtime("vm_boot: no GPU device initialized".into()));
    }
    let gpu_dev = unsafe { &*(device_ptr as *const flowgpu_vulkan::VulkanCompute) };
    let vm = flowgpu_vulkan::vm::vm_create(gpu_dev, n_instances, reg_size, globals_size)
        .map_err(|e| CliError::Runtime(format!("vm_boot: {}", e)))?;
    let vm_id = VM_NEXT_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
    GPU_VMS.with(|vms| vms.borrow_mut().insert(vm_id, vm));
    return Ok(Value::Float(vm_id as f32));
}
```

**Step 3: Add vm_write_register builtin**

```rust
// vm_write_register(vm_id, instance, reg_idx, array_name) → 0.0 on success
// Writes a .flow array into a VM register SSBO region.
if name == "vm_write_register" && args.len() == 4 {
    let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let instance = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let reg_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    // Fourth arg is an array reference
    let arr_name = if let ScalarExpr::Ref(n) = &args[3] { n.clone() }
        else { return Err(CliError::Compile("vm_write_register: 4th arg must be array name".into())); };
    let data = gpu_array_get(&arr_name, arrays)?;
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    if device_ptr == 0 {
        return Err(CliError::Runtime("vm_write_register: no GPU".into()));
    }
    let gpu_dev = unsafe { &*(device_ptr as *const flowgpu_vulkan::VulkanCompute) };
    GPU_VMS.with(|vms| {
        let vms = vms.borrow();
        let vm = vms.get(&vm_id).ok_or_else(||
            CliError::Runtime(format!("vm_write_register: unknown VM {}", vm_id)))?;
        flowgpu_vulkan::vm::vm_write_reg(gpu_dev, vm, instance, reg_idx, &data)
            .map_err(|e| CliError::Runtime(format!("vm_write_register: {}", e)))
    })?;
    return Ok(Value::Float(0.0));
}
```

**Step 4: Add vm_read_register builtin**

```rust
// vm_read_register(vm_id, instance, reg_idx, count) → stores result in GPU_ARRAYS
// Returns the data as a .flow array.
if name == "vm_read_register" && args.len() == 4 {
    let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let instance = eval_scalar(&args[1], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let reg_idx = eval_scalar(&args[2], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let count = eval_scalar(&args[3], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    if device_ptr == 0 {
        return Err(CliError::Runtime("vm_read_register: no GPU".into()));
    }
    let gpu_dev = unsafe { &*(device_ptr as *const flowgpu_vulkan::VulkanCompute) };
    let data = GPU_VMS.with(|vms| {
        let vms = vms.borrow();
        let vm = vms.get(&vm_id).ok_or_else(||
            CliError::Runtime(format!("vm_read_register: unknown VM {}", vm_id)))?;
        flowgpu_vulkan::vm::vm_read_reg(gpu_dev, vm, instance, reg_idx, count)
            .map_err(|e| CliError::Runtime(format!("vm_read_register: {}", e)))
    })?;
    // Return as array — detected at LetDecl level via eval_array_fn
    return Err(CliError::Compile("vm_read_register must be used with let".into()));
}
```

Note: `vm_read_register` returns an array, so it needs to be handled in `eval_array_fn` (LetDecl level), not `eval_scalar`. Add in the array-returning function section:

```rust
// In eval_array_fn (search for "gguf_load_vocab" to find the right section):
if name == "vm_read_register" && args.len() == 4 {
    let vm_id = eval_scalar(...)?.as_float()? as u32;
    let instance = eval_scalar(...)?.as_float()? as u32;
    let reg_idx = eval_scalar(...)?.as_float()? as u32;
    let count = eval_scalar(...)?.as_float()? as u32;
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    let gpu_dev = unsafe { &*(device_ptr as *const flowgpu_vulkan::VulkanCompute) };
    let data = GPU_VMS.with(|vms| {
        let vms = vms.borrow();
        let vm = vms.get(&vm_id).ok_or_else(||
            CliError::Runtime(format!("vm_read_register: unknown VM {}", vm_id)))?;
        flowgpu_vulkan::vm::vm_read_reg(gpu_dev, vm, instance, reg_idx, count)
            .map_err(|e| CliError::Runtime(format!("vm_read_register: {}", e)))
    })?;
    return Ok(Some(ArrayResult::GpuFloats(data)));
}
```

**Step 5: Add vm_shutdown builtin**

```rust
// vm_shutdown(vm_id) → 0.0
if name == "vm_shutdown" && args.len() == 1 {
    let vm_id = eval_scalar(&args[0], streams, scalars, gpu, arrays, hashmaps, scalar_fns, struct_defs, rng, mutable_scalars)?
        .as_float()? as u32;
    GPU_VMS.with(|vms| {
        let removed = vms.borrow_mut().remove(&vm_id);
        if removed.is_none() {
            return Err(CliError::Runtime(format!("vm_shutdown: unknown VM {}", vm_id)));
        }
        Ok(())
    })?;
    return Ok(Value::Float(0.0));
}
```

**Step 6: Verify build**

Run: `cargo test -p flowgpu-cli 2>&1 | tail -5`

**Step 7: Commit**

```bash
git add arms/flowgpu-cli/src/compiler.rs
git commit -m "feat(gpu-vm): add vm_boot, vm_write/read_register, vm_shutdown builtins"
```

---

## Task 4: Register Builtins in Preflight

**Files:**
- Modify: `arms/flowgpu-cli/src/preflight.rs` (lines 45, 48, 51, 54)

**Step 1: Add to SCALAR_FNS arrays**

```
Line 45 (SCALAR_FNS_0): no additions (vm_boot takes 3 args)
Line 48 (SCALAR_FNS_1): add "vm_shutdown"
Line 51 (SCALAR_FNS_2): no additions (vm_read_register is array-returning)
Line 54 (SCALAR_FNS_3): add "vm_boot", "vm_write_register"
```

For `vm_read_register` (4 args, array-returning), add to the array function validation section. Search for `"gguf_load_vocab"` or `"gguf_tokenize"` in preflight.rs and add `"vm_read_register"` to the same list with 4-arg validation.

**Step 2: Verify build**

Run: `cargo test 2>&1 | tail -10`

**Step 3: Commit**

```bash
git add arms/flowgpu-cli/src/preflight.rs
git commit -m "feat(gpu-vm): register VM builtins in preflight"
```

---

## Task 5: Test Register I/O from .flow

Write a .flow test that boots a VM, writes data to a register, reads it back, and verifies correctness.

**Files:**
- Create: `stdlib/gpu/test_vm_registers.flow`

**Step 1: Write test script**

```flow
// stdlib/gpu/test_vm_registers.flow — GPU VM register I/O test

print("=== GPU VM Register I/O Test ===")

// Boot a VM with 1 instance, 16 floats per register, 1 global
let vm = vm_boot(1.0, 16.0, 1.0)
print("  vm_boot OK, handle={vm}")

// Create test data
let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

// Write to instance 0, register 0
let _w = vm_write_register(vm, 0.0, 0.0, input)
print("  vm_write_register OK (8 floats to R0)")

// Read back from instance 0, register 0, 8 floats
let output = vm_read_register(vm, 0.0, 0.0, 8.0)
print("  vm_read_register OK (8 floats from R0)")

// Verify
let mut pass = 1.0
let mut i = 0.0
while i < 8.0
  let expected = i + 1.0
  let actual = output[int(i)]
  if actual != expected
    print("  FAIL: output[{i}] = {actual}, expected {expected}")
    pass = 0.0
  end
  i = i + 1.0
end

if pass == 1.0
  print("  PASS: all 8 values match")
end

// Write to register 5, read back
let data2 = [100.0, 200.0, 300.0]
let _w2 = vm_write_register(vm, 0.0, 5.0, data2)
let out2 = vm_read_register(vm, 0.0, 5.0, 3.0)
if out2[0] == 100.0
  print("  PASS: register 5 isolation OK")
else
  print("  FAIL: register 5 got {out2[0]}, expected 100")
end

// Shutdown
let _s = vm_shutdown(vm)
print("  vm_shutdown OK")
print("=== DONE ===")
```

**Step 2: Build and run test**

Run: `cargo build 2>&1 | tail -3`
Run: `./target/debug/octoflow.exe run stdlib/gpu/test_vm_registers.flow --allow-ffi`

Expected output:
```
=== GPU VM Register I/O Test ===
  vm_boot OK, handle=1
  vm_write_register OK (8 floats to R0)
  vm_read_register OK (8 floats from R0)
  PASS: all 8 values match
  PASS: register 5 isolation OK
  vm_shutdown OK
=== DONE ===
```

**Step 3: Commit**

```bash
git add stdlib/gpu/test_vm_registers.flow
git commit -m "test(gpu-vm): register I/O roundtrip test"
```

---

## Task 6: Dispatch Chain Builder in vm.rs

Add the ability to build and execute a multi-dispatch command buffer. This is the core execution engine.

**Files:**
- Modify: `arms/flowgpu-vulkan/src/vm.rs`

**Step 1: Add VmProgram struct and builder**

```rust
/// A compiled VM program — a VkCommandBuffer with pre-recorded dispatches.
pub struct VmProgram {
    pub command_buffer: VkCommandBuffer,
    pub command_pool: VkCommandPool,
    pub fence: VkFence,
    pub device: VkDevice,
    pub pipelines: Vec<VkPipeline>,        // Keep alive for command buffer lifetime
    pub pipe_layouts: Vec<VkPipelineLayout>,
    pub ds_layouts: Vec<VkDescriptorSetLayout>,
    pub desc_pool: VkDescriptorPool,
}

/// One dispatch operation in a VM program.
pub struct VmOp {
    pub spirv: Vec<u8>,
    pub push_constants: Vec<f32>,
    pub workgroups: (u32, u32, u32),
}

/// Build a VM program: a command buffer with sequential dispatches + barriers.
/// All dispatches bind the same 5 SSBOs (registers, metrics, heap, globals, control).
pub fn vm_build_program(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    ops: &[VmOp],
) -> Result<VmProgram, VulkanError> {
    // 1. Create command pool + buffer + fence
    // 2. Determine binding count (5 if heap exists, 4 otherwise)
    // 3. For each op:
    //    a. Create/cache pipeline from SPIR-V
    //    b. Create descriptor set binding SSBOs
    //    c. Record: bind pipeline, bind descriptors, push constants, dispatch
    //    d. Record: pipeline barrier (SHADER_WRITE -> SHADER_READ)
    // 4. End command buffer
    // Return VmProgram handle
    todo!()
}

/// Execute a VM program (submit + wait).
pub fn vm_execute_program(
    gpu: &VulkanCompute,
    prog: &VmProgram,
) -> Result<(), VulkanError> {
    // vkResetFences + vkQueueSubmit + vkWaitForFences
    todo!()
}

/// Destroy a VM program, freeing command buffer and cached pipelines.
pub fn vm_destroy_program(prog: VmProgram) {
    // cleanup (or let Drop handle it)
    drop(prog);
}
```

**Step 2: Implement vm_build_program**

This is the most complex function. Key pattern from dispatch.rs (gpu_run_dispatch):

```rust
pub fn vm_build_program(
    gpu: &VulkanCompute,
    vm: &VmHandle,
    ops: &[VmOp],
) -> Result<VmProgram, VulkanError> {
    let has_heap = vm.heap.is_some();
    let num_bindings: u32 = if has_heap { 5 } else { 4 };

    // Create command pool
    let pool_info = VkCommandPoolCreateInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queueFamilyIndex: gpu.queue_family_index,
    };
    let mut cmd_pool: VkCommandPool = 0;
    unsafe { vkCreateCommandPool(gpu.device, &pool_info, std::ptr::null(), &mut cmd_pool); }

    // Allocate command buffer
    let alloc_info = VkCommandBufferAllocateInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        pNext: std::ptr::null(),
        commandPool: cmd_pool,
        level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount: 1,
    };
    let mut cmd_buf: VkCommandBuffer = std::ptr::null_mut();
    unsafe { vkAllocateCommandBuffers(gpu.device, &alloc_info, &mut cmd_buf); }

    // Create fence
    let fence_info = VkFenceCreateInfo {
        sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
    };
    let mut fence: VkFence = 0;
    unsafe { vkCreateFence(gpu.device, &fence_info, std::ptr::null(), &mut fence); }

    // Create descriptor pool (enough for all ops)
    let pool_size = VkDescriptorPoolSize {
        descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount: num_bindings * ops.len() as u32,
    };
    let dp_info = VkDescriptorPoolCreateInfo {
        sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        maxSets: ops.len() as u32,
        poolSizeCount: 1,
        pPoolSizes: &pool_size,
    };
    let mut desc_pool: VkDescriptorPool = 0;
    unsafe { vkCreateDescriptorPool(gpu.device, &dp_info, std::ptr::null(), &mut desc_pool); }

    // Begin command buffer
    let begin_info = VkCommandBufferBeginInfo {
        sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        pInheritanceInfo: std::ptr::null(),
    };
    unsafe { vkBeginCommandBuffer(cmd_buf, &begin_info); }

    let mut pipelines = Vec::new();
    let mut pipe_layouts = Vec::new();
    let mut ds_layouts = Vec::new();

    // Collect SSBO buffers for binding
    // binding 0 = registers, 1 = metrics, 2 = heap (optional), 3 = globals, 4 = control
    // If no heap: 0=registers, 1=metrics, 2=globals, 3=control

    for op in ops {
        // Create descriptor set layout
        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..num_bindings)
            .map(|i| VkDescriptorSetLayoutBinding {
                binding: i,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers: std::ptr::null(),
            }).collect();
        // ... create pipeline, descriptor set, record dispatch + barrier
        // (Full implementation follows existing gpu_run_dispatch pattern)
    }

    unsafe { vkEndCommandBuffer(cmd_buf); }

    Ok(VmProgram {
        command_buffer: cmd_buf,
        command_pool: cmd_pool,
        fence,
        device: gpu.device,
        pipelines,
        pipe_layouts,
        ds_layouts,
        desc_pool,
    })
}
```

**Step 3: Implement vm_execute_program**

```rust
pub fn vm_execute_program(
    gpu: &VulkanCompute,
    prog: &VmProgram,
) -> Result<(), VulkanError> {
    unsafe {
        vkResetFences(gpu.device, 1, &prog.fence);
    }
    let submit = VkSubmitInfo {
        sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
        pNext: std::ptr::null(),
        waitSemaphoreCount: 0,
        pWaitSemaphores: std::ptr::null(),
        pWaitDstStageMask: std::ptr::null(),
        commandBufferCount: 1,
        pCommandBuffers: &prog.command_buffer,
        signalSemaphoreCount: 0,
        pSignalSemaphores: std::ptr::null(),
    };
    unsafe {
        vkQueueSubmit(gpu.compute_queue, 1, &submit, prog.fence);
        vkWaitForFences(gpu.device, 1, &prog.fence, 1, u64::MAX);
    }
    Ok(())
}
```

**Step 4: Verify build**

Run: `cargo test -p flowgpu-vulkan 2>&1 | tail -5`

**Step 5: Commit**

```bash
git add arms/flowgpu-vulkan/src/vm.rs
git commit -m "feat(gpu-vm): dispatch chain builder and executor"
```

---

## Task 7: vm_program and vm_execute Builtins

Wire the dispatch chain builder into compiler.rs builtins.

**Files:**
- Modify: `arms/flowgpu-cli/src/compiler.rs`
- Modify: `arms/flowgpu-cli/src/preflight.rs`

**Step 1: Add VM program thread-local**

After GPU_VMS thread-local:

```rust
    static VM_PROGRAMS: RefCell<HashMap<u32, flowgpu_vulkan::vm::VmProgram>> = RefCell::new(HashMap::new());
    static VM_PROG_NEXT_ID: Cell<u32> = Cell::new(1);
```

**Step 2: Add vm_dispatch builtin (stage a kernel)**

```rust
// vm_dispatch(vm_id, spv_path, push_constants_array, workgroups) → op_id
// Stages a dispatch operation. Accumulates until vm_build is called.
if name == "vm_dispatch" && args.len() == 4 {
    // Parse args: vm_id, spv_path (string), pc_array (array ref), workgroups (float)
    // Store in a staging list on the VmHandle or a separate thread-local
    // Return op index
}
```

**Step 3: Add vm_build builtin (compile staged dispatches into command buffer)**

```rust
// vm_build(vm_id) → program_id
// Builds VkCommandBuffer from all staged dispatches, returns program handle.
if name == "vm_build" && args.len() == 1 {
    let vm_id = eval_scalar(...)?.as_float()? as u32;
    // Read staged ops, call vm_build_program, store in VM_PROGRAMS
    // Return program_id
}
```

**Step 4: Add vm_execute builtin**

```rust
// vm_execute(program_id) → 0.0
// Submits command buffer, waits for completion.
if name == "vm_execute" && args.len() == 1 {
    let prog_id = eval_scalar(...)?.as_float()? as u32;
    let device_ptr = GPU_DEVICE_PTR.with(|c| c.get());
    let gpu_dev = unsafe { &*(device_ptr as *const flowgpu_vulkan::VulkanCompute) };
    VM_PROGRAMS.with(|progs| {
        let progs = progs.borrow();
        let prog = progs.get(&prog_id).ok_or_else(||
            CliError::Runtime(format!("vm_execute: unknown program {}", prog_id)))?;
        flowgpu_vulkan::vm::vm_execute_program(gpu_dev, prog)
            .map_err(|e| CliError::Runtime(format!("vm_execute: {}", e)))
    })?;
    return Ok(Value::Float(0.0));
}
```

**Step 5: Register in preflight**

Add `"vm_dispatch"` to SCALAR_FNS with 4 args, `"vm_build"` and `"vm_execute"` to SCALAR_FNS_1.

**Step 6: Verify build**

Run: `cargo test 2>&1 | tail -10`

**Step 7: Commit**

```bash
git add arms/flowgpu-cli/src/compiler.rs arms/flowgpu-cli/src/preflight.rs
git commit -m "feat(gpu-vm): vm_dispatch, vm_build, vm_execute builtins"
```

---

## Task 8: Emit copy_register Kernel

A minimal SPIR-V kernel that copies one register region to another. This proves the dispatch chain reads/writes SSBOs correctly.

**Files:**
- Create: `stdlib/gpu/emit_vm_copy.flow`
- Output: `stdlib/gpu/kernels/vm_copy.spv`

**Step 1: Write the emitter**

```flow
// stdlib/gpu/emit_vm_copy.flow — emit SPIR-V kernel that copies R[src] to R[dst]
// Binding 0: register SSBO (shared across all instances)
// Push constants: pc[0] = src_offset (in floats), pc[1] = dst_offset, pc[2] = count
use "ir"

let _r = ir_new()
let _ic = ir_set_input_count(1.0)  // 1 SSBO (registers)

let entry = ir_block("entry")

// Load global invocation ID
let gid = ir_load_gid(entry)

// Load push constants
let src_off = ir_push_const(entry, 0.0)   // source offset
let dst_off = ir_push_const(entry, 1.0)   // dest offset
let count = ir_push_const(entry, 2.0)     // element count

// Bounds check: if gid >= count, return
let merge = ir_block("merge")
let body = ir_block("body")
let cond = ir_ult(entry, gid, count)
let _sm = ir_selection_merge(entry, merge)
let _br = ir_term_cond_branch(entry, cond, body, merge)

// body: copy registers[src_off + gid] to registers[dst_off + gid]
let src_idx = ir_iadd(body, src_off, gid)
let val = ir_load_input_at(body, 0.0, src_idx)
let dst_idx = ir_iadd(body, dst_off, gid)
let _st = ir_store_input_at(body, 0.0, dst_idx, val)  // write back to same SSBO
let _br2 = ir_term_branch(body, merge)

let _ret = ir_term_return(merge)

let _emit = ir_emit_spirv("stdlib/gpu/kernels/vm_copy.spv")
print("Emitted vm_copy.spv")
```

**Step 2: Run emitter and validate**

Run: `./target/debug/octoflow.exe run stdlib/gpu/emit_vm_copy.flow --allow-write`
Run: `spirv-val stdlib/gpu/kernels/vm_copy.spv`

Expected: VALID

**Step 3: Commit**

```bash
git add stdlib/gpu/emit_vm_copy.flow stdlib/gpu/kernels/vm_copy.spv
git commit -m "feat(gpu-vm): emit vm_copy.spv kernel for register-to-register copy"
```

---

## Task 9: Matvec Proof — End-to-End VM Test

The milestone test: boot a VM, write input to R0, dispatch a matvec kernel that reads R0 and heap weights, writes result to R1, verify against CPU reference.

**Files:**
- Create: `stdlib/gpu/test_vm_matvec.flow`

**Step 1: Write the end-to-end test**

```flow
// stdlib/gpu/test_vm_matvec.flow — prove GPU VM works with real compute
// Matrix-vector multiply: R1 = W * R0
// W is a 4x4 identity matrix stored in heap
// R0 = [1, 2, 3, 4], expected R1 = [1, 2, 3, 4]

print("=== GPU VM Matvec Test ===")

// Boot VM: 1 instance, 16 floats per register, 16 globals (for weight matrix)
let vm = vm_boot(1.0, 16.0, 16.0)
print("  vm_boot OK")

// Write input vector to R0
let input = [1.0, 2.0, 3.0, 4.0]
let _w = vm_write_register(vm, 0.0, 0.0, input)
print("  R0 = [1, 2, 3, 4]")

// Write identity matrix to globals (heap stand-in for now)
// Row-major 4x4: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
let weights = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
let _wg = vm_write_globals(vm, 0.0, weights)
print("  Globals = 4x4 identity matrix")

// Dispatch matvec kernel
// Push constants: M=4, K=4, reg_r0_offset=0, reg_r1_offset=16
let pc = [4.0, 4.0, 0.0, 16.0]
let _d = vm_dispatch(vm, "stdlib/gpu/kernels/vm_matvec.spv", pc, 1.0)

// Build and execute
let prog = vm_build(vm)
let _e = vm_execute(prog)
print("  vm_execute OK")

// Read result from R1
let result = vm_read_register(vm, 0.0, 1.0, 4.0)
print("  R1 = [{result[0]}, {result[1]}, {result[2]}, {result[3]}]")

// Verify: identity * [1,2,3,4] = [1,2,3,4]
let mut pass = 1.0
let mut i = 0.0
while i < 4.0
  let expected = i + 1.0
  let actual = result[int(i)]
  if actual != expected
    print("  FAIL: R1[{i}] = {actual}, expected {expected}")
    pass = 0.0
  end
  i = i + 1.0
end

if pass == 1.0
  print("  PASS: VM matvec produces correct result")
end

let _s = vm_shutdown(vm)
print("=== DONE ===")
```

**Step 2: Emit vm_matvec.spv kernel**

Create `stdlib/gpu/emit_vm_matvec.flow` that emits a kernel reading from register SSBO (binding 0) and globals SSBO (binding 3), writing result to register SSBO (binding 0) at a different offset.

**Step 3: Build and run**

Run: `cargo build && ./target/debug/octoflow.exe run stdlib/gpu/test_vm_matvec.flow --allow-ffi --allow-read`

Expected:
```
=== GPU VM Matvec Test ===
  vm_boot OK
  R0 = [1, 2, 3, 4]
  Globals = 4x4 identity matrix
  vm_execute OK
  R1 = [1, 2, 3, 4]
  PASS: VM matvec produces correct result
=== DONE ===
```

**Step 4: Test with non-identity matrix**

Change weights to `[2,0,0,0, 0,3,0,0, 0,0,4,0, 0,0,0,5]`
Expected R1 = `[2, 6, 12, 20]`

**Step 5: Commit**

```bash
git add stdlib/gpu/test_vm_matvec.flow stdlib/gpu/emit_vm_matvec.flow stdlib/gpu/kernels/vm_matvec.spv
git commit -m "test(gpu-vm): end-to-end matvec proof with GPU VM"
```

---

## Summary

| Task | What | Files | Commit |
|------|------|-------|--------|
| 1 | VmHandle + SSBO allocation | vm.rs (new), lib.rs | `feat: vm.rs with VmHandle` |
| 2 | Expose dispatch.rs helpers | dispatch.rs, vk_sys.rs | `feat: expose helpers` |
| 3 | VM builtins in compiler | compiler.rs | `feat: vm_boot/write/read/shutdown` |
| 4 | Preflight registration | preflight.rs | `feat: register VM builtins` |
| 5 | Register I/O test | test_vm_registers.flow | `test: register roundtrip` |
| 6 | Dispatch chain builder | vm.rs | `feat: chain builder` |
| 7 | vm_dispatch/build/execute | compiler.rs, preflight.rs | `feat: dispatch builtins` |
| 8 | copy_register kernel | emit_vm_copy.flow | `feat: vm_copy.spv` |
| 9 | Matvec end-to-end test | test_vm_matvec.flow | `test: matvec proof` |

**After Task 9:** The GPU VM foundation is proven. Next plans cover message passing (multi-VM chains), the homeostasis regulator, and LLM mapping.
