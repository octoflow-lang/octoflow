use ash::vk;

/// All GPU pipeline resources: render pass, framebuffers, graphics pipeline,
/// texture, staging buffer, descriptor sets.
pub struct RenderPipeline {
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    // Texture
    texture_image: vk::Image,
    texture_memory: vk::DeviceMemory,
    texture_view: vk::ImageView,
    sampler: vk::Sampler,
    // Staging buffer for CPU -> GPU upload
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_size: u64,
    // Texture dimensions
    pub tex_width: u32,
    pub tex_height: u32,
}

impl RenderPipeline {
    pub fn new(ctx: &crate::vulkan::VulkanContext, width: u32, height: u32) -> Result<Self, String> {
        let device = ctx.device();
        unsafe { Self::init(ctx, device, width, height) }
    }

    unsafe fn init(
        ctx: &crate::vulkan::VulkanContext,
        device: &ash::Device,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        // --- Render pass ---
        let color_attachment = vk::AttachmentDescription::default()
            .format(ctx.swapchain_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_refs = [color_ref];

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);
        let subpasses = [subpass];
        let attachments = [color_attachment];

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        let dependencies = [dependency];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create render pass: {e}"))?;

        // --- Framebuffers ---
        let framebuffers: Vec<vk::Framebuffer> = ctx
            .swapchain_image_views()
            .iter()
            .map(|&view| {
                let attachments = [view];
                let fb_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(ctx.swapchain_extent.width)
                    .height(ctx.swapchain_extent.height)
                    .layers(1);
                device.create_framebuffer(&fb_info, None).unwrap()
            })
            .collect();

        // --- Descriptor set layout (one sampler at binding 0) ---
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [binding];

        let ds_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = device
            .create_descriptor_set_layout(&ds_layout_info, None)
            .map_err(|e| format!("Failed to create descriptor set layout: {e}"))?;

        // --- Pipeline layout ---
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {e}"))?;

        // --- Shader modules (inline SPIR-V) ---
        let vert_spv = load_vert_spirv();
        let frag_spv = load_frag_spirv();

        let vert_module = create_shader_module(device, &vert_spv)?;
        let frag_module = create_shader_module(device, &frag_spv)?;

        let entry_name = c"main";
        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name);
        let frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name);
        let stages = [vert_stage, frag_stage];

        // No vertex input (fullscreen triangle generated in shader)
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: ctx.swapchain_extent.width as f32,
            height: ctx.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: ctx.swapchain_extent,
        };
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let blend_attachments = [blend_attachment];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_attachments);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blend)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_pipelines, e)| format!("Failed to create graphics pipeline: {e}"))?;
        let pipeline = pipelines[0];

        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        // --- Texture (R8G8B8A8_SRGB) ---
        let tex_format = vk::Format::R8G8B8A8_SRGB;
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(tex_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let texture_image = device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create texture image: {e}"))?;

        let mem_reqs = device.get_image_memory_requirements(texture_image);
        let mem_type = ctx.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);
        let texture_memory = device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("Failed to allocate texture memory: {e}"))?;
        device
            .bind_image_memory(texture_image, texture_memory, 0)
            .map_err(|e| format!("Failed to bind texture memory: {e}"))?;

        // Texture image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(texture_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(tex_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let texture_view = device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create texture view: {e}"))?;

        // Sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create sampler: {e}"))?;

        // --- Staging buffer ---
        let staging_size = (width * height * 4) as u64;
        let buf_info = vk::BufferCreateInfo::default()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = device
            .create_buffer(&buf_info, None)
            .map_err(|e| format!("Failed to create staging buffer: {e}"))?;
        let buf_mem_reqs = device.get_buffer_memory_requirements(staging_buffer);
        let buf_mem_type = ctx.find_memory_type(
            buf_mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let buf_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_mem_reqs.size)
            .memory_type_index(buf_mem_type);
        let staging_memory = device
            .allocate_memory(&buf_alloc, None)
            .map_err(|e| format!("Failed to allocate staging memory: {e}"))?;
        device
            .bind_buffer_memory(staging_buffer, staging_memory, 0)
            .map_err(|e| format!("Failed to bind staging buffer: {e}"))?;

        // --- Descriptor pool + set ---
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| format!("Failed to create descriptor pool: {e}"))?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("Failed to allocate descriptor set: {e}"))?;
        let descriptor_set = descriptor_sets[0];

        // Update descriptor set to point at our texture
        let image_info_desc = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(texture_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let image_infos = [image_info_desc];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);
        device.update_descriptor_sets(&[write], &[]);

        // Transition texture to SHADER_READ_ONLY initially (so first frame doesn't crash)
        transition_image_layout(
            ctx,
            texture_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        Ok(RenderPipeline {
            render_pass,
            framebuffers,
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            texture_image,
            texture_memory,
            texture_view,
            sampler,
            staging_buffer,
            staging_memory,
            staging_size,
            tex_width: width,
            tex_height: height,
        })
    }

    /// Upload RGBA pixels from CPU to the GPU texture.
    pub fn upload_framebuffer(
        &self,
        ctx: &crate::vulkan::VulkanContext,
        pixels: &[u8],
    ) -> Result<(), String> {
        let device = ctx.device();
        let expected = (self.tex_width * self.tex_height * 4) as usize;
        if pixels.len() != expected {
            return Err(format!(
                "Pixel buffer size mismatch: got {} expected {}",
                pixels.len(),
                expected
            ));
        }

        unsafe {
            // Copy to staging buffer
            let ptr = device
                .map_memory(
                    self.staging_memory,
                    0,
                    self.staging_size,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| format!("Failed to map staging memory: {e}"))?;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr as *mut u8, pixels.len());
            device.unmap_memory(self.staging_memory);

            // Transition image to TRANSFER_DST
            transition_image_layout(
                ctx,
                self.texture_image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )?;

            // Copy buffer to image
            let cmd = begin_single_time_commands(ctx)?;
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: self.tex_width,
                    height: self.tex_height,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffer,
                self.texture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            end_single_time_commands(ctx, cmd)?;

            // Transition back to SHADER_READ_ONLY
            transition_image_layout(
                ctx,
                self.texture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?;
        }

        Ok(())
    }

    /// Record render commands into the given command buffer.
    pub fn render_frame(
        &self,
        ctx: &crate::vulkan::VulkanContext,
        image_index: u32,
    ) -> Result<(), String> {
        let device = ctx.device();
        let cmd = ctx.command_buffer();

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Failed to begin command buffer: {e}"))?;

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.07, 0.07, 0.12, 1.0], // #12121e
                },
            };
            let clear_values = [clear_value];

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: ctx.swapchain_extent,
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(
                cmd,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            // Draw fullscreen triangle (3 vertices, generated in vertex shader)
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);

            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("Failed to end command buffer: {e}"))?;
        }

        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.texture_view, None);
            device.destroy_image(self.texture_image, None);
            device.free_memory(self.texture_memory, None);
            device.destroy_buffer(self.staging_buffer, None);
            device.free_memory(self.staging_memory, None);

            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);

            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            device.destroy_render_pass(self.render_pass, None);
        }
    }
}

// --- Helper functions ---

unsafe fn create_shader_module(
    device: &ash::Device,
    code: &[u32],
) -> Result<vk::ShaderModule, String> {
    let info = vk::ShaderModuleCreateInfo::default().code(code);
    device
        .create_shader_module(&info, None)
        .map_err(|e| format!("Failed to create shader module: {e}"))
}

unsafe fn begin_single_time_commands(
    ctx: &crate::vulkan::VulkanContext,
) -> Result<vk::CommandBuffer, String> {
    let device = ctx.device();
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool())
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let bufs = device
        .allocate_command_buffers(&alloc_info)
        .map_err(|e| format!("Failed to allocate temp command buffer: {e}"))?;
    let cmd = bufs[0];
    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device
        .begin_command_buffer(cmd, &begin_info)
        .map_err(|e| format!("Failed to begin temp command buffer: {e}"))?;
    Ok(cmd)
}

unsafe fn end_single_time_commands(
    ctx: &crate::vulkan::VulkanContext,
    cmd: vk::CommandBuffer,
) -> Result<(), String> {
    let device = ctx.device();
    device
        .end_command_buffer(cmd)
        .map_err(|e| format!("Failed to end temp command buffer: {e}"))?;
    let bufs = [cmd];
    let submit_info = vk::SubmitInfo::default().command_buffers(&bufs);
    device
        .queue_submit(ctx.graphics_queue(), &[submit_info], vk::Fence::null())
        .map_err(|e| format!("Failed to submit temp command: {e}"))?;
    device
        .queue_wait_idle(ctx.graphics_queue())
        .map_err(|e| format!("Failed to wait for queue: {e}"))?;
    device.free_command_buffers(ctx.command_pool(), &bufs);
    Ok(())
}

unsafe fn transition_image_layout(
    ctx: &crate::vulkan::VulkanContext,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<(), String> {
    let cmd = begin_single_time_commands(ctx)?;
    let device = ctx.device();

    let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        ),
        _ => {
            return Err(format!(
                "Unsupported layout transition: {old_layout:?} -> {new_layout:?}"
            ));
        }
    };

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    device.cmd_pipeline_barrier(
        cmd,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );

    end_single_time_commands(ctx, cmd)
}

// --- SPIR-V shaders (pre-compiled from GLSL via glslangValidator) ---

fn load_vert_spirv() -> Vec<u32> {
    let bytes = include_bytes!("../shaders/fullscreen.vert.spv");
    bytes_to_u32(bytes)
}

fn load_frag_spirv() -> Vec<u32> {
    let bytes = include_bytes!("../shaders/fullscreen.frag.spv");
    bytes_to_u32(bytes)
}

fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "SPIR-V must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
