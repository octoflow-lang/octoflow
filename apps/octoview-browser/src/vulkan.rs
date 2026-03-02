use ash::vk;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::ffi::CStr;

#[allow(dead_code)]
pub struct VulkanContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    queue_family_index: u32,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

#[allow(dead_code)]
impl VulkanContext {
    pub fn new(window: &winit::window::Window) -> Result<Self, String> {
        unsafe { Self::init(window) }
    }

    unsafe fn init(window: &winit::window::Window) -> Result<Self, String> {
        let entry = ash::Entry::load()
            .map_err(|e| format!("Failed to load Vulkan entry: {e}"))?;

        // Instance with surface extensions
        let app_info = vk::ApplicationInfo::default()
            .application_name(CStr::from_bytes_with_nul(b"OctoView Browser\0").unwrap())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(CStr::from_bytes_with_nul(b"OctoView\0").unwrap())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let extension_names = get_required_extensions(window)?;
        let extension_ptrs: Vec<*const i8> = extension_names
            .iter()
            .map(|s| s.as_ptr())
            .collect();

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_ptrs);

        let instance = entry
            .create_instance(&instance_info, None)
            .map_err(|e| format!("Failed to create Vulkan instance: {e}"))?;

        // Create surface
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = create_surface(&entry, &instance, window)?;

        // Pick physical device
        let (physical_device, queue_family_index) =
            pick_physical_device(&instance, &surface_loader, surface)?;

        // Logical device
        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority);
        let queue_infos = [queue_info];

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions);

        let device = instance
            .create_device(physical_device, &device_info, None)
            .map_err(|e| format!("Failed to create logical device: {e}"))?;

        let graphics_queue = device.get_device_queue(queue_family_index, 0);

        // Swapchain
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let surface_caps = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .map_err(|e| format!("Failed to get surface capabilities: {e}"))?;

        let surface_formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .map_err(|e| format!("Failed to get surface formats: {e}"))?;

        let surface_format = surface_formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surface_formats[0]);

        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: size.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };

        let image_count = if surface_caps.max_image_count > 0 {
            (surface_caps.min_image_count + 1).min(surface_caps.max_image_count)
        } else {
            surface_caps.min_image_count + 1
        };

        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = swapchain_loader
            .create_swapchain(&swapchain_info, None)
            .map_err(|e| format!("Failed to create swapchain: {e}"))?;

        let swapchain_images = swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(|e| format!("Failed to get swapchain images: {e}"))?;

        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                device.create_image_view(&view_info, None).unwrap()
            })
            .collect();

        // Command pool and buffers
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = device
            .create_command_pool(&pool_info, None)
            .map_err(|e| format!("Failed to create command pool: {e}"))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| format!("Failed to allocate command buffers: {e}"))?;

        // Sync primitives
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let image_available_semaphore = device
            .create_semaphore(&semaphore_info, None)
            .map_err(|e| format!("Failed to create semaphore: {e}"))?;
        let render_finished_semaphore = device
            .create_semaphore(&semaphore_info, None)
            .map_err(|e| format!("Failed to create semaphore: {e}"))?;
        let in_flight_fence = device
            .create_fence(&fence_info, None)
            .map_err(|e| format!("Failed to create fence: {e}"))?;

        Ok(VulkanContext {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            graphics_queue,
            queue_family_index,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format: surface_format.format,
            swapchain_extent: extent,
            command_pool,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        })
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    pub fn swapchain_image_views(&self) -> &[vk::ImageView] {
        &self.swapchain_image_views
    }

    pub fn swapchain_image_count(&self) -> usize {
        self.swapchain_images.len()
    }

    /// Acquire next swapchain image. Returns image index.
    pub fn acquire_next_image(&self) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.in_flight_fence])?;

            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphore,
                vk::Fence::null(),
            )
        }
    }

    /// Submit command buffer and present.
    pub fn submit_and_present(&self, image_index: u32) -> Result<(), String> {
        let cmd = self.command_buffers[0];
        let wait_semaphores = [self.image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphore];
        let command_buffers = [cmd];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], self.in_flight_fence)
                .map_err(|e| format!("Queue submit failed: {e}"))?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
                .map_err(|e| format!("Present failed: {e}"))?;
        }

        Ok(())
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffers[0]
    }

    /// Wait for device idle (for cleanup).
    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }

    /// Resize swapchain (call on WindowEvent::Resized).
    pub fn recreate_swapchain(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.wait_idle();

        unsafe {
            // Destroy old image views
            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }

            let surface_caps = self
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .map_err(|e| format!("Failed to get surface capabilities: {e}"))?;

            let extent = vk::Extent2D {
                width: width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            };

            let image_count = if surface_caps.max_image_count > 0 {
                (surface_caps.min_image_count + 1).min(surface_caps.max_image_count)
            } else {
                surface_caps.min_image_count + 1
            };

            let swapchain_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(self.swapchain_format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .old_swapchain(self.swapchain);

            let new_swapchain = self
                .swapchain_loader
                .create_swapchain(&swapchain_info, None)
                .map_err(|e| format!("Failed to recreate swapchain: {e}"))?;

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain = new_swapchain;
            self.swapchain_extent = extent;

            self.swapchain_images = self
                .swapchain_loader
                .get_swapchain_images(self.swapchain)
                .map_err(|e| format!("Failed to get swapchain images: {e}"))?;

            self.swapchain_image_views = self
                .swapchain_images
                .iter()
                .map(|&image| {
                    let view_info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.swapchain_format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    self.device.create_image_view(&view_info, None).unwrap()
                })
                .collect();
        }

        Ok(())
    }

    /// Find a memory type that satisfies the type filter and has the required properties.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }
        Err("No suitable memory type found".to_string())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_fence(self.in_flight_fence, None);

            self.device.destroy_command_pool(self.command_pool, None);

            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// --- Platform-specific helpers ---

fn get_required_extensions(
    _window: &winit::window::Window,
) -> Result<Vec<&'static CStr>, String> {
    Ok(vec![
        ash::khr::surface::NAME,
        ash::khr::win32_surface::NAME,
    ])
}

unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, String> {
    let window_handle = window
        .window_handle()
        .map_err(|e| format!("Failed to get window handle: {e}"))?;

    match window_handle.as_raw() {
        RawWindowHandle::Win32(handle) => {
            let win32_loader = ash::khr::win32_surface::Instance::new(entry, instance);
            let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                .hwnd(handle.hwnd.get() as isize)
                .hinstance(
                    handle
                        .hinstance
                        .map(|h| h.get() as isize)
                        .unwrap_or(0),
                );
            win32_loader
                .create_win32_surface(&create_info, None)
                .map_err(|e| format!("Failed to create Win32 surface: {e}"))
        }
        _ => Err("Unsupported window handle type (expected Win32)".to_string()),
    }
}

unsafe fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), String> {
    let devices = instance
        .enumerate_physical_devices()
        .map_err(|e| format!("Failed to enumerate GPUs: {e}"))?;

    if devices.is_empty() {
        return Err("No Vulkan-capable GPU found".to_string());
    }

    // Prefer discrete GPU
    let mut best: Option<(vk::PhysicalDevice, u32, bool)> = None;

    for &pd in &devices {
        let props = instance.get_physical_device_properties(pd);
        let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

        let queue_families = instance.get_physical_device_queue_family_properties(pd);
        for (i, qf) in queue_families.iter().enumerate() {
            let i = i as u32;
            let supports_graphics = qf.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_present = surface_loader
                .get_physical_device_surface_support(pd, i, surface)
                .unwrap_or(false);

            if supports_graphics && supports_present {
                match best {
                    None => best = Some((pd, i, is_discrete)),
                    Some((_, _, prev_discrete)) if !prev_discrete && is_discrete => {
                        best = Some((pd, i, is_discrete));
                    }
                    _ => {}
                }
            }
        }
    }

    best.map(|(pd, qi, _)| (pd, qi))
        .ok_or_else(|| "No GPU with graphics + present support found".to_string())
}
