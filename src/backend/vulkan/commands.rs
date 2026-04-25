//! One-shot command-buffer helper for staging copies.
//!
//! Phase 1 / Step 1.3: `CommandContext` owns a transient command pool
//! pinned to the compute queue family. `one_shot` allocates a single
//! primary command buffer, runs the caller's recording closure inside
//! `vkBeginCommandBuffer` / `vkEndCommandBuffer`, submits with a
//! one-shot fence, waits for completion, and frees the command buffer.
//!
//! Sized for upload/readback only — Step 1.4 will add a longer-lived
//! recording path for the actual dispatch.

use ash::vk;

pub struct CommandContext {
    pub pool: vk::CommandPool,
}

impl CommandContext {
    pub fn new(device: &ash::Device, queue_family: u32) -> Result<Self, vk::Result> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.create_command_pool(&info, None)? };
        Ok(Self { pool })
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe { device.destroy_command_pool(self.pool, None) };
    }

    pub fn one_shot<F>(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        record: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce(vk::CommandBuffer),
    {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmds = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let cmd = cmds[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin)? };
        record(cmd);
        unsafe { device.end_command_buffer(cmd)? };

        let cmds_arr = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds_arr);
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let result: Result<(), Box<dyn std::error::Error>> = (|| unsafe {
            device.queue_submit(queue, std::slice::from_ref(&submit), fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            Ok(())
        })();
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.pool, &cmds_arr);
        }
        result
    }
}
