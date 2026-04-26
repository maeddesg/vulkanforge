//! One-shot command-buffer helper for dispatches and uploads.
//!
//! Phase 3C: the inner buffer + fence are now **persistent**. Each
//! `one_shot` call resets and reuses the same `vk::CommandBuffer` and
//! `vk::Fence` instead of allocating/destroying per dispatch — Phase
//! 3A's profiler showed that allocate / free / create / destroy
//! cycle was 2-4 ms of fixed overhead per forward pass, and forward
//! passes happen up to 60+ times per second during decode.
//!
//! Pool now uses `RESET_COMMAND_BUFFER` so the per-call
//! `vkResetCommandBuffer` is legal; the `TRANSIENT` hint is gone for
//! the same reason.

use ash::vk;

pub struct CommandContext {
    pub pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl CommandContext {
    pub fn new(device: &ash::Device, queue_family: u32) -> Result<Self, vk::Result> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmds = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let cmd = cmds[0];

        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };

        Ok(Self { pool, cmd, fence })
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_fence(self.fence, None);
            // The command buffer is freed implicitly by destroying
            // the pool — no explicit `free_command_buffers` needed.
            device.destroy_command_pool(self.pool, None);
        }
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
        let cmd = self.cmd;
        let fence = self.fence;

        unsafe {
            // Reset command buffer + fence for this dispatch.
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.reset_fences(std::slice::from_ref(&fence))?;
        }

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin)? };
        record(cmd);
        unsafe { device.end_command_buffer(cmd)? };

        let cmds_arr = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds_arr);
        unsafe {
            device.queue_submit(queue, std::slice::from_ref(&submit), fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    }
}
