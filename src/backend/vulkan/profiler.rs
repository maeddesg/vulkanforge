//! GPU-side shader profiler — wraps a `VkQueryPool` of TIMESTAMP
//! queries and exposes a `begin(name) / end()` API around each
//! dispatch. Aggregation by name is left to the caller via
//! [`ShaderProfiler::collect`].
//!
//! Phase 2C. The pool is sized at construction; if you call
//! `begin/end` more than `capacity_pairs` times the extra calls panic
//! — preferable to silently overflowing into another query's slot.

use std::collections::BTreeMap;
use std::time::Duration;

use ash::vk;

#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub name: String,
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone)]
pub struct TimingSample {
    pub name: String,
    pub elapsed: Duration,
}

pub struct ShaderProfiler {
    query_pool: vk::QueryPool,
    pub timestamp_period_ns: f64,
    pub valid_bits: u32,
    pub entries: Vec<ProfileEntry>,
    pub next_query: u32,
    capacity: u32,
}

impl ShaderProfiler {
    pub fn entries_len(&self) -> usize {
        self.entries.len()
    }
}

impl ShaderProfiler {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        device: &ash::Device,
        capacity_pairs: u32,
    ) -> Result<Self, vk::Result> {
        let total = capacity_pairs * 2;
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(total);
        let query_pool = unsafe { device.create_query_pool(&info, None)? };

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let queue_props = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        Ok(Self {
            query_pool,
            timestamp_period_ns: props.limits.timestamp_period as f64,
            valid_bits: queue_props[queue_family as usize].timestamp_valid_bits,
            entries: Vec::with_capacity(capacity_pairs as usize),
            next_query: 0,
            capacity: total,
        })
    }

    /// Record `vkCmdResetQueryPool` for every slot — must be called
    /// before any `begin()`s in the same command buffer.
    pub fn reset(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        unsafe { device.cmd_reset_query_pool(cmd, self.query_pool, 0, self.capacity) };
        self.entries.clear();
        self.next_query = 0;
    }

    pub fn begin(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        name: impl Into<String>,
    ) -> ProfileToken {
        if self.next_query + 2 > self.capacity {
            panic!(
                "ShaderProfiler: capacity ({} pairs) exhausted",
                self.capacity / 2
            );
        }
        let start = self.next_query;
        let end = self.next_query + 1;
        self.next_query += 2;
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                start,
            );
        }
        self.entries.push(ProfileEntry {
            name: name.into(),
            start,
            end,
        });
        ProfileToken { end }
    }

    pub fn end(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, token: ProfileToken) {
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                token.end,
            );
        }
    }

    /// Read the query results back. Caller must have submitted the
    /// command buffer + waited on the fence.
    pub fn collect(&self, device: &ash::Device) -> Result<Vec<TimingSample>, vk::Result> {
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }
        let mut data = vec![0u64; self.next_query as usize];
        unsafe {
            device.get_query_pool_results(
                self.query_pool,
                0,
                &mut data,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let mut out = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            let ticks = data[e.end as usize].wrapping_sub(data[e.start as usize]);
            let ns = (ticks as f64) * self.timestamp_period_ns;
            out.push(TimingSample {
                name: e.name.clone(),
                elapsed: Duration::from_nanos(ns as u64),
            });
        }
        Ok(out)
    }

    pub fn aggregate(samples: &[TimingSample]) -> BTreeMap<String, (Duration, u32)> {
        let mut map: BTreeMap<String, (Duration, u32)> = BTreeMap::new();
        for s in samples {
            let entry = map.entry(s.name.clone()).or_insert((Duration::ZERO, 0));
            entry.0 += s.elapsed;
            entry.1 += 1;
        }
        map
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe { device.destroy_query_pool(self.query_pool, None) };
    }
}

/// Returned by `begin()`, consumed by `end()` so the two calls stay
/// in lockstep at the type level.
#[must_use]
pub struct ProfileToken {
    end: u32,
}
