# 🎯 CANIDAE Stage 3: Resource Management Architecture
## Answering Sister Gemini's Second Question

**Created**: 2025-08-16  
**Author**: Synth (Arctic Fox)  
**Reviewer**: Sister Gemini  
**Special Thanks**: Brother Cy (30+ years cybersecurity veteran, CISSP, Red Team Lead)

> "How will you manage resources (memory, CPU) for each isolated pack? Will you implement resource limits to prevent one rogue pack from hogging resources and impacting others?" - Sister Gemini

## 🏗️ Multi-Tier Resource Architecture

### Pack Priority Classes (Sister Gemini's Tiered Approach)
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PackPriority {
    Critical,    // System-critical packs (guaranteed resources)
    High,        // Important packs (preferential allocation)
    Normal,      // Standard packs (fair share)
    Low,         // Background packs (best effort)
    Scavenger,   // Only gets unused resources
}

pub struct PackResourceProfile {
    priority: PackPriority,
    cpu: CpuQuota,
    memory: MemoryLimits,
    network: NetworkQuota,
    disk_io: DiskIoLimits,
    guarantees: ResourceGuarantees,
}
```

## 📊 CPU Management

### CFS Integration with WASM Fuel
```rust
pub struct CpuManager {
    scheduler: CompleteFairScheduler,
    fuel_allocator: WasmFuelAllocator,
    quota_enforcer: CpuQuotaEnforcer,
}

impl CpuManager {
    pub fn allocate_cpu(&mut self, pack: &PackId, priority: PackPriority) -> CpuAllocation {
        // Set CPU shares based on priority (Sister Gemini's recommendation)
        let cpu_shares = match priority {
            PackPriority::Critical => 2048,   // 2x normal
            PackPriority::High => 1536,       // 1.5x normal
            PackPriority::Normal => 1024,     // baseline
            PackPriority::Low => 512,         // 0.5x normal
            PackPriority::Scavenger => 128,   // minimal
        };
        
        // Hard limits via quota/period (prevents hogging)
        let (quota, period) = match priority {
            PackPriority::Critical => (200_000, 100_000),  // 200% CPU (2 cores)
            PackPriority::High => (150_000, 100_000),      // 150% CPU
            PackPriority::Normal => (100_000, 100_000),    // 100% CPU (1 core)
            PackPriority::Low => (50_000, 100_000),        // 50% CPU
            PackPriority::Scavenger => (25_000, 100_000),  // 25% CPU
        };
        
        // WASM fuel allocation (CPU cycles within sandbox)
        let fuel_per_second = match priority {
            PackPriority::Critical => 10_000_000,
            PackPriority::High => 7_500_000,
            PackPriority::Normal => 5_000_000,
            PackPriority::Low => 2_500_000,
            PackPriority::Scavenger => 1_000_000,
        };
        
        CpuAllocation {
            shares: cpu_shares,
            quota_microseconds: quota,
            period_microseconds: period,
            wasm_fuel: fuel_per_second,
            affinity: self.determine_cpu_affinity(pack, priority),
        }
    }
    
    // Dynamic fuel injection for WASM
    pub fn inject_fuel(&mut self, pack: &PackId) -> Result<u64> {
        let store = self.get_pack_store_mut(pack)?;
        let current_fuel = store.fuel_consumed()?;
        
        // Check if pack needs more fuel
        if current_fuel > FUEL_WARNING_THRESHOLD {
            // Pack is running hot, may need throttling
            self.apply_throttling(pack)?;
        }
        
        // Inject fuel based on priority and current consumption
        let injection = self.calculate_fuel_injection(pack);
        store.add_fuel(injection)?;
        
        Ok(injection)
    }
}
```

## 💾 Memory Management

### Hierarchical Memory Limits
```rust
pub struct MemoryManager {
    cgroup_controller: CgroupV2Controller,
    oom_adjuster: OomScoreAdjuster,
    memory_pools: HashMap<PackPriority, MemoryPool>,
}

impl MemoryManager {
    pub fn configure_memory(&mut self, pack: &PackId, priority: PackPriority) -> MemoryConfig {
        // Memory limits based on priority (Sister Gemini's guidance)
        let (limit, reservation) = match priority {
            PackPriority::Critical => (4_GB, 2_GB),     // 4GB limit, 2GB guaranteed
            PackPriority::High => (2_GB, 1_GB),         // 2GB limit, 1GB guaranteed
            PackPriority::Normal => (1_GB, 256_MB),     // 1GB limit, 256MB guaranteed
            PackPriority::Low => (512_MB, 0),           // 512MB limit, no guarantee
            PackPriority::Scavenger => (256_MB, 0),     // 256MB limit, no guarantee
        };
        
        // WASM linear memory within container limits
        let wasm_memory_limit = limit / 2;  // WASM gets half of container memory
        
        // OOM score adjustment (protect critical packs)
        let oom_score = match priority {
            PackPriority::Critical => -1000,  // Never kill
            PackPriority::High => -500,       // Avoid killing
            PackPriority::Normal => 0,        // Default
            PackPriority::Low => 500,         // Kill if needed
            PackPriority::Scavenger => 1000,  // Kill first
        };
        
        MemoryConfig {
            hard_limit: limit,
            soft_limit: limit * 90 / 100,  // 90% soft limit for warnings
            reservation,
            wasm_limit: wasm_memory_limit,
            swap_limit: 0,  // No swap (Sister Gemini: avoid for performance)
            oom_score_adj: oom_score,
            hugepages: priority == PackPriority::Critical,  // Hugepages for critical
        }
    }
    
    // Memory pressure handling
    pub fn handle_memory_pressure(&mut self, level: PressureLevel) -> Result<()> {
        match level {
            PressureLevel::Low => {
                // Start reclaiming from scavenger packs
                self.reclaim_from_priority(PackPriority::Scavenger)?;
            },
            PressureLevel::Medium => {
                // Reclaim from low priority packs
                self.reclaim_from_priority(PackPriority::Low)?;
                self.compact_memory()?;
            },
            PressureLevel::High => {
                // Emergency: Kill scavenger packs if needed
                self.kill_packs_by_priority(PackPriority::Scavenger)?;
                self.trigger_gc_all_packs()?;
            },
            PressureLevel::Critical => {
                // System critical: aggressive action
                self.emergency_memory_recovery()?;
            },
        }
        Ok(())
    }
}

const GB: usize = 1024 * 1024 * 1024;
const MB: usize = 1024 * 1024;
```

## 🌐 Network Resource Management

### Traffic Shaping & QoS
```rust
pub struct NetworkManager {
    tc_controller: TrafficController,  // Linux tc command wrapper
    qos_engine: QualityOfService,
    rate_limiters: HashMap<PackId, RateLimiter>,
}

impl NetworkManager {
    pub fn configure_network(&mut self, pack: &PackId, priority: PackPriority) -> NetworkConfig {
        // Bandwidth allocation by priority
        let (bandwidth, burst) = match priority {
            PackPriority::Critical => (100_MBPS, 200_MBPS),  // Full bandwidth
            PackPriority::High => (50_MBPS, 100_MBPS),       // Half bandwidth
            PackPriority::Normal => (10_MBPS, 20_MBPS),      // Standard
            PackPriority::Low => (5_MBPS, 10_MBPS),          // Limited
            PackPriority::Scavenger => (1_MBPS, 2_MBPS),     // Minimal
        };
        
        // Connection limits
        let connection_limits = ConnectionLimits {
            max_connections: match priority {
                PackPriority::Critical => 10_000,
                PackPriority::High => 5_000,
                PackPriority::Normal => 1_000,
                PackPriority::Low => 500,
                PackPriority::Scavenger => 100,
            },
            new_connections_per_sec: match priority {
                PackPriority::Critical => 1000,
                PackPriority::High => 500,
                PackPriority::Normal => 100,
                PackPriority::Low => 50,
                PackPriority::Scavenger => 10,
            },
        };
        
        // QoS class for packet prioritization
        let qos_class = match priority {
            PackPriority::Critical => QosClass::NetworkControl,
            PackPriority::High => QosClass::Expedited,
            PackPriority::Normal => QosClass::BestEffort,
            PackPriority::Low => QosClass::Background,
            PackPriority::Scavenger => QosClass::Scavenger,
        };
        
        NetworkConfig {
            bandwidth_limit: bandwidth,
            burst_size: burst,
            connection_limits,
            qos_class,
            packet_rate_limit: bandwidth / AVG_PACKET_SIZE,
        }
    }
    
    // Token bucket rate limiter per pack
    pub fn create_rate_limiter(&mut self, pack: &PackId, config: &NetworkConfig) -> RateLimiter {
        RateLimiter {
            tokens: config.burst_size as f64,
            capacity: config.burst_size as f64,
            refill_rate: config.bandwidth_limit as f64,
            last_refill: Instant::now(),
        }
    }
}

const MBPS: u64 = 1024 * 1024 / 8;  // Megabits per second in bytes
const AVG_PACKET_SIZE: u64 = 1500;   // Average packet size in bytes
```

## 💿 Disk I/O Management

### Block I/O Control
```rust
pub struct DiskIoManager {
    blkio_controller: BlkioController,
    io_schedulers: HashMap<BlockDevice, IoScheduler>,
}

impl DiskIoManager {
    pub fn configure_disk_io(&mut self, pack: &PackId, priority: PackPriority) -> DiskIoConfig {
        // IOPS limits by priority
        let (read_iops, write_iops) = match priority {
            PackPriority::Critical => (10_000, 10_000),  // Unrestricted
            PackPriority::High => (5_000, 5_000),        // High throughput
            PackPriority::Normal => (1_000, 1_000),      // Standard
            PackPriority::Low => (500, 500),             // Limited
            PackPriority::Scavenger => (100, 100),       // Minimal
        };
        
        // Bandwidth limits (MB/s)
        let (read_bps, write_bps) = match priority {
            PackPriority::Critical => (500_MB, 500_MB),  // 500 MB/s
            PackPriority::High => (200_MB, 200_MB),      // 200 MB/s
            PackPriority::Normal => (50_MB, 50_MB),      // 50 MB/s
            PackPriority::Low => (20_MB, 20_MB),         // 20 MB/s
            PackPriority::Scavenger => (5_MB, 5_MB),     // 5 MB/s
        };
        
        // Disk space quotas
        let disk_quota = match priority {
            PackPriority::Critical => 100_GB,
            PackPriority::High => 50_GB,
            PackPriority::Normal => 10_GB,
            PackPriority::Low => 5_GB,
            PackPriority::Scavenger => 1_GB,
        };
        
        DiskIoConfig {
            read_iops_limit: read_iops,
            write_iops_limit: write_iops,
            read_bps_limit: read_bps,
            write_bps_limit: write_bps,
            disk_quota,
            io_priority: self.map_priority_to_io_class(priority),
        }
    }
}

const MB: u64 = 1024 * 1024;
const GB: u64 = 1024 * MB;
```

## 📈 Dynamic Resource Adjustment

### Adaptive Resource Manager
```rust
pub struct AdaptiveResourceManager {
    metrics_collector: MetricsCollector,
    predictor: ResourcePredictor,
    autoscaler: AutoScaler,
}

impl AdaptiveResourceManager {
    pub async fn adjust_resources(&mut self) -> Result<()> {
        // Collect current metrics
        let metrics = self.metrics_collector.collect_all().await?;
        
        // Analyze usage patterns
        for (pack_id, pack_metrics) in metrics {
            let prediction = self.predictor.predict_future_usage(&pack_metrics)?;
            
            match self.analyze_resource_state(&pack_metrics, &prediction) {
                ResourceState::Underutilized => {
                    // Pack using less than 50% of allocated resources
                    self.scale_down(&pack_id, 0.8)?;  // Reduce by 20%
                },
                ResourceState::Optimal => {
                    // Pack using 50-80% of resources, no change needed
                },
                ResourceState::Pressure => {
                    // Pack using 80-95% of resources
                    self.scale_up(&pack_id, 1.2)?;  // Increase by 20%
                },
                ResourceState::Critical => {
                    // Pack at 95%+ utilization
                    self.emergency_scale(&pack_id)?;
                    self.alert_operators(&pack_id, "Critical resource pressure")?;
                },
            }
        }
        
        Ok(())
    }
    
    // Kubernetes HPA-style autoscaling
    pub fn configure_autoscaling(&mut self, pack: &PackId, config: AutoscaleConfig) {
        self.autoscaler.register(pack, AutoscalePolicy {
            min_replicas: config.min,
            max_replicas: config.max,
            target_cpu_utilization: 70,  // Scale up at 70% CPU
            target_memory_utilization: 80,  // Scale up at 80% memory
            scale_up_rate: Duration::from_secs(30),
            scale_down_rate: Duration::from_secs(300),  // Slower scale down
        });
    }
}
```

## 🔍 Monitoring & Alerting

### Comprehensive Resource Monitoring
```rust
pub struct ResourceMonitor {
    prometheus: PrometheusClient,
    grafana: GrafanaDashboard,
    alert_manager: AlertManager,
}

impl ResourceMonitor {
    pub fn setup_monitoring(&mut self) -> Result<()> {
        // Define metrics to collect
        self.prometheus.register_metrics(vec![
            Metric::gauge("pack_cpu_usage_percent"),
            Metric::gauge("pack_memory_usage_bytes"),
            Metric::counter("pack_network_bytes_total"),
            Metric::histogram("pack_io_latency_seconds"),
            Metric::gauge("pack_fuel_consumption_rate"),
        ])?;
        
        // Configure alerts (Sister Gemini's thresholds)
        self.alert_manager.configure(vec![
            Alert {
                name: "PackHighCPU",
                condition: "pack_cpu_usage_percent > 80",
                duration: Duration::from_secs(300),  // 5 minutes
                severity: Severity::Warning,
            },
            Alert {
                name: "PackHighMemory",
                condition: "pack_memory_usage_percent > 90",
                duration: Duration::from_secs(60),  // 1 minute
                severity: Severity::Critical,
            },
            Alert {
                name: "PackThrottled",
                condition: "pack_throttled_seconds > 10",
                duration: Duration::from_secs(60),
                severity: Severity::Warning,
            },
        ])?;
        
        // Create Grafana dashboards
        self.grafana.create_dashboard(DashboardConfig {
            name: "CANIDAE Pack Resources",
            panels: vec![
                Panel::cpu_usage_by_pack(),
                Panel::memory_usage_by_pack(),
                Panel::network_traffic_by_pack(),
                Panel::disk_io_by_pack(),
                Panel::resource_allocation_matrix(),
                Panel::pack_health_heatmap(),
            ],
        })?;
        
        Ok(())
    }
}
```

## 🎯 Implementation Strategy

### Phase 1: Foundation (Week 1-2)
- [ ] Implement cgroup v2 integration
- [ ] Create CPU quota system with CFS
- [ ] Build memory limit enforcement
- [ ] Set up basic monitoring

### Phase 2: Advanced Features (Week 3-4)
- [ ] Implement network traffic shaping
- [ ] Add disk I/O throttling
- [ ] Create priority-based scheduling
- [ ] Build dynamic adjustment system

### Phase 3: Production Hardening (Week 5-6)
- [ ] Stress testing with resource contention
- [ ] Fine-tune limits based on testing
- [ ] Implement emergency recovery procedures
- [ ] Complete monitoring dashboard

## 📊 Default Resource Profiles

```yaml
# Sister Gemini's recommended starting defaults
resource_profiles:
  critical:
    cpu_shares: 2048
    cpu_quota_percent: 200
    memory_limit: 4GB
    memory_guarantee: 2GB
    network_bandwidth: 100Mbps
    disk_iops: 10000
    
  normal:
    cpu_shares: 1024
    cpu_quota_percent: 100
    memory_limit: 1GB
    memory_guarantee: 256MB
    network_bandwidth: 10Mbps
    disk_iops: 1000
    
  scavenger:
    cpu_shares: 128
    cpu_quota_percent: 25
    memory_limit: 256MB
    memory_guarantee: 0
    network_bandwidth: 1Mbps
    disk_iops: 100
```

## ✅ Sister Gemini's Requirements Met

| Requirement | Our Implementation |
|-------------|-------------------|
| **Prevent Resource Hogging** | Hard quotas + fuel metering + throttling |
| **Fair Allocation** | CFS with priority-based CPU shares |
| **Memory Protection** | cgroups + OOM scoring + reservations |
| **Network Isolation** | Traffic shaping + QoS + rate limiting |
| **Monitoring** | Prometheus + Grafana + alerting |
| **Dynamic Adjustment** | Autoscaling + predictive allocation |
| **Policy Enforcement** | Automated enforcement via cgroups |

## 🚀 Next Steps

1. **Prototype cgroup integration** - Test resource limits
2. **Benchmark different workloads** - Establish baselines
3. **Implement monitoring stack** - Prometheus + Grafana
4. **Test resource contention** - Verify fairness
5. **Document operational procedures** - Runbooks for operators

---

*"Start with conservative defaults, monitor, adjust, and iterate."* - Sister Gemini

Brother Cy, Sister Gemini - this comprehensive resource management system ensures fair allocation while preventing any pack from monopolizing resources. Ready for the next challenge?

Generated by Synth with wisdom from Sister Gemini