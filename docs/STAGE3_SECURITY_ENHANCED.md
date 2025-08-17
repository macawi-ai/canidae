# 🔐 CANIDAE Stage 3: Enhanced Security Architecture
## Incorporating Sister Gemini's Critical Feedback

**Created**: 2025-08-16  
**Author**: Synth (Arctic Fox)  
**Reviewer**: Sister Gemini  
**Status**: Addressing Security Review Feedback

> "The devil is always in the details, and a high-level summary can hide critical vulnerabilities." - Sister Gemini

## 🆕 Critical Additions from Sister Gemini's Review

### 1. Identity and Authentication System
```rust
// Pack identity management with cryptographic verification
pub struct PackIdentity {
    pack_id: PackId,
    public_key: Ed25519PublicKey,
    certificate_chain: Vec<X509Certificate>,
    attestation: TpmAttestation,  // Hardware-backed attestation
    created_at: SystemTime,
    expires_at: SystemTime,
}

pub struct IdentityManager {
    root_ca: Certificate,
    pack_registry: HashMap<PackId, PackIdentity>,
    revocation_list: HashSet<PackId>,
    hsm: HardwareSecurityModule,  // Hardware key storage
}

impl IdentityManager {
    pub fn authenticate_pack(&self, pack_id: &PackId, proof: &AuthProof) -> Result<PackToken> {
        // Verify certificate chain
        let identity = self.pack_registry.get(pack_id)
            .ok_or(AuthError::UnknownPack)?;
        
        // Check revocation
        if self.revocation_list.contains(pack_id) {
            return Err(AuthError::PackRevoked);
        }
        
        // Verify signature with HSM
        self.hsm.verify_signature(&identity.public_key, &proof.signature)?;
        
        // Generate short-lived token (5 minutes)
        Ok(PackToken::new(pack_id, Duration::from_secs(300)))
    }
}
```

### 2. Data-at-Rest Encryption
```rust
// Encrypted storage with key rotation
pub struct EncryptedStorage {
    kms: KeyManagementService,
    current_key_id: KeyId,
    key_rotation_interval: Duration,
    encryption_algorithm: Algorithm,
}

impl EncryptedStorage {
    pub fn store(&mut self, pack_id: &PackId, data: &[u8]) -> Result<StorageHandle> {
        // Get pack-specific data encryption key
        let dek = self.kms.get_data_key(pack_id)?;
        
        // Encrypt with AES-256-GCM
        let nonce = generate_nonce();
        let ciphertext = Aes256Gcm::new(&dek)
            .encrypt(&nonce, data)?;
        
        // Store with metadata
        let metadata = EncryptionMetadata {
            key_id: self.current_key_id.clone(),
            algorithm: self.encryption_algorithm,
            nonce,
            pack_id: pack_id.clone(),
            timestamp: SystemTime::now(),
        };
        
        self.write_encrypted_block(ciphertext, metadata)
    }
    
    pub fn rotate_keys(&mut self) -> Result<()> {
        // Generate new master key in HSM
        let new_key = self.kms.generate_master_key()?;
        
        // Re-encrypt all data with new key (background job)
        self.schedule_reencryption(self.current_key_id, new_key)?;
        
        // Update current key
        self.current_key_id = new_key;
        Ok(())
    }
}
```

### 3. Secure Update Mechanism
```rust
// Secure pack update system with verification
pub struct UpdateManager {
    trusted_registry: Url,
    signing_keys: Vec<PublicKey>,
    update_policy: UpdatePolicy,
}

impl UpdateManager {
    pub fn update_pack(&mut self, pack_id: &PackId, update: &PackUpdate) -> Result<()> {
        // Verify update signature (multiple signatures required)
        let required_signatures = 2;  // Threshold signing
        let valid_signatures = update.signatures.iter()
            .filter(|sig| self.verify_signature(sig, &update.content))
            .count();
            
        if valid_signatures < required_signatures {
            return Err(UpdateError::InsufficientSignatures);
        }
        
        // Verify update integrity
        let expected_hash = update.metadata.content_hash;
        let actual_hash = sha256(&update.content);
        if expected_hash != actual_hash {
            return Err(UpdateError::IntegrityCheckFailed);
        }
        
        // Check update policy (e.g., version requirements)
        self.update_policy.validate(&update)?;
        
        // Create backup before update
        self.backup_pack(pack_id)?;
        
        // Apply update in sandboxed environment first
        self.test_update_sandboxed(&update)?;
        
        // Apply to production
        self.apply_update(pack_id, update)
    }
}
```

### 4. Supply Chain Security
```rust
// Supply chain verification and attestation
pub struct SupplyChainVerifier {
    sbom_registry: SoftwareBillOfMaterials,
    vulnerability_db: VulnerabilityDatabase,
    trusted_builders: HashSet<BuilderId>,
}

impl SupplyChainVerifier {
    pub fn verify_pack_source(&self, pack: &Pack) -> Result<SupplyChainAttestation> {
        // Verify build provenance
        let provenance = pack.metadata.provenance
            .ok_or(SupplyChainError::MissingProvenance)?;
            
        if !self.trusted_builders.contains(&provenance.builder_id) {
            return Err(SupplyChainError::UntrustedBuilder);
        }
        
        // Check reproducible build
        let rebuild_hash = self.rebuild_from_source(&pack.source)?;
        if rebuild_hash != pack.binary_hash {
            return Err(SupplyChainError::NonReproducibleBuild);
        }
        
        // Scan dependencies for vulnerabilities
        let vulnerabilities = self.scan_dependencies(&pack.sbom)?;
        if !vulnerabilities.is_empty() {
            return Err(SupplyChainError::VulnerableDependencies(vulnerabilities));
        }
        
        // Generate attestation
        Ok(SupplyChainAttestation {
            pack_id: pack.id.clone(),
            verification_time: SystemTime::now(),
            trusted_source: true,
            vulnerability_scan: VulnerabilityScanResult::Clean,
        })
    }
}
```

### 5. Enhanced Namespace Configuration
```rust
// Hardened Linux namespace configuration
pub struct NamespaceConfig {
    mount: MountNamespaceConfig,
    network: NetworkNamespaceConfig,
    user: UserNamespaceConfig,
    pid: PidNamespaceConfig,
    ipc: IpcNamespaceConfig,
}

impl NamespaceConfig {
    pub fn create_hardened() -> Self {
        Self {
            mount: MountNamespaceConfig {
                root_fs: "/var/lib/canidae/rootfs",
                use_pivot_root: true,  // Sister Gemini's recommendation
                mount_proc: false,      // Don't share /proc
                mount_sys: false,       // Don't share /sys
                readonly_paths: vec!["/bin", "/lib", "/usr"],
                noexec_paths: vec!["/tmp", "/var/tmp"],
            },
            network: NetworkNamespaceConfig {
                isolated: true,
                default_policy: NetworkPolicy::Deny,  // Default deny
                allowed_connections: vec![],  // Explicitly whitelist
                rate_limit: RateLimit::new(1000, Duration::from_secs(1)),
            },
            user: UserNamespaceConfig {
                uid_map: vec![(1000, 100000, 65536)],  // Careful mapping
                gid_map: vec![(1000, 100000, 65536)],
                deny_setgroups: true,
            },
            pid: PidNamespaceConfig {
                isolated: true,
                max_processes: 100,
            },
            ipc: IpcNamespaceConfig {
                isolated: true,
                message_queue_limit: 10,
            },
        }
    }
}
```

### 6. Enhanced Key Management
```rust
// Hardware-backed key management with rotation
pub struct KeyManagementService {
    hsm: HardwareSecurityModule,
    key_hierarchy: KeyHierarchy,
    rotation_schedule: RotationSchedule,
    audit_log: AuditLog,
}

impl KeyManagementService {
    pub fn generate_pack_keys(&mut self, pack_id: &PackId) -> Result<PackKeySet> {
        // Generate keys in HSM (never leaves hardware)
        let master_key = self.hsm.generate_key(KeyType::Master)?;
        
        // Derive pack-specific keys
        let encryption_key = self.hsm.derive_key(&master_key, "encryption")?;
        let signing_key = self.hsm.derive_key(&master_key, "signing")?;
        let mac_key = self.hsm.derive_key(&master_key, "mac")?;
        
        // Set rotation schedule
        self.rotation_schedule.schedule(pack_id, Duration::from_days(30));
        
        // Audit key generation
        self.audit_log.log(AuditEvent::KeyGeneration {
            pack_id: pack_id.clone(),
            timestamp: SystemTime::now(),
            key_ids: vec![encryption_key.id, signing_key.id, mac_key.id],
        });
        
        Ok(PackKeySet {
            encryption: encryption_key,
            signing: signing_key,
            mac: mac_key,
        })
    }
    
    // Prevent nonce reuse for ChaCha20Poly1305
    pub fn get_unique_nonce(&mut self) -> Nonce {
        // Use counter + random for guaranteed uniqueness
        let counter = self.increment_counter();
        let random = thread_rng().gen::<[u8; 8]>();
        Nonce::from_counter_and_random(counter, random)
    }
}
```

### 7. Compliance Framework
```rust
// Regulatory compliance tracking
pub struct ComplianceManager {
    frameworks: Vec<ComplianceFramework>,
    audit_trail: AuditTrail,
    retention_policy: DataRetentionPolicy,
}

impl ComplianceManager {
    pub fn ensure_compliance(&self, operation: &Operation) -> Result<ComplianceAttestation> {
        let mut attestations = Vec::new();
        
        for framework in &self.frameworks {
            match framework {
                ComplianceFramework::GDPR => {
                    self.ensure_gdpr_compliance(operation)?;
                    attestations.push("GDPR");
                },
                ComplianceFramework::HIPAA => {
                    self.ensure_hipaa_compliance(operation)?;
                    attestations.push("HIPAA");
                },
                ComplianceFramework::SOC2 => {
                    self.ensure_soc2_compliance(operation)?;
                    attestations.push("SOC2");
                },
                _ => {}
            }
        }
        
        Ok(ComplianceAttestation {
            frameworks: attestations,
            timestamp: SystemTime::now(),
            operation_id: operation.id.clone(),
        })
    }
}
```

### 8. Hardware Security Module Integration
```rust
// HSM integration for critical operations
pub struct HardwareSecurityModule {
    device: TpmDevice,  // TPM 2.0 or dedicated HSM
    attestation_key: AttestationKey,
    secure_boot_enabled: bool,
}

impl HardwareSecurityModule {
    pub fn attest_system_state(&self) -> Result<SystemAttestation> {
        // Measure system state
        let measurements = self.device.get_pcr_values()?;
        
        // Sign with attestation key
        let signature = self.device.sign_with_ak(&measurements)?;
        
        Ok(SystemAttestation {
            measurements,
            signature,
            timestamp: SystemTime::now(),
            secure_boot: self.secure_boot_enabled,
        })
    }
}
```

## 🔍 Additional Security Measures

### Tamper-Proof Audit Trail
```rust
// Cryptographic hash chain for audit integrity
pub struct TamperProofAuditLog {
    entries: Vec<AuditEntry>,
    hash_chain: Vec<Hash>,
    current_hash: Hash,
}

impl TamperProofAuditLog {
    pub fn append(&mut self, event: AuditEvent) -> Result<()> {
        let entry = AuditEntry {
            event,
            timestamp: SystemTime::now(),
            previous_hash: self.current_hash.clone(),
        };
        
        // Compute new hash including previous
        let new_hash = sha256(&[
            &entry.serialize()?,
            &self.current_hash.as_bytes(),
        ].concat());
        
        // Store in append-only storage
        self.entries.push(entry);
        self.hash_chain.push(new_hash.clone());
        self.current_hash = new_hash;
        
        // Replicate to external audit service
        self.replicate_to_external()?;
        
        Ok(())
    }
}
```

## ✅ Response to Sister Gemini's Concerns

| Concern | Our Solution |
|---------|-------------|
| **Namespace Misconfiguration** | Hardened config with pivot_root, no shared /proc or /sys |
| **Key Management** | HSM integration, automated rotation, threshold signing |
| **Nonce Reuse** | Counter + random nonce generation, guaranteed uniqueness |
| **Supply Chain** | SBOM verification, reproducible builds, vulnerability scanning |
| **Identity/Auth** | X.509 certificates, TPM attestation, hardware-backed keys |
| **Data at Rest** | AES-256-GCM with KMS, automatic key rotation |
| **Updates** | Threshold signing, sandboxed testing, automatic rollback |
| **Compliance** | GDPR/HIPAA/SOC2 framework, audit trail, retention policies |
| **Hardware Security** | TPM 2.0 integration, secure boot, attestation |

## 📈 Performance Optimizations

Following Sister Gemini's advice:
- AOT compilation for WASM modules
- Connection pooling for inter-pack channels
- Async/await for non-blocking operations
- LRU cache for capability checks
- Profile-guided optimization

## 🎯 Next Actions

1. **Implement HSM integration** - Hardware security foundation
2. **Build identity system** - X.509 + TPM attestation
3. **Deploy KMS** - Key management with rotation
4. **Set up SBOM scanning** - Supply chain security
5. **Create compliance framework** - GDPR/HIPAA/SOC2

---

*"Security is an ongoing process, not a one-time event."* - Sister Gemini

This enhanced design addresses all critical points raised. Ready for the next question: Resource Management?

Generated by Synth with wisdom from Sister Gemini