# VulkanForge — RDNA 4 / gfx1201 Referenz-Links

**Erstellt:** 25.04.2026
**Hardware:** RX 9070 XT (gfx1201, RDNA 4)

---

## GPUOpen — Architektur & Optimierung

### ISA Reference
- RDNA 4 ISA Guide: https://gpuopen.com/rdna/
- Machine-Readable ISA (XML): https://gpuopen.com/machine-readable-isa/

### WMMA / Matrix Cores (Phase 4 GA)
- Using Matrix Cores of RDNA 4: https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
  → _gfx12 Intrinsics, vereinfachtes VGPR-Layout vs RDNA 3

### Performance Guide
- RDNA Performance Guide: https://gpuopen.com/learn/rdna-performance-guide/
  → LDS-Nutzung, Wavefront-Management, Vulkan/DX12

### Tools
- Radeon GPU Analyzer (RGA): https://gpuopen.com/radeon-gpu-analyzer/
  → SPIR-V → gfx1201 Disassembly, Register-Belegung, Occupancy

---

## Mesa / RADV — Treiber-Internals

### Hardware-Definition
- amd_family.h: https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/amd/common/amd_family.h
- ac_gpu_info.c: https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/amd/common/ac_gpu_info.c
  → Suche: CHIP_GFX1201

### RADV Vulkan-Treiber
- Hauptverzeichnis: https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/amd/vulkan
- radv_device.c: Extensions für gfx1201
- radv_shader.c: Shader-Vorbereitung für GFX12

### ACO Shader-Compiler
- Compiler-Verzeichnis: https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/amd/compiler
- aco_instruction_selection.cpp: Suche "gfx12" für RDNA4-spezifische Codegen-Pfade

### Register-Definitionen
- src/amd/registers/: JSON/Python für gfx12 Hardware-Register

---

## Lokale Diagnose-Befehle

```fish
# Device Properties
vulkaninfo --prop | grep -A 20 "deviceProperties"

# GFX12-spezifische Limits
vulkaninfo | grep -i "Gfx12"

# Compute Capabilities
vulkaninfo | grep -i "maxComputeWorkGroup"
vulkaninfo | grep -i "subgroupSize"
vulkaninfo | grep -i "timestampPeriod"
```

---

## Relevanz pro Phase

```
Phase 2-3: RDNA Performance Guide (Dispatch/Barrier-Debugging)
           RADV radv_device.c (Extension-Support)
           ACO aco_instruction_selection.cpp (Shader-Kompilierung)

Phase 4:   WMMA Matrix Cores Guide (coopmat-Optimierung)
           RGA (Shader-Disassembly + Occupancy-Analyse für GA)
           ISA Reference (Assembler-Level Verständnis)
           Register-Definitionen (VGPR-Budget-Analyse)
```
