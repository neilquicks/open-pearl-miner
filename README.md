# Open Pearl Miner

A high-performance **Pearl (PRL)** proof-of-work miner for **NVIDIA and AMD (beta) GPUs**.

No Python, CUDA toolkit, or PyTorch required **to run** — the CUDA runtime is bundled
in pre-built releases. Just an NVIDIA driver and the standalone binary.
Build it yourself or grab a pre-built release. See [License](#license) for dev-fee terms.

> Mixed-GPU rigs are supported. Ampere/Ada cards run a fused int8 tensor-core kernel
> (GEMM + in-mainloop transcript fold + tensor-core noise generation), bit-exact with
> the Pascal DP4A path; Pascal cards run DP4A.
> 
## Features

- **AMD cards beta support**
- **~7.0 TH/s** sustained on a single Tesla P40 (Pascal DP4A).
- **~30 TH/s** on a single RTX 4050 mobile (Ada) — fused int8 **tensor-core** GEMM +
  in-mainloop transcript fold + tensor-core noise generation, all bit-exact with DP4A.
- **Multi-GPU** — auto-detects every GPU and runs one worker per card, near-linear scaling.
- **Continuous mining** — no idle time waiting between pool jobs.
- **Background proof submission** — finding a share never stalls the search.
- **Solo mining** — mine to your local pearl-gateway node.
- **Pool mining** — built for LuckyPool's Pearl stratum (default) with more pool protocols planned.
- **HiveOS** — full HiveOS custom-miner package.

## Roadmap

- **Apple Silicon (Metal)** — Metal `simdgroup_matrix` int8 path for M-series GPUs.
- **Per-kernel autotuning** — pick tile/stage parameters per GPU at startup.

## Pre-built Releases

Grab the latest binary from the [Releases](https://github.com/neilquicks/Open-Pearl-Miner/releases) page:

| File | Platform |
|------|----------|
| `p40-miner-windows-x64.zip` | Windows x64 |
| `p40-miner-linux-x64.tar.gz` | Linux (glibc >= 2.31) |
| `p40-miner-hiveos-<ver>.tar.gz` | HiveOS custom miner |

**Requirements:** An up-to-date NVIDIA/AMD driver, and **Windows x64**, **Linux x86-64** (glibc >= 2.31), or **HiveOS**.

## Hashrate Nvidia/AMD

# AMD cards

> Real mining performance may vary depending on drivers,
> overclocking, algorithm updates and network conditions.

| GPU | Hashrate (TH/s) | Power (W) | Efficiency (GH/W) |
|------|----------------|-----------|-------------------|
| AMD RX 570 4GB | 8 TH/s | 85 | 94 |
| AMD RX 580 8GB | 10 TH/s | 105 | 95 |
| AMD RX 5500 XT 8GB | 14 TH/s | 90 | 156 |
| AMD RX 5600 XT | 20 TH/s | 110 | 182 |
| AMD RX 5700 | 28 TH/s | 125 | 224 |
| AMD RX 5700 XT | 32 TH/s | 140 | 229 |
| AMD RX 6600 | 38 TH/s | 65 | 585 |
| AMD RX 6600 XT | 44 TH/s | 80 | 550 |
| AMD RX 6650 XT | 47 TH/s | 85 | 553 |
| AMD RX 6700 XT | 58 TH/s | 115 | 504 |
| AMD RX 6750 XT | 62 TH/s | 125 | 496 |
| AMD RX 6800 | 75 TH/s | 145 | 517 |
| AMD RX 6800 XT | 82 TH/s | 175 | 469 |
| AMD RX 6900 XT | 88 TH/s | 190 | 463 |
| AMD RX 6950 XT | 94 TH/s | 215 | 437 |
| AMD RX 7600 | 52 TH/s | 95 | 547 |
| AMD RX 7700 XT | 73 TH/s | 170 | 429 |
| AMD RX 7800 XT | 86 TH/s | 200 | 430 |
| AMD RX 7900 GRE | 98 TH/s | 220 | 445 |
| AMD RX 7900 XT | 115 TH/s | 270 | 426 |
| AMD RX 7900 XTX | 128 TH/s | 320 | 400 |


### Quick Start

```bat
:: Windows
p40-miner.exe --wallet prl1YOURWALLET --worker rig1
```

```bash
# Linux
./p40-miner --wallet prl1YOURWALLET --worker rig1
```

That's it — it auto-detects all GPUs and starts mining.

## Build from Source

### Prerequisites

| Dependency | Windows | Linux |
|---|---|---|
| **CUDA Toolkit** | 12.x ([NVIDIA](https://developer.nvidia.com/cuda-downloads)) | 12.x |
| **CUTLASS headers** | `git clone --depth 1 https://github.com/NVIDIA/cutlass` | same |
| **Python** | >= 3.12 | >= 3.12 |
| **MSVC** | Visual Studio 2022 (C++ workload) | — |
| **pip packages** | `numpy`, `blake3`, `py-pearl-mining` | same |

Set `CUTLASS_DIR` to the directory containing `cutlass/` and `cute/` headers.

### Windows

```bat
git clone https://github.com/neilquicks/Pascal-Pearl-Miner.git
cd Pascal-Pearl-Miner

:: 1. Build the CUDA library (p40cuda.dll)
packaging\build_capi.bat

:: 2. Install Python deps
pip install numpy blake3 py-pearl-mining

:: 3. Run directly from source
python packaging\p40_miner_lite_main.py --wallet prl1YOURWALLET

:: Or freeze a standalone binary with PyInstaller:
pip install pyinstaller
pyinstaller packaging\p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi
dist\p40-miner\p40-miner.exe --wallet prl1YOURWALLET
```

To rebuild the full CUDA extension (includes torch bindings):

```bat
pip install -e .   # requires torch + CUTLASS_DIR
```

### Linux

```bash
git clone https://github.com/neilquicks/Pascal-Pearl-Miner.git
cd Pascal-Pearl-Miner

# 1. Build the CUDA library (libp40cuda.so)
CUTLASS_DIR=/path/to/cutlass/include bash packaging/build_capi.sh

# 2. Install Python deps
pip install numpy blake3 py-pearl-mining

# 3. Run directly from source
python packaging/p40_miner_lite_main.py --wallet prl1YOURWALLET

# Or freeze a standalone binary:
pip install pyinstaller
pyinstaller packaging/p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi
./dist/p40-miner/p40-miner --wallet prl1YOURWALLET
```

### HiveOS Package

Build the Linux binary first, then:

```bash
bash packaging/hiveos/build_hiveos_package.sh  # -> p40-miner-hiveos-<ver>.tar.gz
```

Upload to a URL or `scp` to the rig, then add as a Custom miner in HiveOS.

### Development Install

```bash
pip install -e .   # editable install of the torch-based extension
CUTLASS_DIR=... python -c "import p40_pearl_gemm"  # smoke test
```

## Usage

### Options

| Flag | Default | Description |
|---|---|---|
| `--wallet` | _(required)_ | Your Pearl payout address |
| `--worker` | `p40` | Worker name shown on the pool |
| `--pool` | _(auto by GPU)_ | Stratum `host:port`. Auto: Pascal→`pearl-cpu-eu1…:3370`, sm_80+→`pearl-eu2…:3360` |
| `--devices` | _(auto-detect all)_ | GPU selection, e.g. `0,1,2` or `all` |
| `--region` | `4096` | Sub-output search size |
| `--solo` | _(off)_ | Solo mine to local pearl-gateway `HOST:PORT` |

### Multi-GPU

Auto-detects every GPU and runs one worker per card (workers named `<worker>-gpu0`,
`-gpu1`, ...) with a combined-hashrate summary. Each GPU runs as an independent pinned
process with blocking-sync CUDA, giving near-linear scaling on 4–8 GPU rigs. Just run it:

```
p40-miner.exe --wallet prl1YOURWALLET
```

Use `--devices` to select specific cards:

```
p40-miner.exe --wallet prl1YOURWALLET --devices 0,2
```

### Pool Mining

The default pool is picked automatically by GPU class: **tensor-core cards (`sm_80+`)**
use `pearl-eu2.luckypool.io:3360` (GPU difficulty), while **Pascal / DP4A cards** (P40,
GTX 10-series) use `pearl-cpu-eu1.luckypool.io:3370`. On a mixed rig each card picks its
own. Override for any card with `--pool`:

```
p40-miner.exe --wallet prl1YOURWALLET --pool pearl-cpu-eu1.luckypool.io:3370
```

(The CPU pool's low difficulty produces frequent shares whose ~1 GB proof snapshot
stalls a fast GPU between grids — hence the GPU-difficulty default for tensor-core
cards.) Currently only LuckyPool's stratum protocol is supported; more planned.

### Solo Mining

```
p40-miner.exe --solo GATEWAY_HOST:PORT
```

No wallet flag needed — the node's configured address receives block rewards.
The same 2% dev fee applies (GPU mines to the dev's pool wallet during the 2% window).

## HiveOS

Download `p40-miner-1.6.1.tar.gz` and install it as a **Custom** miner:

1. **Flight Sheet → Miner = Custom.**
2. Set the **Installation URL** to the tarball, *or* `scp` it to the rig and run
   `tar -C /hive/miners/custom -xzf p40-miner-1.6.1.tar.gz`.
3. **Wallet and worker:** your Pearl wallet `prl1...` (worker name auto-appended).
4. **Pool URL:** `pearl-eu2.luckypool.io:3360` (default LuckyPool GPU pool).
5. **Extra config arguments** (optional): `--devices 0,1`, `--region 4096`, etc.

The miner reports per-GPU TH/s and accepted shares to the HiveOS dashboard.
Built against glibc 2.31, so it runs on both the *focal* (20.04) and *jammy* (22.04)
HiveOS images and newer.

## License

**Pascal Pearl Miner License** — see [LICENSE](LICENSE).

- Free to use, modify, and distribute (including commercially), **provided the 2% dev fee is retained**.
- Removal or bypass of the dev fee is a license violation in any distributed or commercial deployment.
- **Personal-use exemption**: you may disable the dev fee for strictly personal, non-commercial mining on your own hardware — but the modified version must not be distributed.
