# HPC Encryption Benchmark

Parallel encryption benchmark for scientific analysis. Compares XOR (memory-bound) and AES-256-CTR (compute-bound) algorithms across CPU and GPU technologies.

## Quick Start

```bash
# Clone and build
git clone <repo-url> && cd PROJEKT_MAG
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

# Run benchmark
./hpc_benchmark --sizes 1,10,100 --iterations 3
```

> **Note**: OpenSSL, OpenMP, Metal (macOS), and CUDA (Linux) are auto-detected. No manual configuration needed.

## Platforms

| Platform              | Engines                   |
| --------------------- | ------------------------- |
| macOS (Apple Silicon) | Sequential, OpenMP, Metal |
| Linux (NVIDIA GPU)    | Sequential, OpenMP, CUDA  |

## Requirements

### macOS

```bash
xcode-select --install
brew install cmake openssl libomp
```

### Linux/WSL

```bash
sudo apt update
sudo apt install build-essential cmake libssl-dev libomp-dev
# For CUDA: install NVIDIA CUDA Toolkit 11+
```

## Build Options

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_OPENMP=ON \
         -DBUILD_METAL=ON \      # macOS only
         -DBUILD_CUDA=ON \       # Linux with NVIDIA
         -DOPENSSL_ROOT_DIR=$(brew --prefix openssl)
```

## Usage

```bash
./hpc_benchmark [options]

--sizes <list>         File sizes in MB (default: 1,10,100)
--iterations <n>       Iterations per test (default: 3)
--verify / --no-verify Enable/disable verification
--thread-scaling       Test multiple thread counts
--max-threads <n>      Maximum threads for scaling
--output <file>        Output CSV file (default: <platform>_results.csv)
--help                 Show help
```

> **Default output**: Results are saved to the current directory as `macOS_results.csv` (macOS) or `WSL_results.csv` / `Linux_results.csv` (Linux).

### Examples

```bash
./hpc_benchmark                                    # Default benchmark
./hpc_benchmark --sizes 1,10,100,500 --iterations 5
./hpc_benchmark --no-thread-scaling --max-threads 8
./hpc_benchmark --output ../results/my_results.csv
```

## Output

Results saved to `<platform>_results.csv`:

| Column         | Description                     |
| -------------- | ------------------------------- |
| Algorithm      | XOR or AES-256-CTR              |
| Engine         | Sequential, OpenMP, Metal, CUDA |
| FileSize_MB    | Test file size                  |
| Throughput_MBs | Processing speed                |
| Speedup        | Relative to sequential          |
| Efficiency     | Speedup / threads               |
| Energy_Joules  | Energy consumed                 |

## Visualization

```bash
cd scripts
python3 generate_charts.py
# or use generate_charts.ipynb
```

## Project Structure

```
PROJEKT_MAG/
├── CMakeLists.txt          # Build configuration
├── src/
│   ├── main.cpp            # CLI benchmark
│   ├── common/             # Timer, CSV, verification, power
│   └── engines/
│       ├── xor/            # XOR: sequential, openmp, cuda, metal
│       └── aes/            # AES: sequential, openmp, cuda, metal
├── scripts/                # Python visualization
└── results/                # CSV output files
```

## Algorithms

| Algorithm   | Type          | Characteristic                    |
| ----------- | ------------- | --------------------------------- |
| XOR         | Memory-bound  | Limited by RAM bandwidth          |
| AES-256-CTR | Compute-bound | CPU: OpenSSL AES-NI, GPU: T-table |

## Research Metrics

- **Throughput (MB/s)**: Data encrypted per second
- **Speedup**: Parallel time / sequential time
- **Efficiency**: Speedup / thread count (parallel efficiency)
- **Energy (J)**: Measured via Intel RAPL (Linux) or estimated (macOS)

## License

MIT License
