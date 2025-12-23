# HPC Encryption Benchmark - Komendy

## Instalacja zależności

### macOS

```bash
# Zainstaluj Homebrew (jeśli nie masz)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Zainstaluj wymagane pakiety
brew install cmake openssl libomp

# Zainstaluj Xcode Command Line Tools
xcode-select --install
```

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake libssl-dev libomp-dev ocl-icd-opencl-dev

# Dla NVIDIA GPU
# Pobierz i zainstaluj CUDA Toolkit z: https://developer.nvidia.com/cuda-downloads
```

## Budowanie projektu

```bash
# Przejdź do katalogu projektu
cd /Users/mptb/Documents/Studia/Data_Science/Projekt_MAG/PROJEKT_MAG

# Utwórz katalog build
mkdir build
cd build

# Konfiguracja CMake (macOS)
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=$(brew --prefix openssl)

# Konfiguracja CMake (Linux)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Kompilacja
make -j$(sysctl -n hw.ncpu)  # macOS
make -j$(nproc)              # Linux
```

## Uruchamianie benchmarków

### Podstawowe użycie

```bash
# Standardowy benchmark
./hpc_benchmark

# Z weryfikacją
./hpc_benchmark --verify

# Bez weryfikacji (szybsze)
./hpc_benchmark --no-verify
```

### Zaawansowane opcje

```bash
# Własne rozmiary plików (w MB)
./hpc_benchmark --sizes 1,10,100,500,1000

# Więcej iteracji dla dokładniejszych wyników
./hpc_benchmark --iterations 5

# Maksymalna liczba wątków
./hpc_benchmark --max-threads 8

# Wyłącz testy skalowania wątków
./hpc_benchmark --no-thread-scaling

# Własny plik wyjściowy
./hpc_benchmark --output my_results.csv

# Pełny przykład
./hpc_benchmark --sizes 1,10,100 --iterations 3 --max-threads 14 --output results.csv
```

## Generowanie wykresów

### Przygotowanie środowiska Python

```bash
# Utwórz wirtualne środowisko Python
python3 -m venv .venv

# Aktywuj środowisko
source .venv/bin/activate  # macOS/Linux
# lub
.venv\Scripts\activate     # Windows

# Zainstaluj wymagane biblioteki
pip install pandas matplotlib
```

### Generowanie wykresów

```bash
# Z katalogu build
python3 ../scripts/generate_charts.py results.csv

# Lub z katalogu głównego projektu
python3 scripts/generate_charts.py build/results.csv
```

### Wygenerowane wykresy

Po uruchomieniu skryptu zostaną utworzone następujące pliki PNG:

- `speedup_by_threads.png` - Przyśpieszenie vs liczba wątków
- `efficiency_by_threads.png` - Efektywność równoległości
- `throughput_comparison.png` - Porównanie przepustowości
- `time_comparison.png` - Porównanie czasów wykonania
- `energy_efficiency.png` - Efektywność energetyczna

## Czyszczenie projektu

```bash
# Usuń pliki build
cd build
rm -rf *

# Lub usuń cały katalog build
cd ..
rm -rf build

# Dezaktywuj środowisko Python
deactivate
```

## Pełny workflow

```bash
# 1. Budowanie
cd PROJEKT_MAG
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=$(brew --prefix openssl)
make -j$(sysctl -n hw.ncpu)

# 2. Uruchomienie benchmarku
./hpc_benchmark --sizes 1,10,100 --iterations 3 --output results.csv

# 3. Przygotowanie Python
cd ..
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib

# 4. Generowanie wykresów
python3 scripts/generate_charts.py build/results.csv

# 5. Wyświetlenie wyników
cat build/results.csv
open build/*.png  # macOS
xdg-open build/*.png  # Linux
```

## Rozwiązywanie problemów

### OpenMP nie działa

```bash
# macOS - zainstaluj libomp
brew install libomp

# Przebuduj projekt
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=$(brew --prefix openssl)
make -j$(sysctl -n hw.ncpu)
```

### Metal Toolchain brak

```bash
# Metal shadery są osadzone w kodzie, nie potrzebujesz zewnętrznego toolchain
# Jeśli nadal występuje błąd, upewnij się że:
xcode-select --install
```

### CUDA nie znaleziona (Linux)

```bash
# Sprawdź czy CUDA jest zainstalowana
nvcc --version

# Dodaj CUDA do PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Przykładowe wyniki

```
═══════════════════════════════════════════════════════════════════════════
VERIFICATION SUMMARY
═══════════════════════════════════════════════════════════════════════════
  Total Tests: 48
  Passed:      48 (100%)
  Failed:      0 (0%)

  ✓ All encryption/decryption operations completed successfully!

Results saved to: results.csv
Generate charts with: python3 scripts/generate_charts.py results.csv
```
