# MyDialect - Chapter 1 Example

This is a minimal example of an out-of-tree MLIR dialect created following Chapter 1 of the MLIR Out-of-Tree Dialect Guide.

## Building

### Prerequisites

- LLVM and MLIR built and installed
- CMake 3.20.0 or later
- C++17 compatible compiler

### Configuration

Set `LLVM_INSTALL_DIR` to point to your LLVM installation directory. You can do this in one of three ways:

1. **Environment variable**:
   ```bash
   export LLVM_INSTALL_DIR=/path/to/llvm-install
   cmake -B build -S .
   ```

2. **CMake cache variable**:
   ```bash
   cmake -B build -S . -DLLVM_INSTALL_DIR=/path/to/llvm-install
   ```

3. **Direct MLIR_DIR**:
   ```bash
   cmake -B build -S . -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir
   ```

### Build

```bash
cd build
cmake --build .
```

The dialect library will be built at `build/lib/libMLIRMyDialect.a` (or `.so` on Linux).

## Structure

```
test-dialect/
├── CMakeLists.txt          # Root CMakeLists.txt
├── include/
│   ├── CMakeLists.txt
│   └── MyDialect/
│       ├── CMakeLists.txt
│       ├── MyDialectDialect.h
│       ├── MyDialectDialect.td
│       ├── MyDialectOps.h
│       └── MyDialectOps.td
└── lib/
    ├── CMakeLists.txt
    └── MyDialect/
        ├── CMakeLists.txt
        ├── MyDialectDialect.cpp
        └── MyDialectOps.cpp
```

## Usage

This dialect defines a simple `mydialect.add` operation that adds two integer values.

Example MLIR code:
```mlir
%0 = arith.constant 5 : i32
%1 = arith.constant 3 : i32
%2 = mydialect.add %0, %1 : i32 -> i32
```

