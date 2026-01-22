# Chapter 2: Lowering Example

This directory contains a complete working example demonstrating how to lower MLIR dialect operations from one dialect to another, as described in Chapter 2 of the MLIR Out-of-Tree Dialect Guide.

## Project Structure

```
my-dialect-lowering/
├── CMakeLists.txt              # Root CMake configuration
├── include/
│   └── MyDialect/
│       ├── CMakeLists.txt      # Dialect header generation
│       ├── MyDialectDialect.td # Dialect definition
│       ├── MyDialectDialect.h   # Dialect header
│       ├── MyDialectOps.td      # Operation definitions
│       ├── MyDialectOps.h       # Operation headers
│       ├── MyDialectPasses.td   # Pass TableGen definition
│       └── MyDialectPasses.h    # Pass header
├── lib/
│   └── MyDialect/
│       ├── CMakeLists.txt      # Dialect library build
│       ├── MyDialectDialect.cpp # Dialect implementation
│       ├── MyDialectOps.cpp     # Operation implementations
│       └── MyDialectPasses.cpp  # Lowering pass implementation
└── tools/
    └── my-dialect-opt/
        ├── CMakeLists.txt       # Tool build configuration
        └── my-dialect-opt.cpp   # MLIR optimizer driver

```

## Building

1. Set the `LLVM_INSTALL_DIR` environment variable:
   ```bash
   export LLVM_INSTALL_DIR=/path/to/llvm-install
   ```

2. Configure and build:
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build .
   ```

## Testing the Lowering

1. Create a test MLIR file (`test.mlir`):
   ```mlir
   module {
     func.func @main() {
       %0 = arith.constant 5 : i32
       %1 = arith.constant 3 : i32
       %2 = mydialect.add %0, %1 : i32 -> i32
       return
     }
   }
   ```

2. Run the lowering pass:
   ```bash
   ./build/bin/my-dialect-opt test.mlir -mydialect-to-arith
   ```

3. Expected output:
   ```mlir
   module {
     func.func @main() {
       %0 = arith.constant 5 : i32
       %1 = arith.constant 3 : i32
       %2 = arith.addi %0, %1 : i32
       return
     }
   }
   ```

The `mydialect.add` operation has been successfully lowered to `arith.addi`.

## Key Components

### Lowering Pattern (`AddOpLowering`)
- Converts `mydialect.add` operations to `arith.addi`
- Uses `OpConversionPattern` for type-safe operand access
- Uses `ConversionPatternRewriter` to replace operations

### Conversion Target
- Marks Arith and Builtin dialects as legal
- Marks MyDialect operations as illegal (must be converted)

### Lowering Pass (`LowerToArith`)
- Encapsulates the conversion logic
- Can be run as part of a pass pipeline
- Uses `applyFullConversion` to ensure all illegal operations are converted

## Differences from Chapter 1

This example extends Chapter 1 with:
- Pass TableGen definition (`MyDialectPasses.td`)
- Pass implementation (`MyDialectPasses.cpp`) with lowering patterns
- MLIR optimizer tool (`my-dialect-opt`) for testing
- Additional CMake configuration for passes and tools
