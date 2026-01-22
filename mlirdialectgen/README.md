# mlirdialectgen

A Python tool to scaffold boilerplate code for a standalone MLIR dialect. This tool generates all the necessary files to create a working out-of-tree MLIR dialect with a simple `add` operation.

## Features

- Generates complete MLIR dialect structure
- Includes a working `add` operation example
- Creates all necessary CMakeLists.txt files
- Generates TableGen definitions (.td files)
- Creates C++ implementation files
- Produces a compilable dialect out of the box

## Installation

This tool uses `uv` for package management. Install dependencies:

```bash
cd mlirdialectgen
uv sync
```

## Usage

### Basic Usage

Generate a dialect in the current directory:

```bash
uv run mlirdialectgen my-dialect
```

This creates a `mydialect/` directory with all necessary files.

### Specify Output Directory

Generate a dialect in a specific directory:

```bash
uv run mlirdialectgen my-dialect --output-dir /path/to/output
```

### Specify LLVM Install Directory

The tool will use `$LLVM_INSTALL_DIR` environment variable if set, or default to `../llvm-install`. You can override it:

```bash
uv run mlirdialectgen my-dialect --llvm-install-dir /path/to/llvm-install
```

### Complete Example

```bash
uv run mlirdialectgen calculator-dialect \
    --output-dir ./dialects \
    --llvm-install-dir $HOME/llvm-install
```

## Generated Structure

The tool generates the following directory structure:

```
my-dialect/
├── CMakeLists.txt              # Root CMake configuration
├── README.md                   # Build instructions
├── include/
│   ├── CMakeLists.txt
│   └── MyDialect/              # Dialect name in PascalCase
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

## What's Included

The generated dialect includes:

- **Dialect Definition**: A basic MLIR dialect with namespace `mydialect`
- **Add Operation**: A simple `add` operation that takes two `i32` operands and returns their sum
- **CMake Build System**: Complete CMake configuration ready to build
- **TableGen Definitions**: All necessary `.td` files for dialect and operations
- **C++ Implementation**: Header and source files for dialect and operations

## Building the Generated Dialect

After generating a dialect, build it:

```bash
cd my-dialect
mkdir build && cd build
cmake .. -DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR
make -j$(nproc)
```

Or using Ninja:

```bash
cd my-dialect
mkdir build && cd build
cmake -G Ninja .. -DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR
ninja
```

## Example: Using the Generated Dialect

The generated dialect includes a simple `add` operation. You can use it in MLIR like this:

```mlir
module {
  func.func @main() {
    %0 = arith.constant 5 : i32
    %1 = arith.constant 3 : i32
    %2 = mydialect.add %0, %1 : i32 -> i32
    // %2 is now 8 : i32
    return
  }
}
```

## Dialect Name Normalization

The tool normalizes dialect names:
- Hyphens and underscores are removed for the dialect namespace
- The name is converted to lowercase
- PascalCase is used for C++ class names

Examples:
- `my-dialect` → dialect name: `mydialect`, class: `MyDialect`
- `calculator_dialect` → dialect name: `calculatordialect`, class: `CalculatorDialect`
- `MyDialect` → dialect name: `mydialect`, class: `MyDialect`

## Requirements

- Python 3.11 or later
- `uv` package manager
- LLVM with MLIR installed (for building the generated dialect)

## Dependencies

- `click`: Command-line interface
- `jinja2`: Template engine for code generation

## License

This tool is provided as-is for generating MLIR dialect boilerplate code.

## See Also

For detailed information on creating out-of-tree MLIR dialects, see:
- `MLIR_Out_of_Tree_Dialect_Guide.md` - Comprehensive guide to creating MLIR dialects

