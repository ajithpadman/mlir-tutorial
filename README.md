# Creating an Out-of-Tree MLIR Dialect in LLVM

This document provides a comprehensive, step-by-step guide to creating a new MLIR dialect outside the LLVM source tree. It covers everything from basic environment setup to implementing lowering passes.

## Table of Contents

1. [Chapter 1: Basic Setup and Dialect Creation](#chapter-1-basic-setup-and-dialect-creation)
   - [1.1 Building LLVM with MLIR and ClangIR](#11-building-llvm-with-mlir-and-clangir)
   - [1.2 Environment Setup for Out-of-Tree Dialect](#12-environment-setup-for-out-of-tree-dialect)
   - [1.3 Folder Structure](#13-folder-structure)
   - [1.4 CMakeLists.txt Files](#14-cmakeliststxt-files)
   - [1.5 Dialect Definition](#15-dialect-definition)
   - [1.6 Operation Definition (Add Operation Example)](#16-operation-definition-add-operation-example)
   - [1.7 Implementation Files](#17-implementation-files)
   - [1.8 Building the Dialect](#18-building-the-dialect)

2. [Chapter 2: Lowering](#chapter-2-lowering)
   - [2.1 Introduction to Lowering](#21-introduction-to-lowering)
   - [2.2 Conversion Patterns](#22-conversion-patterns)
   - [2.3 Conversion Target](#23-conversion-target)
   - [2.4 Rewrite Pattern Set](#24-rewrite-pattern-set)
   - [2.5 Example: Lowering Add Operation](#25-example-lowering-add-operation)
   - [2.6 Creating a Lowering Pass](#26-creating-a-lowering-pass)
   - [2.7 Complete Lowering Example](#27-complete-lowering-example)

3. [References](#references)

---

## Chapter 1: Basic Setup and Dialect Creation

### 1.1 Building LLVM with MLIR and ClangIR

Before creating an out-of-tree dialect, you need to build and install LLVM with MLIR and ClangIR support.

#### Prerequisites

Before starting, ensure you have the following installed:

- **Git**: For cloning the LLVM repository
- **CMake**: Version 3.20.0 or later
- **C++ Compiler**: Supporting C++17 standard or later (GCC 7+, Clang 5+, or MSVC 2017+)
- **Build System**: Ninja (recommended) or Make
- **Python 3**: Required for some build scripts

#### Cloning the LLVM Repository

Clone the LLVM monorepo which includes LLVM, Clang, and MLIR:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

#### Building LLVM with MLIR and ClangIR

Create a build directory and configure the build with CMake:

```bash
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install \
    -DLLVM_ENABLE_CIR=ON
```

**Key CMake options:**
- `-DCMAKE_BUILD_TYPE=Release`: Build in Release mode (use `Debug` for development)
- `-DLLVM_ENABLE_PROJECTS="clang;mlir"`: Enable Clang and MLIR projects
- `-DLLVM_TARGETS_TO_BUILD=host`: Build only the host target (faster)
- `-DLLVM_ENABLE_ASSERTIONS=ON`: Enable assertions (useful for development)
- `-DCMAKE_INSTALL_PREFIX`: Set your desired installation directory
- `-DLLVM_ENABLE_CIR=ON`: Enable ClangIR (CIR dialect) support

**Note**: For ClangIR integration, refer to the ClangIR documentation for the latest build instructions and any additional configuration options that may be required.

#### Building and Installing

Build LLVM, Clang, and MLIR:

```bash
ninja
```

This will take some time (30 minutes to several hours depending on your machine). After the build completes, install to the specified prefix:

```bash
ninja install
```

This installs all components to `$CMAKE_INSTALL_PREFIX` (e.g., `$HOME/llvm-install`).

#### Setting Up Environment Variable

To make the installation directory easily accessible, add it to your shell configuration file:

```bash
# Add to ~/.bashrc (or ~/.zshrc for zsh)
export LLVM_INSTALL_DIR=$HOME/llvm-install
```

Then reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

Verify the installation:

```bash
echo $LLVM_INSTALL_DIR
ls $LLVM_INSTALL_DIR/lib/cmake/mlir/MLIRConfig.cmake
```

The installation directory structure should look like:
```
$LLVM_INSTALL_DIR/
├── bin/          # Executables (clang, mlir-opt, etc.)
├── include/      # Header files (mlir/, clang/, llvm/)
├── lib/          # Libraries (.a or .so files)
├── lib/cmake/    # CMake config files (mlir/, llvm/)
└── share/        # Documentation and other files
```

### 1.2 Environment Setup for Out-of-Tree Dialect

#### Verifying MLIR Installation

Verify that MLIR is properly installed by checking for the CMake configuration file:

```bash
# The MLIR CMake config should be at:
$LLVM_INSTALL_DIR/lib/cmake/mlir/MLIRConfig.cmake
```

You can verify this by checking if the file exists:
```bash
ls $LLVM_INSTALL_DIR/lib/cmake/mlir/MLIRConfig.cmake
```

Also verify that MLIR headers are available:
```bash
ls $LLVM_INSTALL_DIR/include/mlir/
```

#### Basic CMake Configuration

Create a root `CMakeLists.txt` file in your project directory:

```cmake
cmake_minimum_required(VERSION 3.20.0)
project(my-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Find MLIR package using LLVM_INSTALL_DIR
# LLVM_INSTALL_DIR can be set as:
#   1. Environment variable: export LLVM_INSTALL_DIR=/path/to/llvm-install
#   2. CMake cache variable: cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm-install
#   3. Or specify MLIR_DIR directly: cmake .. -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir

if(DEFINED ENV{LLVM_INSTALL_DIR})
  set(LLVM_INSTALL_DIR $ENV{LLVM_INSTALL_DIR} CACHE PATH "Path to LLVM installation directory")
endif()

if(LLVM_INSTALL_DIR AND NOT MLIR_DIR)
  set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "Path to MLIR CMake config")
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set up include directories and library paths
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Include MLIR CMake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Set output directories
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(lib)
```

### 1.3 Folder Structure

Create the following directory structure for your out-of-tree dialect:

```
my-dialect/
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

This structure separates:
- **include/**: Header files and TableGen definitions (.td files)
- **lib/**: Implementation files (.cpp files)

### 1.4 CMakeLists.txt Files

#### Root CMakeLists.txt

The root `CMakeLists.txt` was shown in section 1.1. It sets up the build environment and includes subdirectories.

#### include/CMakeLists.txt

```cmake
add_subdirectory(MyDialect)
```

#### include/MyDialect/CMakeLists.txt

This file uses MLIR's `add_mlir_dialect()` function to generate C++ code from TableGen definitions:

```cmake
# Generate dialect code from TableGen
add_mlir_dialect(MyDialectOps mydialect)

# Generate documentation (optional)
add_mlir_doc(MyDialectDialect MyDialectDialect MyDialect/ -gen-dialect-doc)
add_mlir_doc(MyDialectOps MyDialectOps MyDialect/ -gen-op-doc)
```

The `add_mlir_dialect()` function:
- Takes the base name of the operations file (`MyDialectOps`) and the dialect namespace (`mydialect`)
- Generates header and implementation files for operations, types, and dialect registration
- Creates a tablegen target `MLIRMyDialectOpsIncGen` that other targets can depend on

#### lib/CMakeLists.txt

```cmake
add_subdirectory(MyDialect)
```

#### lib/MyDialect/CMakeLists.txt

This file builds the dialect library:

```cmake
add_mlir_dialect_library(MLIRMyDialect
  MyDialectDialect.cpp
  MyDialectOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/MyDialect

  DEPENDS
  MLIRMyDialectOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
```

The `add_mlir_dialect_library()` function:
- Creates a library target `MLIRMyDialect`
- Lists all implementation files
- Specifies header directories for IDE support
- Declares dependencies on generated TableGen files
- Links required MLIR libraries

### 1.5 Dialect Definition

#### MyDialectDialect.td

Create `include/MyDialect/MyDialectDialect.td` to define your dialect:

```tablegen
//===- MyDialectDialect.td - MyDialect dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_DIALECT
#define MYDIALECT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MyDialect dialect definition.
//===----------------------------------------------------------------------===//

def MyDialect_Dialect : Dialect {
  let name = "mydialect";
  let summary = "A simple out-of-tree MLIR dialect example.";
  let description = [{
    This dialect demonstrates how to create an out-of-tree MLIR dialect
    with basic operations. It serves as a template for building custom
    dialects outside the LLVM source tree.
  }];
  let cppNamespace = "::mlir::mydialect";

  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base MyDialect operation definition.
//===----------------------------------------------------------------------===//

class MyDialect_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyDialect_Dialect, mnemonic, traits>;

#endif // MYDIALECT_DIALECT
```

Key components:
- **Dialect definition**: `MyDialect_Dialect` specifies the dialect name, summary, description, and C++ namespace
- **Base operation class**: `MyDialect_Op` provides a convenient base class for all operations in this dialect
- **Includes**: `mlir/IR/OpBase.td` provides base definitions for operations and dialects

### 1.6 Operation Definition (Add Operation Example)

#### MyDialectOps.td

Create `include/MyDialect/MyDialectOps.td` to define operations:

```tablegen
//===- MyDialectOps.td - MyDialect dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_OPS
#define MYDIALECT_OPS

include "MyDialectDialect.td"

//===----------------------------------------------------------------------===//
// MyDialect Operations
//===----------------------------------------------------------------------===//

def MyDialect_AddOp : MyDialect_Op<"add"> {
  let summary = "Adds two integer values";
  let description = [{
    The `mydialect.add` operation takes two integer operands and returns
    their sum.

    Example:

    ```mlir
    %0 = arith.constant 5 : i32
    %1 = arith.constant 3 : i32
    %2 = mydialect.add %0, %1 : i32 -> i32
    // %2 is now 8 : i32
    ```
  }];

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];

  let hasVerifier = 0;
}
```

**Note on Assembly Format**: The `assemblyFormat` does NOT include the operation mnemonic (e.g., `"add"`). MLIR automatically prepends the dialect name and mnemonic when printing/parsing operations. The format above will produce output like:
```mlir
%result = mydialect.add %lhs, %rhs : i32 -> i32
```
The `mydialect.add` part is automatically generated from the dialect name (`mydialect`) and the mnemonic specified in the operation definition (`"add"`).

Explanation of the Add operation definition:

- **Operation name**: `MyDialect_AddOp` with mnemonic `"add"` creates `mydialect.add`
- **Arguments**: Two `i32` operands named `lhs` and `rhs`
- **Results**: One `i32` result named `result`
- **Assembly format**: Custom syntax for printing/parsing the operation, including both input and output types

### 1.7 Implementation Files

#### MyDialectDialect.h

Create `include/MyDialect/MyDialectDialect.h`:

```cpp
//===- MyDialectDialect.h - MyDialect dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_MYDIALECTDIALECT_H
#define MYDIALECT_MYDIALECTDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

// Include the auto-generated dialect header (no macro needed)
#include "MyDialect/MyDialectOpsDialect.h.inc"

#endif // MYDIALECT_MYDIALECTDIALECT_H
```

**Note**: The dialect header (`.h.inc`) does not require any macro definition before inclusion. The `GET_OP_CLASSES` macro is only needed for the operations header, which is included in `MyDialectOps.h`.

#### MyDialectDialect.cpp

Create `lib/MyDialect/MyDialectDialect.cpp`:

```cpp
//===- MyDialectDialect.cpp - MyDialect dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyDialect/MyDialectDialect.h"
#include "MyDialect/MyDialectOps.h"

using namespace mlir;
using namespace mlir::mydialect;

#include "MyDialect/MyDialectOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MyDialect dialect.
//===----------------------------------------------------------------------===//

void MyDialectDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MyDialect/MyDialectOps.cpp.inc"
      >();
}
```

The `initialize()` method:
- Registers all operations defined in the dialect
- Uses the `GET_OP_LIST` macro to include all generated operation declarations
- Is called automatically when the dialect is loaded

**Note**: Since we use `useDefaultTypePrinterParser = 1` in the dialect definition, we don't need to implement `parseType()` and `printType()` methods. These are only required if you define custom types in your dialect.

#### MyDialectOps.h

Create `include/MyDialect/MyDialectOps.h`:

```cpp
//===- MyDialectOps.h - MyDialect dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_MYDIALECTOPS_H
#define MYDIALECT_MYDIALECTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "MyDialect/MyDialectOps.h.inc"

#endif // MYDIALECT_MYDIALECTOPS_H
```

The `GET_OP_CLASSES` macro generates:
- Operation class definitions
- Accessor methods for operands and results
- Builder methods
- Verification methods

#### MyDialectOps.cpp

Create `lib/MyDialect/MyDialectOps.cpp`:

```cpp
//===- MyDialectOps.cpp - MyDialect dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyDialect/MyDialectOps.h"
#include "MyDialect/MyDialectDialect.h"

#define GET_OP_CLASSES
#include "MyDialect/MyDialectOps.cpp.inc"
```

This file includes the generated operation implementations.

### 1.8 Building the Dialect

#### Configuration

Create a build directory and configure the project:

```bash
mkdir build && cd build
cmake .. -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir
```

If `LLVM_INSTALL_DIR` is set in your environment, CMake will automatically find MLIR. Alternatively, you can specify it directly:

```bash
cmake .. -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir
```

#### Building

Build the dialect library:

```bash
cmake --build .
```

Or using Ninja (if configured with `-G Ninja`):
```bash
ninja
```

#### Verification

After building, you should see:
- `lib/libMLIRMyDialect.a` (or `.so` on Linux) - the dialect library
- Generated files in `include/MyDialect/` with `.inc` extensions

#### Testing the Build

Create a simple test program to verify the dialect works:

```cpp
// test_dialect.cpp
#include "MyDialect/MyDialectDialect.h"
#include "mlir/IR/MLIRContext.h"

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::mydialect::MyDialectDialect>();
  return 0;
}
```

Compile and run:
```bash
g++ -std=c++17 test_dialect.cpp \
    -I../include \
    -I$LLVM_INSTALL_DIR/include \
    -L$LLVM_INSTALL_DIR/lib \
    -lMLIRMyDialect -lMLIRIR -lLLVMSupport \
    -o test_dialect
./test_dialect
```

---

## Chapter 2: Lowering

### 2.1 Introduction to Lowering

**Lowering** is the process of converting operations from one MLIR dialect to another. This is a fundamental operation in MLIR compilation pipelines, allowing you to progressively transform high-level operations into lower-level representations.

Common lowering scenarios:
- **High-level → Mid-level**: Custom domain-specific operations → Affine/Arith operations
- **Mid-level → Low-level**: Affine operations → LLVM IR operations
- **Dialect-specific → Generic**: Specialized operations → Standard operations

Lowering in MLIR uses:
- **Conversion Patterns**: Rules that specify how to transform operations
- **Conversion Target**: Specification of what operations are legal/illegal after conversion
- **Type Converter**: Optional component for converting types between dialects

### 2.2 Conversion Patterns

Conversion patterns define how to transform operations. In MLIR, patterns inherit from `OpConversionPattern<SourceOp>`.

#### Pattern Structure

```cpp
struct MyOpLowering : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 1. Extract operands and attributes from the source operation
    // 2. Create new operations in the target dialect
    // 3. Replace the source operation with the new operations
    return success();
  }
};
```

Key components:
- **OpAdaptor**: Provides type-safe access to operands (already converted if using type converter)
- **ConversionPatternRewriter**: Used to modify the IR (create operations, replace operations)
- **matchAndRewrite()**: Returns `success()` if conversion succeeded, `failure()` otherwise

### 2.3 Conversion Target

The `ConversionTarget` specifies which operations are legal or illegal after conversion.

#### Setting up ConversionTarget

```cpp
ConversionTarget target(getContext());

// Mark entire dialects as legal
target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();

// Mark specific operations as legal
target.addLegalOp<some::Operation>();

// Mark entire dialects as illegal (must be converted)
target.addIllegalDialect<MyDialect>();

// Mark specific operations as illegal
target.addIllegalOp<mydialect::AddOp>();

// Dynamically mark operations as legal based on conditions
target.addDynamicallyLegalOp<MyDialect::PrintOp>([](MyDialect::PrintOp op) {
  // Only legal if all operands are already converted types
  return llvm::none_of(op->getOperandTypes(),
                       [](Type type) { return isa<TensorType>(type); });
});
```

### 2.4 Rewrite Pattern Set

The `RewritePatternSet` collects all conversion patterns to apply.

#### Creating and Using Patterns

```cpp
RewritePatternSet patterns(&getContext());

// Add conversion patterns
patterns.add<AddOpLowering, MulOpLowering>(&getContext());

// Apply patterns
if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
  signalPassFailure();
```

Two main application functions:
- **`applyPartialConversion()`**: Allows some operations to remain unconverted (marked as legal)
- **`applyFullConversion()`**: Requires all illegal operations to be converted

### 2.5 Example: Lowering Add Operation

Let's create a complete example that lowers `mydialect.add` to `arith.addi`.

#### Lowering Pattern Implementation

Create `lib/MyDialect/MyDialectToArith.cpp`:

```cpp
//===- MyDialectToArith.cpp - MyDialect to Arith lowering ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyDialect/MyDialectDialect.h"
#include "MyDialect/MyDialectOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::mydialect;

namespace {
//===----------------------------------------------------------------------===//
// AddOpLowering: Convert mydialect.add to arith.addi
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpConversionPattern<mydialect::AddOp> {
  using OpConversionPattern<mydialect::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mydialect::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the operands (already converted if using type converter)
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Create arith.addi operation
    rewriter.replaceOpWithNewOp<arith::AddIOp>(
        op, op.getType(), lhs, rhs);

    return success();
  }
};

} // namespace
```

### 2.6 Creating a Lowering Pass

A lowering pass encapsulates the conversion logic and can be run as part of a pass pipeline.

#### Pass Header

Create `include/MyDialect/MyDialectPasses.h`:

```cpp
//===- MyDialectPasses.h - MyDialect Passes Definition -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_MYDIALECTPASSES_H
#define MYDIALECT_MYDIALECTPASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace mydialect {
/// Register all MyDialect passes.
void registerPasses();

} // namespace mydialect
} // namespace mlir

#define GEN_PASS_DECL
#include "MyDialect/MyDialectPasses.h.inc"

#endif // MYDIALECT_MYDIALECTPASSES_H
```

#### Pass TableGen Definition

Create `include/MyDialect/MyDialectPasses.td`:

```tablegen
//===- MyDialectPasses.td - MyDialect Passes Definition ---*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_PASSES
#define MYDIALECT_PASSES

include "mlir/Pass/PassBase.td"

def LowerToArith : Pass<"mydialect-to-arith", "mlir::ModuleOp"> {
  let summary = "Lower MyDialect operations to Arith dialect";
  let description = [{
    This pass converts MyDialect operations to equivalent Arith dialect
    operations. For example, `mydialect.add` is converted to `arith.addi`.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif // MYDIALECT_PASSES
```

#### Pass Implementation

Create `lib/MyDialect/MyDialectPasses.cpp`:

```cpp
//===- MyDialectPasses.cpp - MyDialect Pass Implementation -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyDialect/MyDialectPasses.h"
#include "MyDialect/MyDialectDialect.h"
#include "MyDialect/MyDialectOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::mydialect;

namespace {
//===----------------------------------------------------------------------===//
// AddOpLowering: Convert mydialect.add to arith.addi
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpConversionPattern<mydialect::AddOp> {
  using OpConversionPattern<mydialect::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mydialect::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<arith::AddIOp>(
        op, op.getType(), lhs, rhs);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerToArith
//===----------------------------------------------------------------------===//

struct LowerToArith
    : public PassWrapper<LowerToArith,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToArith)

  StringRef getArgument() const override { return "mydialect-to-arith"; }
  StringRef getDescription() const override {
    return "Lower MyDialect operations to Arith dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // Mark Arith, Func, and Builtin dialects as legal
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect, BuiltinDialect>();

    // Mark MyDialect operations as illegal
    target.addIllegalDialect<MyDialectDialect>();

    // Add conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering>(&getContext());

    // Apply conversion
    if (failed(applyFullConversion(getOperation(), target,
                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// Register the pass manually
namespace mlir::mydialect {
void registerPasses() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<LowerToArith>();
  });
}
} // namespace mlir::mydialect
```

#### Update CMakeLists.txt for Passes

Update `include/MyDialect/CMakeLists.txt`:

```cmake
add_mlir_dialect(MyDialectOps mydialect)

# Generate pass declarations
set(LLVM_TARGET_DEFINITIONS MyDialectPasses.td)
mlir_tablegen(MyDialectPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMyDialectPassesIncGen)
```

Update `lib/MyDialect/CMakeLists.txt`:

```cmake
add_mlir_dialect_library(MLIRMyDialect
  MyDialectDialect.cpp
  MyDialectOps.cpp
  MyDialectPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/MyDialect

  DEPENDS
  MLIRMyDialectOpsIncGen
  MLIRMyDialectPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
)
```

### 2.7 Complete Lowering Example

#### Creating a Test Tool

Create `tools/my-dialect-opt/CMakeLists.txt`:

```cmake
set(LIBS
  MLIRArithDialect
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRMyDialect
)

add_llvm_executable(my-dialect-opt my-dialect-opt.cpp)
llvm_update_compile_flags(my-dialect-opt)
target_link_libraries(my-dialect-opt PRIVATE ${LIBS})
```

Create `tools/my-dialect-opt/my-dialect-opt.cpp`:

```cpp
//===- my-dialect-opt.cpp - MyDialect optimizer driver -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyDialect/MyDialectDialect.h"
#include "MyDialect/MyDialectPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::mydialect::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::mydialect::MyDialectDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MyDialect optimizer driver\n", registry));
}
```

Update root `CMakeLists.txt` to include tools:

```cmake
# ... existing code ...
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
```

Create `tools/CMakeLists.txt`:

```cmake
add_subdirectory(my-dialect-opt)
```

#### Testing the Lowering

Create a test MLIR file `test.mlir`:

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

Run the lowering pass:

```bash
./build/bin/my-dialect-opt test.mlir -mydialect-to-arith
```

Expected output:

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

---

## References

### Official Documentation

- **MLIR Documentation**: https://mlir.llvm.org/
- **MLIR Tutorials**: https://mlir.llvm.org/docs/Tutorials/
- **Defining Dialects**: https://mlir.llvm.org/docs/DefiningDialects/
- **Dialect Conversion**: https://mlir.llvm.org/docs/DialectConversion/
- **ClangIR (CIR Dialect)**: Refer to ClangIR documentation for building LLVM with CIR dialect support

### Example Code

- **Standalone Dialect Example**: `llvm-project/mlir/examples/standalone/`
- **Toy Language Tutorial**: `llvm-project/mlir/examples/toy/`

### Key MLIR Concepts

- **TableGen**: MLIR uses TableGen (.td files) to generate C++ code for operations, types, and attributes
- **Dialect**: A namespace for operations, types, and attributes in MLIR
- **Operations**: The basic unit of computation in MLIR IR
- **Lowering**: Converting operations from one dialect to another
- **Conversion Patterns**: Rules that specify how to transform operations during lowering
- **Conversion Target**: Specification of what operations are legal or illegal after conversion

### Building and Installation

- **LLVM Build Guide**: https://llvm.org/docs/GettingStarted.html
- **MLIR Build Instructions**: https://mlir.llvm.org/getting_started/

---

## Conclusion

This guide has covered the essential steps for creating an out-of-tree MLIR dialect:

1. **Setup**: Environment configuration and folder structure
2. **Dialect Definition**: Using TableGen to define dialects and operations
3. **Implementation**: C++ implementation files for dialect and operations
4. **Lowering**: Converting operations to other dialects using conversion patterns

With this foundation, you can extend your dialect with more operations, types, attributes, and lowering passes as needed for your specific use case.
