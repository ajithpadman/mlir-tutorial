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
// Forward declaration
void registerPasses();
} // namespace mydialect
} // namespace mlir

#define GEN_PASS_DECL
#include "MyDialect/MyDialectPasses.h.inc"

#endif // MYDIALECT_MYDIALECTPASSES_H

