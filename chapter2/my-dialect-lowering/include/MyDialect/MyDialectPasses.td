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

