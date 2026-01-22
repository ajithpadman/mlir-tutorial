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

