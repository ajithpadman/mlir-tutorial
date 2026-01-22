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

