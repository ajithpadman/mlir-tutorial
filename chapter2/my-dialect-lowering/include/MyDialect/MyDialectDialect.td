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

