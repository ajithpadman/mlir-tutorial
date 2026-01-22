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

#endif // MYDIALECT_OPS

