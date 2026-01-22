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

struct LowerToArith : public PassWrapper<LowerToArith, OperationPass<ModuleOp>> {
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

