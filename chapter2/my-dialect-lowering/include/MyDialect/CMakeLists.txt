# Generate dialect code from TableGen
add_mlir_dialect(MyDialectOps mydialect)

# Generate pass declarations
set(LLVM_TARGET_DEFINITIONS MyDialectPasses.td)
mlir_tablegen(MyDialectPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMyDialectPassesIncGen)

# Generate documentation (optional)
add_mlir_doc(MyDialectDialect MyDialectDialect MyDialect/ -gen-dialect-doc)
add_mlir_doc(MyDialectOps MyDialectOps MyDialect/ -gen-op-doc)

