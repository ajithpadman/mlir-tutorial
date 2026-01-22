# Generate dialect code from TableGen
add_mlir_dialect(MyDialectOps mydialect)

# Generate documentation (optional)
add_mlir_doc(MyDialectDialect MyDialectDialect MyDialect/ -gen-dialect-doc)
add_mlir_doc(MyDialectOps MyDialectOps MyDialect/ -gen-op-doc)

