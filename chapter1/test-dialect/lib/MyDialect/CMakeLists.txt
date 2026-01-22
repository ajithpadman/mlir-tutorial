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

