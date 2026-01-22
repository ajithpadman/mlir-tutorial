add_mlir_dialect_library(MLIRMyDialect
  MyDialectDialect.cpp
  MyDialectOps.cpp
  MyDialectPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/MyDialect

  DEPENDS
  MLIRMyDialectOpsIncGen
  MLIRMyDialectPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
)

