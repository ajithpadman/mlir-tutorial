set(LIBS
  MLIRArithDialect
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRMyDialect
)

add_llvm_executable(my-dialect-opt my-dialect-opt.cpp)
llvm_update_compile_flags(my-dialect-opt)
target_link_libraries(my-dialect-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(my-dialect-opt)

