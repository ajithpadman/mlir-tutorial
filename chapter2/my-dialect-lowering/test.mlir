module {
  func.func @main() {
    %0 = arith.constant 5 : i32
    %1 = arith.constant 3 : i32
    %2 = mydialect.add %0, %1 : i32 -> i32
    return
  }
}

