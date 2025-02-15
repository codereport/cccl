//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++98 || c++03 || c++11 || c++14

// <filesystem>

// namespace std::filesystem

#include <filesystem>
#include "test_macros.h"

using namespace std::filesystem;

// expected-error@-3 {{no namespace named 'filesystem' in namespace 'std';}}

int main(int, char**) {


  return 0;
}
