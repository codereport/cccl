//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// multimap();

#include <map>

#include "test_macros.h"

struct X
{
    std::multimap<int, X> m;
    std::multimap<int, X>::iterator i;
    std::multimap<int, X>::const_iterator ci;
#if TEST_STD_VER <= 2014
    // These reverse_iterator specializations require X to be complete in C++20 (and C++17 because of backport of ranges).
    std::multimap<int, X>::reverse_iterator ri;
    std::multimap<int, X>::const_reverse_iterator cri;
#endif // TEST_STD_VER <= 2014
};

int main(int, char**) { return 0; }
