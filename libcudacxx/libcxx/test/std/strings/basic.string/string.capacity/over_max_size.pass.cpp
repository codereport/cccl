//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9
// XFAIL: with_system_cxx_lib=macosx10.8
// XFAIL: with_system_cxx_lib=macosx10.7

// <string>

// size_type max_size() const;

#include <string>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(const S& s)
{
    assert(s.max_size() >= s.size());
    S s2(s);
    const size_t sz = s2.max_size() + 1;
    try { s2.resize(sz, 'x'); }
    catch ( const std::length_error & ) { return ; }
    assert ( false );
}

int main(int, char**)
{
    {
    typedef std::string S;
    test(S());
    test(S("123"));
    test(S("12345678901234567890123456789012345678901234567890"));
    }
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S());
    test(S("123"));
    test(S("12345678901234567890123456789012345678901234567890"));
    }

  return 0;
}
