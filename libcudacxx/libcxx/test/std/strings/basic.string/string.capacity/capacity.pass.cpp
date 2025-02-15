//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// size_type capacity() const;

#include <string>
#include <cassert>

#include "test_allocator.h"
#include "min_allocator.h"

#include "test_macros.h"

template <class S>
void
test(S s)
{
    S::allocator_type::throw_after = 0;
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
#endif
    {
        while (s.size() < s.capacity())
            s.push_back(typename S::value_type());
        assert(s.size() == s.capacity());
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    catch (...)
    {
        assert(false);
    }
#endif
    S::allocator_type::throw_after = INT_MAX;
}

int main(int, char**)
{
    {
    typedef std::basic_string<char, std::char_traits<char>, test_allocator<char> > S;
    S s;
    test(s);
    s.assign(10, 'a');
    s.erase(5);
    test(s);
    s.assign(100, 'a');
    s.erase(50);
    test(s);
    }
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S s;
    assert(s.capacity() > 0);
    }

  return 0;
}
