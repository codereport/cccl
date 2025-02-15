//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ostreambuf_iterator

#include <iterator>

#include <ostream>
#include <string>

using iterator = std::ostreambuf_iterator<char>;
static_assert(!std::indirectly_readable<iterator>);
static_assert(std::indirectly_writable<iterator, char>);
static_assert(std::weakly_incrementable<iterator>);
static_assert(std::input_or_output_iterator<iterator>);
static_assert(!std::sentinel_for<iterator, iterator>);
static_assert(!std::sized_sentinel_for<iterator, iterator>);
static_assert(!std::input_iterator<iterator>);
static_assert(std::indirectly_movable<char*, iterator>);
static_assert(std::indirectly_movable_storable<char*, iterator>);
static_assert(std::indirectly_copyable<char*, iterator>);
static_assert(std::indirectly_copyable_storable<char*, iterator>);
static_assert(!std::indirectly_swappable<iterator, iterator>);

int main(int, char**)
{
  return 0;
}
