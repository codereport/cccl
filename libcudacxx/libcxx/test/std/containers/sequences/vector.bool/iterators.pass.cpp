//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// iterator       begin();
// iterator       end();
// const_iterator begin()  const;
// const_iterator end()    const;
// const_iterator cbegin() const;
// const_iterator cend()   const;

#include <vector>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef bool T;
        typedef std::vector<T> C;
        C c;
        C::iterator i = c.begin();
        C::iterator j = c.end();
        assert(std::distance(i, j) == 0);
        assert(i == j);
    }
    {
        typedef bool T;
        typedef std::vector<T> C;
        const C c;
        C::const_iterator i = c.begin();
        C::const_iterator j = c.end();
        assert(std::distance(i, j) == 0);
        assert(i == j);
    }
    {
        typedef bool T;
        typedef std::vector<T> C;
        C c;
        C::const_iterator i = c.cbegin();
        C::const_iterator j = c.cend();
        assert(std::distance(i, j) == 0);
        assert(i == j);
        assert(i == c.end());
    }
    {
        typedef bool T;
        typedef std::vector<T> C;
        C::iterator i;
        C::const_iterator j;
    }
    {
        typedef bool T;
        typedef std::vector<T, min_allocator<T>> C;
        C c;
        C::iterator i = c.begin();
        C::iterator j = c.end();
        assert(std::distance(i, j) == 0);
        assert(i == j);
    }
    {
        typedef bool T;
        typedef std::vector<T, min_allocator<T>> C;
        const C c;
        C::const_iterator i = c.begin();
        C::const_iterator j = c.end();
        assert(std::distance(i, j) == 0);
        assert(i == j);
    }
    {
        typedef bool T;
        typedef std::vector<T, min_allocator<T>> C;
        C c;
        C::const_iterator i = c.cbegin();
        C::const_iterator j = c.cend();
        assert(std::distance(i, j) == 0);
        assert(i == j);
        assert(i == c.end());
    }
    {
        typedef bool T;
        typedef std::vector<T, min_allocator<T>> C;
        C::iterator i;
        C::const_iterator j;
    }
#if TEST_STD_VER > 2011
    { // N3644 testing
        std::vector<bool>::iterator ii1{}, ii2{};
        std::vector<bool>::iterator ii4 = ii1;
        std::vector<bool>::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );

        assert (!(ii1 != ii2 ));

        assert ( (ii1 == cii ));
        assert ( (cii == ii1 ));
        assert (!(ii1 != cii ));
        assert (!(cii != ii1 ));
        assert (!(ii1 <  cii ));
        assert (!(cii <  ii1 ));
        assert ( (ii1 <= cii ));
        assert ( (cii <= ii1 ));
        assert (!(ii1 >  cii ));
        assert (!(cii >  ii1 ));
        assert ( (ii1 >= cii ));
        assert ( (cii >= ii1 ));
        assert (cii - ii1 == 0);
        assert (ii1 - cii == 0);
    }
#endif

  return 0;
}
