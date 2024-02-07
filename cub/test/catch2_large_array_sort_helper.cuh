/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/util_type.cuh>

#include <thrust/memory.h>
#include <thrust/random.h>
#include <thrust/reverse.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/std/limits>

#include <cstdint>

#include "thrust/random.h"
#include <c2h/cpu_timer.cuh>
#include <c2h/device_policy.cuh>
#include <c2h/vector.cuh>

namespace detail
{
template <typename key_type>
struct index_to_sorted_key
{
  std::size_t num_items;

  template <typename index_type>
  _CCCL_HOST_DEVICE key_type operator()(index_type idx) const
  {
    constexpr double max_key = static_cast<double>(::cuda::std::numeric_limits<key_type>::max());
    const double conversion  = max_key / num_items;
    return static_cast<key_type>(idx * conversion);
  }
};
} // namespace detail

template <typename key_type, typename value_type = cub::NullType>
struct large_array_sort_helper
{
  // Sorted keys/values in host memory
  c2h::host_vector<key_type> keys_ref;
  c2h::host_vector<value_type> values_ref;

  // Unsorted keys/values in device memory
  c2h::device_vector<key_type> keys_in;
  c2h::device_vector<value_type> values_in;

  // Allocated device memory for output keys/values
  c2h::device_vector<key_type> keys_out;
  c2h::device_vector<value_type> values_out;

  // Double buffer for keys/values. Aliases the in/out arrays.
  cub::DoubleBuffer<key_type> keys_buffer;
  cub::DoubleBuffer<value_type> values_buffer;

  void initialize_for_unstable_key_sort(std::size_t num_items, bool is_descending)
  {
    c2h::cpu_timer timer;

    // Preallocate device memory ASAP so we fail quickly on bad_alloc
    keys_in.resize(num_items);
    keys_out.resize(num_items);
    keys_buffer =
      cub::DoubleBuffer<key_type>(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()));

    timer.print_elapsed_seconds_and_reset("Device Alloc");
    thrust::tabulate(
      c2h::device_policy, keys_out.begin(), keys_out.end(), detail::index_to_sorted_key<key_type>{num_items});
    timer.print_elapsed_seconds_and_reset("Tabulate");
    if (is_descending)
    {
      thrust::reverse(c2h::device_policy, keys_out.begin(), keys_out.end());
      timer.print_elapsed_seconds_and_reset("Reverse");
    }
    thrust::shuffle_copy(
      c2h::device_policy, keys_out.cbegin(), keys_out.cend(), keys_in.begin(), thrust::default_random_engine{});
    timer.print_elapsed_seconds_and_reset("Shuffle");
    keys_ref = keys_out;
    timer.print_elapsed_seconds_and_reset("D->H Copy Ref");
  }
};
