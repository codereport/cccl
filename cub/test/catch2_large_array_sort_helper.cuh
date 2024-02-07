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

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/memory.h>
#include <thrust/random.h>
#include <thrust/reverse.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstdint>
#include <limits>

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

template <typename key_type>
struct summary
{
  key_type key{};
  std::size_t index{};
  std::size_t count{};
};

template <typename key_type>
struct index_to_summary
{
  using summary_t = summary<key_type>;

  std::size_t num_items;
  std::size_t num_summaries;

  template <typename index_type>
  _CCCL_HOST_DEVICE summary_t operator()(index_type idx) const
  {
    constexpr key_type max_key = ::cuda::std::numeric_limits<key_type>::max();

    const double key_conversion = static_cast<double>(max_key) / static_cast<double>(num_summaries);
    const key_type key          = static_cast<key_type>(idx * key_conversion);

    const std::size_t elements_per_summary = num_items / num_summaries;
    const std::size_t run_index            = idx * elements_per_summary;

    const std::size_t run_size = idx == (num_summaries - 1) ? (num_items - run_index) : elements_per_summary;

    return summary_t{key, run_index, run_size};
  }
};

template <typename key_type, typename summary_iter_t>
struct summary_to_sorted_keys
{
  using summary_t = typename ::cuda::std::iterator_traits<summary_iter_t>::value_type;

  summary_iter_t summary_begin;
  summary_iter_t summary_end;

  struct binary_search_op
  {
    bool operator()(const summary_t& summary, std::size_t idx) const
    {
      return summary.index + summary.count < idx;
    }
  };

  template <typename index_type>
  key_type operator()(index_type idx) const
  {
    auto summary_iter =
      thrust::lower_bound(thrust::seq, summary_begin, summary_end, static_cast<std::size_t>(idx), binary_search_op{});
    return summary_iter->key;
  }
};

template <typename value_type>
struct index_to_value
{
  template <typename index_type>
  value_type operator()(index_type index)
  {
    return static_cast<value_type>(index);
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
    thrust::fill(c2h::device_policy, keys_out.begin(), keys_out.end(), key_type{});
    timer.print_elapsed_seconds_and_reset("Reset Output");
  }

  void initialize_for_stable_key_value_sort(std::size_t num_items, bool is_descending)
  {
    static_assert(!::cuda::std::is_same<value_type, cub::NullType>::value, "value_type must be valid.");
    using summary_t            = detail::summary<key_type>;
    constexpr key_type max_key = ::cuda::std::numeric_limits<key_type>::max();

    // Build the sorted reference arrays on the device in keys/values_out:
    keys_out.resize(num_items);
    values_out.resize(num_items);

    const std::size_t max_summary_mem = num_items * (sizeof(key_type) + sizeof(value_type));
    const std::size_t max_summaries   = cub::DivideAndRoundUp(max_summary_mem, sizeof(summary_t));
    const std::size_t num_summaries   = std::min(max_summaries, num_items, static_cast<std::size_t>(max_key));

    c2h::device_vector<summary_t> d_summaries;
    // Overallocate -- if this fails, there won't be be enough free device memory for the input arrays.
    // Better to fail now before spending time computing the inputs/outputs.
    d_summaries.reserve(max_summaries);
    d_summaries.resize(num_summaries);

    // Populate the summaries using evenly spaced keys and constant sized runs, padding the last run to fill.
    thrust::tabulate(c2h::device_policy,
                     d_summaries.begin(),
                     d_summaries.end(),
                     detail::index_to_summary<key_type>{num_items, num_summaries});

    // Use the summaries to populate the sorted keys
    using summary_iterator_t = typename c2h::device_vector<summary_t>::const_iterator;
    thrust::tabulate(
      c2h::device_policy,
      keys_out.begin(),
      keys_out.end(),
      detail::summary_to_sorted_keys<key_type, summary_iterator_t>{d_summaries.cbegin(), d_summaries.cend()});

    // The sorted values will be the index truncated to the value type:
    thrust::tabulate(c2h::device_policy, values_out.begin(), values_out.end(), detail::index_to_value<value_type>{});

    // Copy the summaries to host memory and release device summary memory.
    c2h::host_vector<summary_t> h_summaries = d_summaries;
    d_summaries.clear();
    d_summaries.shrink_to_fit();

    // Build the unsorted key/value arrays on host in key/value_ref using Andy's algorithm.

    // Release the host summary memory.
    // Copy the unsorted data from key/value_ref -> key/value_in.
    // Copy the sorted keys/values_out to host key/value_ref.
    // Clear the output arrays to prevent no-op issues.
  }
};
