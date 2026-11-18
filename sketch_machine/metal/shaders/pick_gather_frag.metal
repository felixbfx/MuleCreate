/*
 * Mule Create
 * Copyright (C) 2026 Felix Barthel
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This file is part of Mule Create.
 *
 * Mule Create is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, version 3 of the License.
 *
 * Mule Create is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Mule Create. If not, see <https://www.gnu.org/licenses/>.
 */

// Native Metal shader. Maintain this source directly; metal/build.py compiles
// it.
#include <simd/simd.h>

#include <metal_stdlib>

using namespace metal;

struct PickGatherUniforms {
  float count;
  char padding0[12];
  float4 line_info;
};

struct pick_gather_frag_out {
  float4 f_out [[color(0)]];
  float f_prim [[color(1)]];
};

struct pick_gather_frag_in {
  int2 v_xy [[user(locn0)]];
};

fragment pick_gather_frag_out
pick_gather_frag(pick_gather_frag_in stage_input [[stage_in]],
                 texture2d<float> pick_texture [[texture(0)]],
                 texture2d<float> primitive_texture [[texture(1)]],
                 sampler pick_sampler [[sampler(0)]],
                 sampler primitive_sampler [[sampler(1)]]) {
  pick_gather_frag_out stage_output = {};
  int2 sample_pixel = clamp(
      stage_input.v_xy, int2(0),
      int2(pick_texture.get_width(), pick_texture.get_height()) - int2(1));
  stage_output.f_out = pick_texture.read(uint2(sample_pixel), 0);
  stage_output.f_prim = primitive_texture.read(uint2(sample_pixel), 0).x;
  return stage_output;
}

static_assert(sizeof(PickGatherUniforms) == 32, "Uniform size");
static_assert(__builtin_offsetof(PickGatherUniforms, count) == 0,
              "count offset");
static_assert(__builtin_offsetof(PickGatherUniforms, line_info) == 16,
              "line_info offset");
