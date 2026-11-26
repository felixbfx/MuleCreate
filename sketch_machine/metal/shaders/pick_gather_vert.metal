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
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <simd/simd.h>

#include <metal_stdlib>

using namespace metal;

struct PickGatherUniforms {
  float count;
  char padding0[12];
  float4 line_info;
};

struct pick_gather_vert_out {
  int2 v_xy [[user(locn0)]];
  float4 position [[position]];
  float point_size [[point_size]];
};

struct pick_gather_vert_in {
  float2 in_xy [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread int2& pixel_coordinates_output,
    thread float2& vertex_pixel_coordinates, thread float4& clip_position,
    thread float& point_diameter, thread uint& vertex_index,
    constant PickGatherUniforms& uniforms) {
  pixel_coordinates_output = int2(vertex_pixel_coordinates);
  point_diameter = 1.0;
  clip_position =
      float4((((float(int(vertex_index)) + 0.5) / uniforms.count) * 2.0) - 1.0,
             0.0, 0.0, 1.0);
}

vertex pick_gather_vert_out
pick_gather_vert(pick_gather_vert_in stage_input [[stage_in]],
                 constant PickGatherUniforms& uniforms [[buffer(0)]],
                 uint vertex_index [[vertex_id]]) {
  pick_gather_vert_out stage_output = {};
  shade_vertex(stage_output.v_xy, stage_input.in_xy, stage_output.position,
               stage_output.point_size, vertex_index, uniforms);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(PickGatherUniforms) == 32, "Uniform size");
static_assert(__builtin_offsetof(PickGatherUniforms, count) == 0,
              "count offset");
static_assert(__builtin_offsetof(PickGatherUniforms, line_info) == 16,
              "line_info offset");
