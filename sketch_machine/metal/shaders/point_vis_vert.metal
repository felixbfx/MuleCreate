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

struct PointVisUniforms {
  float count;
  char padding0[12];
  float4x4 vp;
  float2 size;
  char padding1[8];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float occl_on;
  float xray_behind;
  char padding2[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding3[12];
  float4 line_info;
};

struct point_vis_vert_out {
  int2 v_xy [[user(locn0)]];
  int v_front [[user(locn1)]];
  float3 v_p [[user(locn2)]];
  float4 position [[position]];
  float point_size [[point_size]];
};

struct point_vis_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    constant PointVisUniforms& uniforms, thread float3& vertex_position,
    thread int& front_visible_output, thread int2& pixel_coordinates_output,
    thread float3& world_point_output, thread float4& clip_position,
    thread float& point_diameter, thread uint& vertex_index) {
  float4 projected_position = uniforms.vp * float4(vertex_position, 1.0);
  front_visible_output = int(projected_position.w > 0.0);
  float2 normalized_device_coordinates =
      projected_position.xy /
      float2(fast::max(abs(projected_position.w), 1e-12));
  pixel_coordinates_output = int2(floor(
      ((normalized_device_coordinates * 0.5) + float2(0.5)) * uniforms.size));
  world_point_output = vertex_position;
  point_diameter = 1.0;
  clip_position =
      float4((((float(int(vertex_index)) + 0.5) / uniforms.count) * 2.0) - 1.0,
             0.0, 0.0, 1.0);
}

vertex point_vis_vert_out point_vis_vert(point_vis_vert_in stage_input
                                         [[stage_in]],
                                         constant PointVisUniforms& uniforms
                                         [[buffer(0)]],
                                         uint vertex_index [[vertex_id]]) {
  point_vis_vert_out stage_output = {};
  shade_vertex(uniforms, stage_input.in_pos, stage_output.v_front,
               stage_output.v_xy, stage_output.v_p, stage_output.position,
               stage_output.point_size, vertex_index);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(PointVisUniforms) == 192, "Uniform size");
static_assert(__builtin_offsetof(PointVisUniforms, count) == 0, "count offset");
static_assert(__builtin_offsetof(PointVisUniforms, vp) == 16, "vp offset");
static_assert(__builtin_offsetof(PointVisUniforms, size) == 80, "size offset");
static_assert(__builtin_offsetof(PointVisUniforms, eye) == 96, "eye offset");
static_assert(__builtin_offsetof(PointVisUniforms, view_back) == 112,
              "view_back offset");
static_assert(__builtin_offsetof(PointVisUniforms, ortho) == 124,
              "ortho offset");
static_assert(__builtin_offsetof(PointVisUniforms, occl_on) == 128,
              "occl_on offset");
static_assert(__builtin_offsetof(PointVisUniforms, xray_behind) == 132,
              "xray_behind offset");
static_assert(__builtin_offsetof(PointVisUniforms, occl_tol) == 144,
              "occl_tol offset");
static_assert(__builtin_offsetof(PointVisUniforms, occl_min) == 156,
              "occl_min offset");
static_assert(__builtin_offsetof(PointVisUniforms, aid_pass) == 160,
              "aid_pass offset");
static_assert(__builtin_offsetof(PointVisUniforms, line_info) == 176,
              "line_info offset");
