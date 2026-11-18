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

struct InfiniteUniforms {
  float4x4 vp;
  float bias0;
  char padding0[12];
  float3 origin;
  float3 axis_u;
  float3 axis_v;
  float2 half_size;
  float has_core;
  char padding1[4];
  float2 tile_size;
  float2 tile_offset;
  float2 grid_step;
  char padding2[8];
  float4 fill;
  packed_float3 line;
  float grid_alpha;
  float tile_alpha;
  char padding3[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float reach;
  char padding4[12];
  float4 owner;
  float occl_on;
  float xray_behind;
  char padding5[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding6[12];
  float4 line_info;
};

struct infinite_vert_out {
  float3 v_world [[user(locn0)]];
  float4 position [[position]];
};

struct infinite_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& world_position_output, thread float3& vertex_position,
    constant InfiniteUniforms& uniforms, thread float4& clip_position) {
  world_position_output = vertex_position;
  float4 unbiased_clip_position = uniforms.vp * float4(vertex_position, 1.0);
  clip_position = float4(
      unbiased_clip_position.xy,
      unbiased_clip_position.z + (uniforms.bias0 * unbiased_clip_position.w),
      unbiased_clip_position.w);
}

vertex infinite_vert_out infinite_vert(infinite_vert_in stage_input
                                       [[stage_in]],
                                       constant InfiniteUniforms& uniforms
                                       [[buffer(0)]]) {
  infinite_vert_out stage_output = {};
  shade_vertex(stage_output.v_world, stage_input.in_pos, uniforms,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(InfiniteUniforms) == 352, "Uniform size");
static_assert(__builtin_offsetof(InfiniteUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(InfiniteUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(InfiniteUniforms, origin) == 80,
              "origin offset");
static_assert(__builtin_offsetof(InfiniteUniforms, axis_u) == 96,
              "axis_u offset");
static_assert(__builtin_offsetof(InfiniteUniforms, axis_v) == 112,
              "axis_v offset");
static_assert(__builtin_offsetof(InfiniteUniforms, half_size) == 128,
              "half_size offset");
static_assert(__builtin_offsetof(InfiniteUniforms, has_core) == 136,
              "has_core offset");
static_assert(__builtin_offsetof(InfiniteUniforms, tile_size) == 144,
              "tile_size offset");
static_assert(__builtin_offsetof(InfiniteUniforms, tile_offset) == 152,
              "tile_offset offset");
static_assert(__builtin_offsetof(InfiniteUniforms, grid_step) == 160,
              "grid_step offset");
static_assert(__builtin_offsetof(InfiniteUniforms, fill) == 176, "fill offset");
static_assert(__builtin_offsetof(InfiniteUniforms, line) == 192, "line offset");
static_assert(__builtin_offsetof(InfiniteUniforms, grid_alpha) == 204,
              "grid_alpha offset");
static_assert(__builtin_offsetof(InfiniteUniforms, tile_alpha) == 208,
              "tile_alpha offset");
static_assert(__builtin_offsetof(InfiniteUniforms, eye) == 224, "eye offset");
static_assert(__builtin_offsetof(InfiniteUniforms, view_back) == 240,
              "view_back offset");
static_assert(__builtin_offsetof(InfiniteUniforms, ortho) == 252,
              "ortho offset");
static_assert(__builtin_offsetof(InfiniteUniforms, reach) == 256,
              "reach offset");
static_assert(__builtin_offsetof(InfiniteUniforms, owner) == 272,
              "owner offset");
static_assert(__builtin_offsetof(InfiniteUniforms, occl_on) == 288,
              "occl_on offset");
static_assert(__builtin_offsetof(InfiniteUniforms, xray_behind) == 292,
              "xray_behind offset");
static_assert(__builtin_offsetof(InfiniteUniforms, occl_tol) == 304,
              "occl_tol offset");
static_assert(__builtin_offsetof(InfiniteUniforms, occl_min) == 316,
              "occl_min offset");
static_assert(__builtin_offsetof(InfiniteUniforms, aid_pass) == 320,
              "aid_pass offset");
static_assert(__builtin_offsetof(InfiniteUniforms, line_info) == 336,
              "line_info offset");
