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

struct FlatUniforms {
  float4x4 vp;
  float bias0;
  char padding0[12];
  float4 color;
  float grain;
  char padding1[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float milk;
  char padding2[12];
  float4 owner;
  float under;
  float soft;
  float occl_on;
  float xray_behind;
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding3[12];
  float4 line_info;
};

struct flat_vert_out {
  float3 v_world [[user(locn0)]];
  float4 position [[position]];
};

struct flat_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& world_position_output, thread float3& vertex_position,
    constant FlatUniforms& uniforms, thread float4& clip_position) {
  world_position_output = vertex_position;
  float4 unbiased_clip_position = uniforms.vp * float4(vertex_position, 1.0);
  clip_position = float4(
      unbiased_clip_position.xy,
      unbiased_clip_position.z + (uniforms.bias0 * unbiased_clip_position.w),
      unbiased_clip_position.w);
}

vertex flat_vert_out flat_vert(flat_vert_in stage_input [[stage_in]],
                               constant FlatUniforms& uniforms [[buffer(0)]]) {
  flat_vert_out stage_output = {};
  shade_vertex(stage_output.v_world, stage_input.in_pos, uniforms,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(FlatUniforms) == 240, "Uniform size");
static_assert(__builtin_offsetof(FlatUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(FlatUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(FlatUniforms, color) == 80, "color offset");
static_assert(__builtin_offsetof(FlatUniforms, grain) == 96, "grain offset");
static_assert(__builtin_offsetof(FlatUniforms, eye) == 112, "eye offset");
static_assert(__builtin_offsetof(FlatUniforms, view_back) == 128,
              "view_back offset");
static_assert(__builtin_offsetof(FlatUniforms, ortho) == 140, "ortho offset");
static_assert(__builtin_offsetof(FlatUniforms, milk) == 144, "milk offset");
static_assert(__builtin_offsetof(FlatUniforms, owner) == 160, "owner offset");
static_assert(__builtin_offsetof(FlatUniforms, under) == 176, "under offset");
static_assert(__builtin_offsetof(FlatUniforms, soft) == 180, "soft offset");
static_assert(__builtin_offsetof(FlatUniforms, occl_on) == 184,
              "occl_on offset");
static_assert(__builtin_offsetof(FlatUniforms, xray_behind) == 188,
              "xray_behind offset");
static_assert(__builtin_offsetof(FlatUniforms, occl_tol) == 192,
              "occl_tol offset");
static_assert(__builtin_offsetof(FlatUniforms, occl_min) == 204,
              "occl_min offset");
static_assert(__builtin_offsetof(FlatUniforms, aid_pass) == 208,
              "aid_pass offset");
static_assert(__builtin_offsetof(FlatUniforms, line_info) == 224,
              "line_info offset");
