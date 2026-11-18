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

struct PictureUniforms {
  float4x4 vp;
  float bias0;
  char padding0[12];
  float4x4 picture_m;
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float4 owner;
  float under;
  float occl_on;
  float xray_behind;
  char padding1[4];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding2[12];
  float4 line_info;
};

struct picture_vert_out {
  float3 v_world [[user(locn0)]];
  float4 position [[position]];
};

struct picture_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& world_position_output, thread float3& vertex_position,
    constant PictureUniforms& uniforms, thread float4& clip_position) {
  world_position_output = vertex_position;
  float4 unbiased_clip_position = uniforms.vp * float4(vertex_position, 1.0);
  clip_position = float4(
      unbiased_clip_position.xy,
      unbiased_clip_position.z + (uniforms.bias0 * unbiased_clip_position.w),
      unbiased_clip_position.w);
}

vertex picture_vert_out picture_vert(picture_vert_in stage_input [[stage_in]],
                                     constant PictureUniforms& uniforms
                                     [[buffer(0)]]) {
  picture_vert_out stage_output = {};
  shade_vertex(stage_output.v_world, stage_input.in_pos, uniforms,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(PictureUniforms) == 256, "Uniform size");
static_assert(__builtin_offsetof(PictureUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(PictureUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(PictureUniforms, picture_m) == 80,
              "picture_m offset");
static_assert(__builtin_offsetof(PictureUniforms, eye) == 144, "eye offset");
static_assert(__builtin_offsetof(PictureUniforms, view_back) == 160,
              "view_back offset");
static_assert(__builtin_offsetof(PictureUniforms, ortho) == 172,
              "ortho offset");
static_assert(__builtin_offsetof(PictureUniforms, owner) == 176,
              "owner offset");
static_assert(__builtin_offsetof(PictureUniforms, under) == 192,
              "under offset");
static_assert(__builtin_offsetof(PictureUniforms, occl_on) == 196,
              "occl_on offset");
static_assert(__builtin_offsetof(PictureUniforms, xray_behind) == 200,
              "xray_behind offset");
static_assert(__builtin_offsetof(PictureUniforms, occl_tol) == 208,
              "occl_tol offset");
static_assert(__builtin_offsetof(PictureUniforms, occl_min) == 220,
              "occl_min offset");
static_assert(__builtin_offsetof(PictureUniforms, aid_pass) == 224,
              "aid_pass offset");
static_assert(__builtin_offsetof(PictureUniforms, line_info) == 240,
              "line_info offset");
