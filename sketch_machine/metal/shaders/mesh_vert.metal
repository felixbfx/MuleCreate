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

struct MeshUniforms {
  float4x4 mvp;
  float4x4 model;
  float3 eye;
  float4 tint;
  float alpha;
  float lift;
  float light_on;
  char padding0[4];
  float3 sun_dir;
  float3 sun_col;
  packed_float3 sky_col;
  float away_dark;
  float shadow_dark;
  float bands;
  float shadow_on;
  char padding1[4];
  float4x4 shadow_vp;
  float shadow_bias;
  float shadow_lift;
  float dome;
  char padding2[4];
  float4 line_info;
};

struct mesh_vert_out {
  float3 v_world [[user(locn0)]];
  float3 v_color [[user(locn1)]];
  float4 position [[position]];
};

struct mesh_vert_in {
  float3 in_pos [[attribute(0)]];
  float3 in_color [[attribute(1)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& world_position_output, constant MeshUniforms& uniforms,
    thread float3& vertex_position, thread float3& color_output,
    thread float3& vertex_color, thread float4& clip_position) {
  world_position_output = (uniforms.model * float4(vertex_position, 1.0)).xyz;
  color_output = vertex_color;
  clip_position = uniforms.mvp * float4(vertex_position, 1.0);
}

vertex mesh_vert_out mesh_vert(mesh_vert_in stage_input [[stage_in]],
                               constant MeshUniforms& uniforms [[buffer(0)]]) {
  mesh_vert_out stage_output = {};
  shade_vertex(stage_output.v_world, uniforms, stage_input.in_pos,
               stage_output.v_color, stage_input.in_color,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(MeshUniforms) == 336, "Uniform size");
static_assert(__builtin_offsetof(MeshUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(MeshUniforms, model) == 64, "model offset");
static_assert(__builtin_offsetof(MeshUniforms, eye) == 128, "eye offset");
static_assert(__builtin_offsetof(MeshUniforms, tint) == 144, "tint offset");
static_assert(__builtin_offsetof(MeshUniforms, alpha) == 160, "alpha offset");
static_assert(__builtin_offsetof(MeshUniforms, lift) == 164, "lift offset");
static_assert(__builtin_offsetof(MeshUniforms, light_on) == 168,
              "light_on offset");
static_assert(__builtin_offsetof(MeshUniforms, sun_dir) == 176,
              "sun_dir offset");
static_assert(__builtin_offsetof(MeshUniforms, sun_col) == 192,
              "sun_col offset");
static_assert(__builtin_offsetof(MeshUniforms, sky_col) == 208,
              "sky_col offset");
static_assert(__builtin_offsetof(MeshUniforms, away_dark) == 220,
              "away_dark offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_dark) == 224,
              "shadow_dark offset");
static_assert(__builtin_offsetof(MeshUniforms, bands) == 228, "bands offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_on) == 232,
              "shadow_on offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_vp) == 240,
              "shadow_vp offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_bias) == 304,
              "shadow_bias offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_lift) == 308,
              "shadow_lift offset");
static_assert(__builtin_offsetof(MeshUniforms, dome) == 312, "dome offset");
static_assert(__builtin_offsetof(MeshUniforms, line_info) == 320,
              "line_info offset");
