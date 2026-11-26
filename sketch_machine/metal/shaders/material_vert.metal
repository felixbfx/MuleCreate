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

// Implementation of signed integer mod accurate to SPIR-V specification
template <typename DividendType, typename DivisorType>
inline DividendType signed_modulo(DividendType dividend, DivisorType divisor) {
  DividendType remainder = dividend - divisor * (dividend / divisor);
  return select(DividendType(remainder + divisor), remainder,
                remainder == 0 || (dividend >= 0) == (divisor >= 0));
}

struct MaterialUniforms {
  float4x4 vp;
  float bias0;
  int occluders_only;
  int surf_rows;
  float lift_min;
  float stain;
  float layer;
  float front_alpha;
  float mono;
  float light_on;
  char padding0[12];
  float3 sun_dir;
  float3 sun_col;
  packed_float3 sky_col;
  float away_dark;
  float shadow_dark;
  float calm;
  float bands;
  float shadow_on;
  float4x4 shadow_vp;
  float shadow_bias;
  float shadow_lift;
  float dome;
  char padding1[4];
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

struct material_vert_out {
  float3 v_world [[user(locn0)]];
  float3 v_normal [[user(locn1)]];
  float3 v_calm [[user(locn2)]];
  int v_slot [[user(locn3)]];
  float4 position [[position]];
};

struct material_vert_in {
  float3 in_pos [[attribute(0)]];
  float3 in_normal [[attribute(1)]];
  float3 in_calm [[attribute(2)]];
  float in_slot [[attribute(3)]];
};

static inline __attribute__((always_inline)) float4
surf(thread const int& surface_index, thread const int& texel_column,
     texture2d<float> surface_table, sampler surface_table_sampler,
     constant MaterialUniforms& uniforms) {
  return surface_table.read(
      uint2(int2(texel_column + (10 * (surface_index / uniforms.surf_rows)),
                 signed_modulo(surface_index, uniforms.surf_rows))),
      0);
}

static inline __attribute__((always_inline)) void shade_vertex(
    texture2d<float> surface_table, sampler surface_table_sampler,
    constant MaterialUniforms& uniforms, thread float& vertex_surface_index,
    thread int& surface_index_output, thread float3& vertex_position,
    thread float3& normal_output, thread float3& vertex_normal,
    thread float3& smooth_normal_output, thread float3& vertex_smooth_normal,
    thread float3& world_position_output, thread float4& clip_position) {
  int surface_index = int(vertex_surface_index + 0.5);
  surface_index_output = surface_index;
  float3 world_position = vertex_position;
  normal_output = vertex_normal;
  smooth_normal_output = vertex_smooth_normal;
  bool occluder_pass = uniforms.occluders_only == 1;
  bool surface_excluded;

  if (occluder_pass) {
    int occluder_surface_index = surface_index;
    int occluder_column = 6;
    surface_excluded = surf(occluder_surface_index, occluder_column,
                            surface_table, surface_table_sampler, uniforms)
                           .z < 0.5;
  } else {
    surface_excluded = occluder_pass;
  }

  if (surface_excluded) {
    world_position_output = world_position;
    clip_position = float4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  int transform_surface_index = surface_index;
  int transform_column = 5;

  if (surf(transform_surface_index, transform_column, surface_table,
           surface_table_sampler, uniforms)
          .w > 0.5) {
    int transform_row0_surface_index = surface_index;
    int transform_row0_column = 7;
    int transform_row1_surface_index = surface_index;
    int transform_row1_column = 8;
    int transform_row2_surface_index = surface_index;
    int transform_row2_column = 9;
    float4x4 surface_transform = transpose(float4x4(
        float4(surf(transform_row0_surface_index, transform_row0_column,
                    surface_table, surface_table_sampler, uniforms)),
        float4(surf(transform_row1_surface_index, transform_row1_column,
                    surface_table, surface_table_sampler, uniforms)),
        float4(surf(transform_row2_surface_index, transform_row2_column,
                    surface_table, surface_table_sampler, uniforms)),
        float4(0.0, 0.0, 0.0, 1.0)));
    float3x3 normal_transform =
        float3x3(surface_transform[0].xyz, surface_transform[1].xyz,
                 surface_transform[2].xyz);
    world_position = (surface_transform * float4(vertex_position, 1.0)).xyz;
    normal_output = fast::normalize(normal_transform * vertex_normal) *
                    length(vertex_normal);
    smooth_normal_output =
        fast::normalize(normal_transform * vertex_smooth_normal) *
        length(vertex_smooth_normal);
  }

  world_position_output = world_position;
  float4 unbiased_clip_position = uniforms.vp * float4(world_position, 1.0);
  clip_position = float4(
      unbiased_clip_position.xy,
      unbiased_clip_position.z - (uniforms.bias0 * unbiased_clip_position.w),
      unbiased_clip_position.w);
}

vertex material_vert_out
material_vert(material_vert_in stage_input [[stage_in]],
              constant MaterialUniforms& uniforms [[buffer(0)]],
              texture2d<float> surface_table [[texture(0)]],
              sampler surface_table_sampler [[sampler(0)]]) {
  material_vert_out stage_output = {};
  shade_vertex(surface_table, surface_table_sampler, uniforms,
               stage_input.in_slot, stage_output.v_slot, stage_input.in_pos,
               stage_output.v_normal, stage_input.in_normal,
               stage_output.v_calm, stage_input.in_calm, stage_output.v_world,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(MaterialUniforms) == 352, "Uniform size");
static_assert(__builtin_offsetof(MaterialUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(MaterialUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(MaterialUniforms, occluders_only) == 68,
              "occluders_only offset");
static_assert(__builtin_offsetof(MaterialUniforms, surf_rows) == 72,
              "surf_rows offset");
static_assert(__builtin_offsetof(MaterialUniforms, lift_min) == 76,
              "lift_min offset");
static_assert(__builtin_offsetof(MaterialUniforms, stain) == 80,
              "stain offset");
static_assert(__builtin_offsetof(MaterialUniforms, layer) == 84,
              "layer offset");
static_assert(__builtin_offsetof(MaterialUniforms, front_alpha) == 88,
              "front_alpha offset");
static_assert(__builtin_offsetof(MaterialUniforms, mono) == 92, "mono offset");
static_assert(__builtin_offsetof(MaterialUniforms, light_on) == 96,
              "light_on offset");
static_assert(__builtin_offsetof(MaterialUniforms, sun_dir) == 112,
              "sun_dir offset");
static_assert(__builtin_offsetof(MaterialUniforms, sun_col) == 128,
              "sun_col offset");
static_assert(__builtin_offsetof(MaterialUniforms, sky_col) == 144,
              "sky_col offset");
static_assert(__builtin_offsetof(MaterialUniforms, away_dark) == 156,
              "away_dark offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_dark) == 160,
              "shadow_dark offset");
static_assert(__builtin_offsetof(MaterialUniforms, calm) == 164, "calm offset");
static_assert(__builtin_offsetof(MaterialUniforms, bands) == 168,
              "bands offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_on) == 172,
              "shadow_on offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_vp) == 176,
              "shadow_vp offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_bias) == 240,
              "shadow_bias offset");
static_assert(__builtin_offsetof(MaterialUniforms, shadow_lift) == 244,
              "shadow_lift offset");
static_assert(__builtin_offsetof(MaterialUniforms, dome) == 248, "dome offset");
static_assert(__builtin_offsetof(MaterialUniforms, eye) == 256, "eye offset");
static_assert(__builtin_offsetof(MaterialUniforms, view_back) == 272,
              "view_back offset");
static_assert(__builtin_offsetof(MaterialUniforms, ortho) == 284,
              "ortho offset");
static_assert(__builtin_offsetof(MaterialUniforms, occl_on) == 288,
              "occl_on offset");
static_assert(__builtin_offsetof(MaterialUniforms, xray_behind) == 292,
              "xray_behind offset");
static_assert(__builtin_offsetof(MaterialUniforms, occl_tol) == 304,
              "occl_tol offset");
static_assert(__builtin_offsetof(MaterialUniforms, occl_min) == 316,
              "occl_min offset");
static_assert(__builtin_offsetof(MaterialUniforms, aid_pass) == 320,
              "aid_pass offset");
static_assert(__builtin_offsetof(MaterialUniforms, line_info) == 336,
              "line_info offset");
