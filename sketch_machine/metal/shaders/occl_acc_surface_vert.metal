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

struct OcclAccSurfaceUniforms {
  float4x4 vp;
  float bias0;
  int occluders_only;
  int surf_rows;
  char padding0[4];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float occl_on;
  float xray_behind;
  char padding1[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding2[12];
  float4 line_info;
};

struct occl_acc_surface_vert_out {
  float3 v_world [[user(locn0)]];
  float3 v_normal [[user(locn1)]];
  float3 v_calm [[user(locn2)]];
  int v_slot [[user(locn3)]];
  float4 position [[position]];
};

struct occl_acc_surface_vert_in {
  float3 in_pos [[attribute(0)]];
  float3 in_normal [[attribute(1)]];
  float3 in_calm [[attribute(2)]];
  float in_slot [[attribute(3)]];
};

static inline __attribute__((always_inline)) float4
surf(thread const int& surface_index, thread const int& texel_column,
     texture2d<float> surface_table, sampler surface_table_sampler,
     constant OcclAccSurfaceUniforms& uniforms) {
  return surface_table.read(
      uint2(int2(texel_column + (10 * (surface_index / uniforms.surf_rows)),
                 signed_modulo(surface_index, uniforms.surf_rows))),
      0);
}

static inline __attribute__((always_inline)) void shade_vertex(
    texture2d<float> surface_table, sampler surface_table_sampler,
    constant OcclAccSurfaceUniforms& uniforms,
    thread float& vertex_surface_index, thread int& surface_index_output,
    thread float3& vertex_position, thread float3& normal_output,
    thread float3& vertex_normal, thread float3& smooth_normal_output,
    thread float3& vertex_smooth_normal, thread float3& world_position_output,
    thread float4& clip_position) {
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

vertex occl_acc_surface_vert_out
occl_acc_surface_vert(occl_acc_surface_vert_in stage_input [[stage_in]],
                      constant OcclAccSurfaceUniforms& uniforms [[buffer(0)]],
                      texture2d<float> surface_table [[texture(0)]],
                      sampler surface_table_sampler [[sampler(0)]]) {
  occl_acc_surface_vert_out stage_output = {};
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

static_assert(sizeof(OcclAccSurfaceUniforms) == 176, "Uniform size");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, bias0) == 64,
              "bias offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, occluders_only) == 68,
              "occluders_only offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, surf_rows) == 72,
              "surf_rows offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, eye) == 80,
              "eye offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, view_back) == 96,
              "view_back offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, ortho) == 108,
              "ortho offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, occl_on) == 112,
              "occl_on offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, xray_behind) == 116,
              "xray_behind offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, occl_tol) == 128,
              "occl_tol offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, occl_min) == 140,
              "occl_min offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, aid_pass) == 144,
              "aid_pass offset");
static_assert(__builtin_offsetof(OcclAccSurfaceUniforms, line_info) == 160,
              "line_info offset");
