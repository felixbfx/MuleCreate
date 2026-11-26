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

struct OcclSurfaceUniforms {
  float4x4 vp;
  float bias0;
  int occluders_only;
  int surf_rows;
  char padding0[4];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float4 line_info;
};

struct occl_surface_frag_out {
  float4 f_cover [[color(0)]];
};

struct occl_surface_frag_in {
  float3 v_world [[user(locn0)]];
  int v_slot [[user(locn3)]];
};

static inline __attribute__((always_inline)) float4
surf(thread const int& surface_index, thread const int& texel_column,
     texture2d<float> surface_table, sampler surface_table_sampler,
     constant OcclSurfaceUniforms& uniforms) {
  return surface_table.read(
      uint2(int2(texel_column + (10 * (surface_index / uniforms.surf_rows)),
                 signed_modulo(surface_index, uniforms.surf_rows))),
      0);
}

fragment occl_surface_frag_out
occl_surface_frag(occl_surface_frag_in stage_input [[stage_in]],
                  constant OcclSurfaceUniforms& uniforms [[buffer(0)]],
                  texture2d<float> surface_table [[texture(0)]],
                  sampler surface_table_sampler [[sampler(0)]]) {
  occl_surface_frag_out stage_output = {};
  int radius_surface_index = stage_input.v_slot;
  int radius_column = 6;
  float surface_radius = surf(radius_surface_index, radius_column,
                              surface_table, surface_table_sampler, uniforms)
                             .x;
  int owner_surface_index = stage_input.v_slot;
  int owner_column = 5;
  float mesh_identifier = surf(owner_surface_index, owner_column, surface_table,
                               surface_table_sampler, uniforms)
                              .z;
  float3 surface_normal = fast::normalize(
      cross(dfdx(stage_input.v_world), dfdy(stage_input.v_world)));
  float3 selected_view_direction;

  if (uniforms.ortho > 0.5) {
    selected_view_direction = float3(uniforms.view_back);
  } else {
    selected_view_direction =
        fast::normalize(uniforms.eye - stage_input.v_world);
  }

  float3 view_direction = selected_view_direction;
  float selected_view_distance;

  if (uniforms.ortho > 0.5) {
    selected_view_distance =
        dot(uniforms.eye - stage_input.v_world, float3(uniforms.view_back));
  } else {
    selected_view_distance = length(stage_input.v_world - uniforms.eye);
  }

  float view_distance = selected_view_distance;
  float world_pixel_size = length(fwidth(stage_input.v_world));
  stage_output.f_cover = float4(
      view_distance, abs(dot(surface_normal, view_direction)),
      fast::max(surface_radius, 3.0 * world_pixel_size), mesh_identifier);
  return stage_output;
}

static_assert(sizeof(OcclSurfaceUniforms) == 128, "Uniform size");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, bias0) == 64,
              "bias offset");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, occluders_only) == 68,
              "occluders_only offset");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, surf_rows) == 72,
              "surf_rows offset");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, eye) == 80, "eye offset");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, view_back) == 96,
              "view_back offset");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, ortho) == 108,
              "ortho offset");
static_assert(__builtin_offsetof(OcclSurfaceUniforms, line_info) == 112,
              "line_info offset");
