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

struct BlockOcclUniforms {
  float4x4 mvp;
  int vis_col;
  char padding0[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float radius;
  char padding1[12];
  float4 line_info;
};

struct block_occl_frag_out {
  float4 f_cover [[color(0)]];
};

struct block_occl_frag_in {
  float3 v_world [[user(locn0)]];
  int v_slot [[user(locn2)]];
};

static inline __attribute__((always_inline)) float4
tab(thread const int& mesh_index, thread const int& texel_column,
    texture2d<float> mesh_table, sampler mesh_table_sampler) {
  int linear_texel_index = (mesh_index * 3) + texel_column;
  return mesh_table.read(uint2(int2(signed_modulo(linear_texel_index, 1024),
                                    linear_texel_index / 1024)),
                         0);
}

fragment block_occl_frag_out
block_occl_frag(block_occl_frag_in stage_input [[stage_in]],
                constant BlockOcclUniforms& uniforms [[buffer(0)]],
                texture2d<float> mesh_table [[texture(0)]],
                sampler mesh_table_sampler [[sampler(0)]]) {
  block_occl_frag_out stage_output = {};
  int owner_mesh_index = stage_input.v_slot;
  int owner_column = 1;
  float mesh_identifier =
      tab(owner_mesh_index, owner_column, mesh_table, mesh_table_sampler).x;
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
      fast::max(uniforms.radius, 3.0 * world_pixel_size), mesh_identifier);
  return stage_output;
}

static_assert(sizeof(BlockOcclUniforms) == 144, "Uniform size");
static_assert(__builtin_offsetof(BlockOcclUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(BlockOcclUniforms, vis_col) == 64,
              "vis_col offset");
static_assert(__builtin_offsetof(BlockOcclUniforms, eye) == 80, "eye offset");
static_assert(__builtin_offsetof(BlockOcclUniforms, view_back) == 96,
              "view_back offset");
static_assert(__builtin_offsetof(BlockOcclUniforms, ortho) == 108,
              "ortho offset");
static_assert(__builtin_offsetof(BlockOcclUniforms, radius) == 112,
              "radius offset");
static_assert(__builtin_offsetof(BlockOcclUniforms, line_info) == 128,
              "line_info offset");
