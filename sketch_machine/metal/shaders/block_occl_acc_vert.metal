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

struct BlockOcclAccUniforms {
  float4x4 mvp;
  int vis_col;
  char padding0[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float occl_on;
  float xray_behind;
  char padding1[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  float acc_value;
  char padding2[8];
  float4 line_info;
};

struct block_occl_acc_vert_out {
  float3 v_world [[user(locn0)]];
  float3 v_color [[user(locn1)]];
  int v_slot [[user(locn2)]];
  float4 position [[position]];
};

struct block_occl_acc_vert_in {
  float3 in_pos [[attribute(0)]];
  float3 in_color [[attribute(1)]];
  float in_slot [[attribute(2)]];
};

static inline __attribute__((always_inline)) float4
tab(thread const int& mesh_index, thread const int& texel_column,
    texture2d<float> mesh_table, sampler mesh_table_sampler) {
  int linear_texel_index = (mesh_index * 3) + texel_column;
  return mesh_table.read(uint2(int2(signed_modulo(linear_texel_index, 1024),
                                    linear_texel_index / 1024)),
                         0);
}

static inline __attribute__((always_inline)) void shade_vertex(
    texture2d<float> mesh_table, sampler mesh_table_sampler,
    thread int& surface_index_output, thread float& vertex_surface_index,
    thread float3& world_position_output, thread float3& vertex_position,
    thread float3& color_output, thread float3& vertex_color,
    constant BlockOcclAccUniforms& uniforms, thread float4& clip_position) {
  surface_index_output = int(vertex_surface_index + 0.5);
  world_position_output = vertex_position;
  color_output = vertex_color;
  bool visibility_result;

  if (uniforms.vis_col == 2) {
    int visibility_mesh_index = surface_index_output;
    int primary_visibility_column = 2;
    visibility_result = tab(visibility_mesh_index, primary_visibility_column,
                            mesh_table, mesh_table_sampler)
                            .x > 0.5;
  } else {
    int secondary_mesh_index = surface_index_output;
    int secondary_visibility_column = 1;
    visibility_result = tab(secondary_mesh_index, secondary_visibility_column,
                            mesh_table, mesh_table_sampler)
                            .y > 0.5;
  }

  bool mesh_visible = visibility_result;
  float4 selected_clip_position;

  if (mesh_visible) {
    selected_clip_position = uniforms.mvp * float4(vertex_position, 1.0);
  } else {
    selected_clip_position = float4(2.0, 2.0, 2.0, 1.0);
  }

  clip_position = selected_clip_position;
}

vertex block_occl_acc_vert_out
block_occl_acc_vert(block_occl_acc_vert_in stage_input [[stage_in]],
                    constant BlockOcclAccUniforms& uniforms [[buffer(0)]],
                    texture2d<float> mesh_table [[texture(0)]],
                    sampler mesh_table_sampler [[sampler(0)]]) {
  block_occl_acc_vert_out stage_output = {};
  shade_vertex(mesh_table, mesh_table_sampler, stage_output.v_slot,
               stage_input.in_slot, stage_output.v_world, stage_input.in_pos,
               stage_output.v_color, stage_input.in_color, uniforms,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(BlockOcclAccUniforms) == 176, "Uniform size");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, vis_col) == 64,
              "vis_col offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, eye) == 80,
              "eye offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, view_back) == 96,
              "view_back offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, ortho) == 108,
              "ortho offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, occl_on) == 112,
              "occl_on offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, xray_behind) == 116,
              "xray_behind offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, occl_tol) == 128,
              "occl_tol offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, occl_min) == 140,
              "occl_min offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, aid_pass) == 144,
              "aid_pass offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, acc_value) == 148,
              "acc_value offset");
static_assert(__builtin_offsetof(BlockOcclAccUniforms, line_info) == 160,
              "line_info offset");
