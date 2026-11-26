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

struct BlockPickUniforms {
  float4x4 mvp;
  int vis_col;
  char padding0[12];
  float4 line_info;
};

struct block_pick_frag_out {
  float4 f_out [[color(0)]];
  float f_prim [[color(1)]];
};

struct block_pick_frag_in {
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

fragment block_pick_frag_out
block_pick_frag(block_pick_frag_in stage_input [[stage_in]],
                texture2d<float> mesh_table [[texture(0)]],
                sampler mesh_table_sampler [[sampler(0)]],
                uint primitive_index [[primitive_id]]) {
  block_pick_frag_out stage_output = {};
  int pick_mesh_index = stage_input.v_slot;
  int pick_column = 2;
  float4 pick_metadata =
      tab(pick_mesh_index, pick_column, mesh_table, mesh_table_sampler);
  stage_output.f_out = float4(stage_input.v_world, pick_metadata.x);
  stage_output.f_prim =
      float(int(primitive_index) -
            ((int(pick_metadata.y) * 65536) + int(pick_metadata.z)));
  return stage_output;
}

static_assert(sizeof(BlockPickUniforms) == 96, "Uniform size");
static_assert(__builtin_offsetof(BlockPickUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(BlockPickUniforms, vis_col) == 64,
              "vis_col offset");
static_assert(__builtin_offsetof(BlockPickUniforms, line_info) == 80,
              "line_info offset");
