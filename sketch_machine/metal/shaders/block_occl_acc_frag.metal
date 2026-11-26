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

struct block_occl_acc_frag_out {
  float f_acc [[color(0)]];
};

struct block_occl_acc_frag_in {
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

static inline __attribute__((always_inline)) float view_dist(
    thread const float3& world_position,
    constant BlockOcclAccUniforms& uniforms) {
  float distance_to_eye;

  if (uniforms.ortho > 0.5) {
    distance_to_eye =
        dot(uniforms.eye - world_position, float3(uniforms.view_back));
  } else {
    distance_to_eye = length(world_position - uniforms.eye);
  }

  return distance_to_eye;
}

static inline __attribute__((always_inline)) bool front_here(
    thread const float3& world_position, thread const float4& owner_ids,
    constant BlockOcclAccUniforms& uniforms, texture2d<float> occlusion_map,
    sampler occlusion_sampler, thread float4& fragment_position) {
  if (uniforms.occl_on < 0.5) {
    return true;
  }

  float4 front_occluder =
      occlusion_map.read(uint2(int2(fragment_position.xy)), 0);

  if (front_occluder.z <= 0.0) {
    return true;
  }

  float3 query_position = world_position;
  float view_distance = view_dist(query_position, uniforms);
  float depth_difference = view_distance - front_occluder.x;
  float depth_epsilon = 2e-5 * abs(view_distance);

  if (depth_difference <= depth_epsilon) {
    return true;
  }

  bool front_owner_valid = front_occluder.w >= 0.0;
  bool shares_owner_id;

  if (front_owner_valid) {
    shares_owner_id = any(owner_ids == float4(front_occluder.w));
  } else {
    shares_owner_id = front_owner_valid;
  }

  bool same_owner = shares_owner_id;
  bool known_front_owner = front_occluder.w >= 0.0;
  bool both_owners_known;

  if (known_front_owner) {
    both_owners_known = fast::max(fast::max(owner_ids.x, owner_ids.y),
                                  fast::max(owner_ids.z, owner_ids.w)) >= 0.0;
  } else {
    both_owners_known = known_front_owner;
  }

  bool owners_known = both_owners_known;
  float owner_tolerance;

  if (same_owner) {
    owner_tolerance = uniforms.occl_tol[0u];
  } else {
    float foreign_tolerance;

    if (owners_known) {
      foreign_tolerance = uniforms.occl_tol[2u];
    } else {
      foreign_tolerance = uniforms.occl_tol[1u];
    }

    owner_tolerance = foreign_tolerance;
  }

  float depth_tolerance = owner_tolerance;
  bool within_projected_tolerance =
      (depth_difference * front_occluder.y) <=
      ((depth_tolerance * front_occluder.z) + depth_epsilon);
  bool within_depth_tolerance;

  if (within_projected_tolerance) {
    within_depth_tolerance =
        depth_difference <= ((same_owner ? 4.0 : 2.0) * front_occluder.z);
  } else {
    within_depth_tolerance = within_projected_tolerance;
  }

  return within_depth_tolerance;
}

fragment block_occl_acc_frag_out
block_occl_acc_frag(block_occl_acc_frag_in stage_input [[stage_in]],
                    constant BlockOcclAccUniforms& uniforms [[buffer(0)]],
                    texture2d<float> occlusion_map [[texture(0)]],
                    texture2d<float> mesh_table [[texture(1)]],
                    sampler occlusion_sampler [[sampler(0)]],
                    sampler mesh_table_sampler [[sampler(1)]],
                    float4 fragment_position [[position]]) {
  block_occl_acc_frag_out stage_output = {};
  int owner_mesh_index = stage_input.v_slot;
  int owner_column = 1;
  float4 owner_ids = float4(
      tab(owner_mesh_index, owner_column, mesh_table, mesh_table_sampler).x,
      -1.0, -1.0, -1.0);
  float3 query_position = stage_input.v_world;
  float4 query_owner_ids = owner_ids;

  if (!front_here(query_position, query_owner_ids, uniforms, occlusion_map,
                  occlusion_sampler, fragment_position)) {
    discard_fragment();
  }

  stage_output.f_acc = uniforms.acc_value;
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
