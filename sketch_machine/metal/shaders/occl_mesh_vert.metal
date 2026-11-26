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

struct OcclMeshUniforms {
  float4x4 mvp;
  float4x4 model;
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float radius;
  float mesh_id;
  char padding0[8];
  float4 line_info;
};

struct occl_mesh_vert_out {
  float3 v_world [[user(locn0)]];
  float4 position [[position]];
};

struct occl_mesh_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& world_position_output, constant OcclMeshUniforms& uniforms,
    thread float3& vertex_position, thread float4& clip_position) {
  world_position_output = (uniforms.model * float4(vertex_position, 1.0)).xyz;
  clip_position = uniforms.mvp * float4(vertex_position, 1.0);
}

vertex occl_mesh_vert_out occl_mesh_vert(occl_mesh_vert_in stage_input
                                         [[stage_in]],
                                         constant OcclMeshUniforms& uniforms
                                         [[buffer(0)]]) {
  occl_mesh_vert_out stage_output = {};
  shade_vertex(stage_output.v_world, uniforms, stage_input.in_pos,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(OcclMeshUniforms) == 192, "Uniform size");
static_assert(__builtin_offsetof(OcclMeshUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(OcclMeshUniforms, model) == 64,
              "model offset");
static_assert(__builtin_offsetof(OcclMeshUniforms, eye) == 128, "eye offset");
static_assert(__builtin_offsetof(OcclMeshUniforms, view_back) == 144,
              "view_back offset");
static_assert(__builtin_offsetof(OcclMeshUniforms, ortho) == 156,
              "ortho offset");
static_assert(__builtin_offsetof(OcclMeshUniforms, radius) == 160,
              "radius offset");
static_assert(__builtin_offsetof(OcclMeshUniforms, mesh_id) == 164,
              "mesh_id offset");
static_assert(__builtin_offsetof(OcclMeshUniforms, line_info) == 176,
              "line_info offset");
