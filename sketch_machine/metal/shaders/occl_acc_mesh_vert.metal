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

struct OcclAccMeshUniforms {
  float4x4 mvp;
  float4x4 model;
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float occl_on;
  float xray_behind;
  char padding0[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding1[12];
  float4 owner;
  float acc_value;
  char padding2[12];
  float4 line_info;
};

struct occl_acc_mesh_vert_out {
  float3 v_world [[user(locn0)]];
  float4 position [[position]];
};

struct occl_acc_mesh_vert_in {
  float3 in_pos [[attribute(0)]];
};

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& world_position_output,
    constant OcclAccMeshUniforms& uniforms, thread float3& vertex_position,
    thread float4& clip_position) {
  world_position_output = (uniforms.model * float4(vertex_position, 1.0)).xyz;
  clip_position = uniforms.mvp * float4(vertex_position, 1.0);
}

vertex occl_acc_mesh_vert_out
occl_acc_mesh_vert(occl_acc_mesh_vert_in stage_input [[stage_in]],
                   constant OcclAccMeshUniforms& uniforms [[buffer(0)]]) {
  occl_acc_mesh_vert_out stage_output = {};
  shade_vertex(stage_output.v_world, uniforms, stage_input.in_pos,
               stage_output.position);
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(OcclAccMeshUniforms) == 256, "Uniform size");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, model) == 64,
              "model offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, eye) == 128,
              "eye offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, view_back) == 144,
              "view_back offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, ortho) == 156,
              "ortho offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, occl_on) == 160,
              "occl_on offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, xray_behind) == 164,
              "xray_behind offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, occl_tol) == 176,
              "occl_tol offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, occl_min) == 188,
              "occl_min offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, aid_pass) == 192,
              "aid_pass offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, owner) == 208,
              "owner offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, acc_value) == 224,
              "acc_value offset");
static_assert(__builtin_offsetof(OcclAccMeshUniforms, line_info) == 240,
              "line_info offset");
