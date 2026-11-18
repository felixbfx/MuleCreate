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

struct occl_mesh_frag_out {
  float4 f_cover [[color(0)]];
};

struct occl_mesh_frag_in {
  float3 v_world [[user(locn0)]];
};

fragment occl_mesh_frag_out occl_mesh_frag(occl_mesh_frag_in stage_input
                                           [[stage_in]],
                                           constant OcclMeshUniforms& uniforms
                                           [[buffer(0)]]) {
  occl_mesh_frag_out stage_output = {};
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
      fast::max(uniforms.radius, 3.0 * world_pixel_size), uniforms.mesh_id);
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
