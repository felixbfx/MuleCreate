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
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <simd/simd.h>

#include <metal_stdlib>

using namespace metal;

template <typename T, size_t ElementCount>
struct ShaderArray {
  T elements[ElementCount ? ElementCount : 1];

  thread T& operator[](size_t element_index) thread {
    return elements[element_index];
  }
  constexpr const thread T& operator[](size_t element_index) const thread {
    return elements[element_index];
  }

  device T& operator[](size_t element_index) device {
    return elements[element_index];
  }
  constexpr const device T& operator[](size_t element_index) const device {
    return elements[element_index];
  }

  constexpr const constant T& operator[](size_t element_index) const constant {
    return elements[element_index];
  }

  threadgroup T& operator[](size_t element_index) threadgroup {
    return elements[element_index];
  }
  constexpr const threadgroup T& operator[](size_t element_index) const
      threadgroup {
    return elements[element_index];
  }
};

// Implementation of signed integer mod accurate to SPIR-V specification
template <typename DividendType, typename DivisorType>
inline DividendType signed_modulo(DividendType dividend, DivisorType divisor) {
  DividendType remainder = dividend - divisor * (dividend / divisor);
  return select(DividendType(remainder + divisor), remainder,
                remainder == 0 || (dividend >= 0) == (divisor >= 0));
}

struct OcclSurfaceWideUniforms {
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

constant ShaderArray<int, 6> line_endpoint_indices =
    ShaderArray<int, 6>({0, 0, 1, 0, 1, 1});
constant ShaderArray<float, 6> line_side_signs =
    ShaderArray<float, 6>({-1.0, 1.0, 1.0, -1.0, 1.0, -1.0});

struct occl_surface_wide_vert_out {
  float3 v_world [[user(locn0)]];
  float3 v_normal [[user(locn1)]];
  float3 v_calm [[user(locn2)]];
  int v_slot [[user(locn3)]];
  float4 position [[position]];
};

struct occl_surface_wide_vert_in {
  float3 line_start_position [[attribute(0)]];
  float3 line_start_normal [[attribute(1)]];
  float3 line_start_smooth_normal [[attribute(2)]];
  float line_start_surface_index [[attribute(3)]];
  float3 line_end_position [[attribute(4)]];
  float3 line_end_normal [[attribute(5)]];
  float3 line_end_smooth_normal [[attribute(6)]];
  float line_end_surface_index [[attribute(7)]];
};

static inline __attribute__((always_inline)) void interpolate_line_vertex(
    thread const float& endpoint_weight, thread float& vertex_surface_index,
    thread float3& vertex_position, thread float3& vertex_normal,
    thread float3& vertex_smooth_normal, thread float3& line_start_position,
    thread float3& line_end_position, thread float3& line_start_normal,
    thread float3& line_end_normal, thread float3& line_start_smooth_normal,
    thread float3& line_end_smooth_normal,
    thread float& line_start_surface_index,
    thread float& line_end_surface_index) {
  vertex_position =
      mix(line_start_position, line_end_position, float3(endpoint_weight));
  vertex_normal =
      mix(line_start_normal, line_end_normal, float3(endpoint_weight));
  vertex_smooth_normal = mix(line_start_smooth_normal, line_end_smooth_normal,
                             float3(endpoint_weight));
  vertex_surface_index =
      mix(line_start_surface_index, line_end_surface_index, endpoint_weight);
}

static inline __attribute__((always_inline)) float4
surf(thread const int& surface_index, thread const int& texel_column,
     texture2d<float> surface_table, sampler surface_table_sampler,
     constant OcclSurfaceWideUniforms& uniforms) {
  return surface_table.read(
      uint2(int2(texel_column + (10 * (surface_index / uniforms.surf_rows)),
                 signed_modulo(surface_index, uniforms.surf_rows))),
      0);
}

static inline __attribute__((always_inline)) void shade_vertex(
    texture2d<float> surface_table, sampler surface_table_sampler,
    constant OcclSurfaceWideUniforms& uniforms,
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

vertex occl_surface_wide_vert_out
occl_surface_wide_vert(occl_surface_wide_vert_in stage_input [[stage_in]],
                       constant OcclSurfaceWideUniforms& uniforms [[buffer(0)]],
                       texture2d<float> surface_table [[texture(0)]],
                       sampler surface_table_sampler [[sampler(0)]],
                       uint vertex_index [[vertex_id]]) {
  occl_surface_wide_vert_out stage_output = {};
  float start_sample_weight = 0.0;
  float vertex_surface_index;
  float3 vertex_position;
  float3 vertex_normal;
  float3 vertex_smooth_normal;
  interpolate_line_vertex(
      start_sample_weight, vertex_surface_index, vertex_position, vertex_normal,
      vertex_smooth_normal, stage_input.line_start_position,
      stage_input.line_end_position, stage_input.line_start_normal,
      stage_input.line_end_normal, stage_input.line_start_smooth_normal,
      stage_input.line_end_smooth_normal, stage_input.line_start_surface_index,
      stage_input.line_end_surface_index);
  shade_vertex(surface_table, surface_table_sampler, uniforms,
               vertex_surface_index, stage_output.v_slot, vertex_position,
               stage_output.v_normal, vertex_normal, stage_output.v_calm,
               vertex_smooth_normal, stage_output.v_world,
               stage_output.position);
  float4 start_clip_position = stage_output.position;
  float end_sample_weight = 1.0;
  interpolate_line_vertex(
      end_sample_weight, vertex_surface_index, vertex_position, vertex_normal,
      vertex_smooth_normal, stage_input.line_start_position,
      stage_input.line_end_position, stage_input.line_start_normal,
      stage_input.line_end_normal, stage_input.line_start_smooth_normal,
      stage_input.line_end_smooth_normal, stage_input.line_start_surface_index,
      stage_input.line_end_surface_index);
  shade_vertex(surface_table, surface_table_sampler, uniforms,
               vertex_surface_index, stage_output.v_slot, vertex_position,
               stage_output.v_normal, vertex_normal, stage_output.v_calm,
               vertex_smooth_normal, stage_output.v_world,
               stage_output.position);
  float4 end_clip_position = stage_output.position;
  bool start_behind_near_plane = start_clip_position.w < 1e-5;
  bool both_endpoints_clipped;

  if (start_behind_near_plane) {
    both_endpoints_clipped = end_clip_position.w < 1e-5;
  } else {
    both_endpoints_clipped = start_behind_near_plane;
  }

  if (both_endpoints_clipped) {
    stage_output.position = float4(2.0, 2.0, 2.0, 1.0);
    return stage_output;
  }

  float start_clip_weight = 0.0;
  float end_clip_weight = 1.0;

  if (start_clip_position.w < 1e-5) {
    start_clip_weight = (1e-5 - start_clip_position.w) /
                        (end_clip_position.w - start_clip_position.w);
  }

  if (end_clip_position.w < 1e-5) {
    end_clip_weight = (1e-5 - start_clip_position.w) /
                      (end_clip_position.w - start_clip_position.w);
  }

  if (start_clip_weight > 0.0) {
    float clipped_start_sample_weight = start_clip_weight;
    interpolate_line_vertex(
        clipped_start_sample_weight, vertex_surface_index, vertex_position,
        vertex_normal, vertex_smooth_normal, stage_input.line_start_position,
        stage_input.line_end_position, stage_input.line_start_normal,
        stage_input.line_end_normal, stage_input.line_start_smooth_normal,
        stage_input.line_end_smooth_normal,
        stage_input.line_start_surface_index,
        stage_input.line_end_surface_index);
    shade_vertex(surface_table, surface_table_sampler, uniforms,
                 vertex_surface_index, stage_output.v_slot, vertex_position,
                 stage_output.v_normal, vertex_normal, stage_output.v_calm,
                 vertex_smooth_normal, stage_output.v_world,
                 stage_output.position);
    start_clip_position = stage_output.position;
  }

  if (end_clip_weight < 1.0) {
    float clipped_end_sample_weight = end_clip_weight;
    interpolate_line_vertex(
        clipped_end_sample_weight, vertex_surface_index, vertex_position,
        vertex_normal, vertex_smooth_normal, stage_input.line_start_position,
        stage_input.line_end_position, stage_input.line_start_normal,
        stage_input.line_end_normal, stage_input.line_start_smooth_normal,
        stage_input.line_end_smooth_normal,
        stage_input.line_start_surface_index,
        stage_input.line_end_surface_index);
    shade_vertex(surface_table, surface_table_sampler, uniforms,
                 vertex_surface_index, stage_output.v_slot, vertex_position,
                 stage_output.v_normal, vertex_normal, stage_output.v_calm,
                 vertex_smooth_normal, stage_output.v_world,
                 stage_output.position);
    end_clip_position = stage_output.position;
  }

  int triangle_corner = signed_modulo(int(vertex_index), 6);
  float corner_sample_weight = (line_endpoint_indices[triangle_corner] == 0)
                                   ? start_clip_weight
                                   : end_clip_weight;
  interpolate_line_vertex(
      corner_sample_weight, vertex_surface_index, vertex_position,
      vertex_normal, vertex_smooth_normal, stage_input.line_start_position,
      stage_input.line_end_position, stage_input.line_start_normal,
      stage_input.line_end_normal, stage_input.line_start_smooth_normal,
      stage_input.line_end_smooth_normal, stage_input.line_start_surface_index,
      stage_input.line_end_surface_index);
  shade_vertex(surface_table, surface_table_sampler, uniforms,
               vertex_surface_index, stage_output.v_slot, vertex_position,
               stage_output.v_normal, vertex_normal, stage_output.v_calm,
               vertex_smooth_normal, stage_output.v_world,
               stage_output.position);
  float4 expanded_clip_position = stage_output.position;
  float2 half_viewport_pixels =
      fast::max(uniforms.line_info.xy * 0.5, float2(1.0));
  float2 screen_direction =
      ((end_clip_position.xy / float2(end_clip_position.w)) *
       half_viewport_pixels) -
      ((start_clip_position.xy / float2(start_clip_position.w)) *
       half_viewport_pixels);
  float2 normalized_screen_direction;

  if (length(screen_direction) > 1e-6) {
    normalized_screen_direction = fast::normalize(screen_direction);
  } else {
    normalized_screen_direction = float2(1.0, 0.0);
  }

  screen_direction = normalized_screen_direction;
  float expansion_clip_w = expanded_clip_position.w;
  float4 unexpanded_clip_position = expanded_clip_position;
  float2 expanded_clip_xy =
      unexpanded_clip_position.xy +
      (((float2(-screen_direction.y, screen_direction.x) *
         ((line_side_signs[triangle_corner] * uniforms.line_info.z) * 0.5)) /
        half_viewport_pixels) *
       expansion_clip_w);
  expanded_clip_position.x = expanded_clip_xy.x;
  expanded_clip_position.y = expanded_clip_xy.y;
  stage_output.position = expanded_clip_position;
  stage_output.position.y = -stage_output.position.y;
  stage_output.position.z =
      0.5 * (stage_output.position.z + stage_output.position.w);
  return stage_output;
}

static_assert(sizeof(OcclSurfaceWideUniforms) == 128, "Uniform size");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, vp) == 0,
              "vp offset");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, bias0) == 64,
              "bias offset");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, occluders_only) == 68,
              "occluders_only offset");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, surf_rows) == 72,
              "surf_rows offset");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, eye) == 80,
              "eye offset");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, view_back) == 96,
              "view_back offset");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, ortho) == 108,
              "ortho offset");
static_assert(__builtin_offsetof(OcclSurfaceWideUniforms, line_info) == 112,
              "line_info offset");
