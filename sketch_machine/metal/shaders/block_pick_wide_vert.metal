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

struct BlockPickWideUniforms {
  float4x4 mvp;
  int vis_col;
  char padding0[12];
  float4 line_info;
};

constant ShaderArray<int, 6> line_endpoint_indices =
    ShaderArray<int, 6>({0, 0, 1, 0, 1, 1});
constant ShaderArray<float, 6> line_side_signs =
    ShaderArray<float, 6>({-1.0, 1.0, 1.0, -1.0, 1.0, -1.0});

struct block_pick_wide_vert_out {
  float3 v_world [[user(locn0)]];
  float3 v_color [[user(locn1)]];
  int v_slot [[user(locn2)]];
  float4 position [[position]];
};

struct block_pick_wide_vert_in {
  float3 line_start_position [[attribute(0)]];
  float3 line_start_color [[attribute(1)]];
  float line_start_surface_index [[attribute(2)]];
  float3 line_end_position [[attribute(3)]];
  float3 line_end_color [[attribute(4)]];
  float line_end_surface_index [[attribute(5)]];
};

static inline __attribute__((always_inline)) void interpolate_line_vertex(
    thread const float& endpoint_weight, thread float& vertex_surface_index,
    thread float3& vertex_position, thread float3& vertex_color,
    thread float3& line_start_position, thread float3& line_end_position,
    thread float3& line_start_color, thread float3& line_end_color,
    thread float& line_start_surface_index,
    thread float& line_end_surface_index) {
  vertex_position =
      mix(line_start_position, line_end_position, float3(endpoint_weight));
  vertex_color = mix(line_start_color, line_end_color, float3(endpoint_weight));
  vertex_surface_index =
      mix(line_start_surface_index, line_end_surface_index, endpoint_weight);
}

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
    constant BlockPickWideUniforms& uniforms, thread float4& clip_position) {
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

vertex block_pick_wide_vert_out
block_pick_wide_vert(block_pick_wide_vert_in stage_input [[stage_in]],
                     constant BlockPickWideUniforms& uniforms [[buffer(0)]],
                     texture2d<float> mesh_table [[texture(0)]],
                     sampler mesh_table_sampler [[sampler(0)]],
                     uint vertex_index [[vertex_id]]) {
  block_pick_wide_vert_out stage_output = {};
  float start_sample_weight = 0.0;
  float vertex_surface_index;
  float3 vertex_position;
  float3 vertex_color;
  interpolate_line_vertex(
      start_sample_weight, vertex_surface_index, vertex_position, vertex_color,
      stage_input.line_start_position, stage_input.line_end_position,
      stage_input.line_start_color, stage_input.line_end_color,
      stage_input.line_start_surface_index, stage_input.line_end_surface_index);
  shade_vertex(mesh_table, mesh_table_sampler, stage_output.v_slot,
               vertex_surface_index, stage_output.v_world, vertex_position,
               stage_output.v_color, vertex_color, uniforms,
               stage_output.position);
  float4 start_clip_position = stage_output.position;
  float end_sample_weight = 1.0;
  interpolate_line_vertex(
      end_sample_weight, vertex_surface_index, vertex_position, vertex_color,
      stage_input.line_start_position, stage_input.line_end_position,
      stage_input.line_start_color, stage_input.line_end_color,
      stage_input.line_start_surface_index, stage_input.line_end_surface_index);
  shade_vertex(mesh_table, mesh_table_sampler, stage_output.v_slot,
               vertex_surface_index, stage_output.v_world, vertex_position,
               stage_output.v_color, vertex_color, uniforms,
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
        vertex_color, stage_input.line_start_position,
        stage_input.line_end_position, stage_input.line_start_color,
        stage_input.line_end_color, stage_input.line_start_surface_index,
        stage_input.line_end_surface_index);
    shade_vertex(mesh_table, mesh_table_sampler, stage_output.v_slot,
                 vertex_surface_index, stage_output.v_world, vertex_position,
                 stage_output.v_color, vertex_color, uniforms,
                 stage_output.position);
    start_clip_position = stage_output.position;
  }

  if (end_clip_weight < 1.0) {
    float clipped_end_sample_weight = end_clip_weight;
    interpolate_line_vertex(
        clipped_end_sample_weight, vertex_surface_index, vertex_position,
        vertex_color, stage_input.line_start_position,
        stage_input.line_end_position, stage_input.line_start_color,
        stage_input.line_end_color, stage_input.line_start_surface_index,
        stage_input.line_end_surface_index);
    shade_vertex(mesh_table, mesh_table_sampler, stage_output.v_slot,
                 vertex_surface_index, stage_output.v_world, vertex_position,
                 stage_output.v_color, vertex_color, uniforms,
                 stage_output.position);
    end_clip_position = stage_output.position;
  }

  int triangle_corner = signed_modulo(int(vertex_index), 6);
  float corner_sample_weight = (line_endpoint_indices[triangle_corner] == 0)
                                   ? start_clip_weight
                                   : end_clip_weight;
  interpolate_line_vertex(
      corner_sample_weight, vertex_surface_index, vertex_position, vertex_color,
      stage_input.line_start_position, stage_input.line_end_position,
      stage_input.line_start_color, stage_input.line_end_color,
      stage_input.line_start_surface_index, stage_input.line_end_surface_index);
  shade_vertex(mesh_table, mesh_table_sampler, stage_output.v_slot,
               vertex_surface_index, stage_output.v_world, vertex_position,
               stage_output.v_color, vertex_color, uniforms,
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

static_assert(sizeof(BlockPickWideUniforms) == 96, "Uniform size");
static_assert(__builtin_offsetof(BlockPickWideUniforms, mvp) == 0,
              "mvp offset");
static_assert(__builtin_offsetof(BlockPickWideUniforms, vis_col) == 64,
              "vis_col offset");
static_assert(__builtin_offsetof(BlockPickWideUniforms, line_info) == 80,
              "line_info offset");
