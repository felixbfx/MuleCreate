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

struct FlatWideUniforms {
  float4x4 vp;
  float bias0;
  char padding0[12];
  float4 color;
  float grain;
  char padding1[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float milk;
  char padding2[12];
  float4 owner;
  float under;
  float soft;
  float occl_on;
  float xray_behind;
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding3[12];
  float4 line_info;
};

constant ShaderArray<int, 6> line_endpoint_indices =
    ShaderArray<int, 6>({0, 0, 1, 0, 1, 1});
constant ShaderArray<float, 6> line_side_signs =
    ShaderArray<float, 6>({-1.0, 1.0, 1.0, -1.0, 1.0, -1.0});

struct flat_wide_vert_out {
  float3 v_world [[user(locn0)]];
  float4 position [[position]];
};

struct flat_wide_vert_in {
  float3 line_start_position [[attribute(0)]];
  float3 line_end_position [[attribute(1)]];
};

static inline __attribute__((always_inline)) void interpolate_line_vertex(
    thread const float& endpoint_weight, thread float3& vertex_position,
    thread float3& line_start_position, thread float3& line_end_position) {
  vertex_position =
      mix(line_start_position, line_end_position, float3(endpoint_weight));
}

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& world_position_output, thread float3& vertex_position,
    constant FlatWideUniforms& uniforms, thread float4& clip_position) {
  world_position_output = vertex_position;
  float4 unbiased_clip_position = uniforms.vp * float4(vertex_position, 1.0);
  clip_position = float4(
      unbiased_clip_position.xy,
      unbiased_clip_position.z + (uniforms.bias0 * unbiased_clip_position.w),
      unbiased_clip_position.w);
}

vertex flat_wide_vert_out flat_wide_vert(flat_wide_vert_in stage_input
                                         [[stage_in]],
                                         constant FlatWideUniforms& uniforms
                                         [[buffer(0)]],
                                         uint vertex_index [[vertex_id]]) {
  flat_wide_vert_out stage_output = {};
  float start_sample_weight = 0.0;
  float3 vertex_position;
  interpolate_line_vertex(start_sample_weight, vertex_position,
                          stage_input.line_start_position,
                          stage_input.line_end_position);
  shade_vertex(stage_output.v_world, vertex_position, uniforms,
               stage_output.position);
  float4 start_clip_position = stage_output.position;
  float end_sample_weight = 1.0;
  interpolate_line_vertex(end_sample_weight, vertex_position,
                          stage_input.line_start_position,
                          stage_input.line_end_position);
  shade_vertex(stage_output.v_world, vertex_position, uniforms,
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
    interpolate_line_vertex(clipped_start_sample_weight, vertex_position,
                            stage_input.line_start_position,
                            stage_input.line_end_position);
    shade_vertex(stage_output.v_world, vertex_position, uniforms,
                 stage_output.position);
    start_clip_position = stage_output.position;
  }

  if (end_clip_weight < 1.0) {
    float clipped_end_sample_weight = end_clip_weight;
    interpolate_line_vertex(clipped_end_sample_weight, vertex_position,
                            stage_input.line_start_position,
                            stage_input.line_end_position);
    shade_vertex(stage_output.v_world, vertex_position, uniforms,
                 stage_output.position);
    end_clip_position = stage_output.position;
  }

  int triangle_corner = signed_modulo(int(vertex_index), 6);
  float corner_sample_weight = (line_endpoint_indices[triangle_corner] == 0)
                                   ? start_clip_weight
                                   : end_clip_weight;
  interpolate_line_vertex(corner_sample_weight, vertex_position,
                          stage_input.line_start_position,
                          stage_input.line_end_position);
  shade_vertex(stage_output.v_world, vertex_position, uniforms,
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

static_assert(sizeof(FlatWideUniforms) == 240, "Uniform size");
static_assert(__builtin_offsetof(FlatWideUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(FlatWideUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(FlatWideUniforms, color) == 80,
              "color offset");
static_assert(__builtin_offsetof(FlatWideUniforms, grain) == 96,
              "grain offset");
static_assert(__builtin_offsetof(FlatWideUniforms, eye) == 112, "eye offset");
static_assert(__builtin_offsetof(FlatWideUniforms, view_back) == 128,
              "view_back offset");
static_assert(__builtin_offsetof(FlatWideUniforms, ortho) == 140,
              "ortho offset");
static_assert(__builtin_offsetof(FlatWideUniforms, milk) == 144, "milk offset");
static_assert(__builtin_offsetof(FlatWideUniforms, owner) == 160,
              "owner offset");
static_assert(__builtin_offsetof(FlatWideUniforms, under) == 176,
              "under offset");
static_assert(__builtin_offsetof(FlatWideUniforms, soft) == 180, "soft offset");
static_assert(__builtin_offsetof(FlatWideUniforms, occl_on) == 184,
              "occl_on offset");
static_assert(__builtin_offsetof(FlatWideUniforms, xray_behind) == 188,
              "xray_behind offset");
static_assert(__builtin_offsetof(FlatWideUniforms, occl_tol) == 192,
              "occl_tol offset");
static_assert(__builtin_offsetof(FlatWideUniforms, occl_min) == 204,
              "occl_min offset");
static_assert(__builtin_offsetof(FlatWideUniforms, aid_pass) == 208,
              "aid_pass offset");
static_assert(__builtin_offsetof(FlatWideUniforms, line_info) == 224,
              "line_info offset");
