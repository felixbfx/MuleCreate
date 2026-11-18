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

struct RibbonWideUniforms {
  float4x4 vp;
  float2 half_vp;
  float focal;
  float min_px;
  float bias0;
  char padding0[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  packed_float3 sun_dir;
  float grain;
  float fade;
  float light_on;
  char padding1[8];
  float3 sun_col;
  packed_float3 sky_col;
  float shadow_dark;
  float bands;
  float shadow_on;
  char padding2[8];
  float4x4 shadow_vp;
  float shadow_bias;
  float shadow_lift;
  float dome;
  float occl_on;
  float xray_behind;
  char padding3[12];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  int plane_n;
  char padding4[8];
  float3 plane_o[16];
  float3 plane_u[16];
  float3 plane_v[16];
  float4 plane_c[16];
  float4 plane_k[16];
  float4 plane_s[16];
  float edge_gain;
  char padding5[12];
  float4 line_info;
};

constant ShaderArray<int, 6> line_endpoint_indices =
    ShaderArray<int, 6>({0, 0, 1, 0, 1, 1});
constant ShaderArray<float, 6> line_side_signs =
    ShaderArray<float, 6>({-1.0, 1.0, 1.0, -1.0, 1.0, -1.0});

struct ribbon_wide_vert_out {
  float4 v_color [[user(locn0)]];
  float v_w [[user(locn1)]];
  float v_side [[user(locn2)]];
  float v_arc [[user(locn3)]];
  float v_px [[user(locn4)]];
  float v_brush [[user(locn5)]];
  float v_light [[user(locn6)]];
  float3 v_wpos [[user(locn7)]];
  float3 v_tan [[user(locn8)]];
  float v_rad [[user(locn9)]];
  float4 v_owner [[user(locn10)]];
  float4 position [[position]];
};

struct ribbon_wide_vert_in {
  float3 line_start_position [[attribute(0)]];
  float3 line_start_previous_position [[attribute(1)]];
  float3 line_start_next_position [[attribute(2)]];
  float line_start_stroke_width [[attribute(3)]];
  float line_start_stroke_side [[attribute(4)]];
  float line_start_arc_length [[attribute(5)]];
  float4 line_start_color [[attribute(6)]];
  float4 line_start_owner_ids [[attribute(7)]];
  float3 line_start_normal [[attribute(8)]];
  float line_start_layer [[attribute(9)]];
  float3 line_end_position [[attribute(10)]];
  float3 line_end_previous_position [[attribute(11)]];
  float3 line_end_next_position [[attribute(12)]];
  float line_end_stroke_width [[attribute(13)]];
  float line_end_stroke_side [[attribute(14)]];
  float line_end_arc_length [[attribute(15)]];
  float4 line_end_color [[attribute(16)]];
  float4 line_end_owner_ids [[attribute(17)]];
  float3 line_end_normal [[attribute(18)]];
  float line_end_layer [[attribute(19)]];
};

static inline __attribute__((always_inline)) void interpolate_line_vertex(
    thread const float& endpoint_weight, thread float3& vertex_position,
    thread float4& vertex_owner_ids, thread float& stroke_side,
    thread float3& vertex_normal, thread float3& next_vertex_position,
    thread float3& previous_vertex_position, thread float& stroke_width,
    thread float& stroke_layer, thread float4& vertex_color,
    thread float& stroke_arc_length, thread float3& line_start_position,
    thread float3& line_end_position,
    thread float3& line_start_previous_position,
    thread float3& line_end_previous_position,
    thread float3& line_start_next_position,
    thread float3& line_end_next_position,
    thread float& line_start_stroke_width, thread float& line_end_stroke_width,
    thread float& line_start_stroke_side, thread float& line_end_stroke_side,
    thread float& line_start_arc_length, thread float& line_end_arc_length,
    thread float4& line_start_color, thread float4& line_end_color,
    thread float4& line_start_owner_ids, thread float4& line_end_owner_ids,
    thread float3& line_start_normal, thread float3& line_end_normal,
    thread float& line_start_layer, thread float& line_end_layer) {
  vertex_position =
      mix(line_start_position, line_end_position, float3(endpoint_weight));
  previous_vertex_position =
      mix(line_start_previous_position, line_end_previous_position,
          float3(endpoint_weight));
  next_vertex_position = mix(line_start_next_position, line_end_next_position,
                             float3(endpoint_weight));
  stroke_width =
      mix(line_start_stroke_width, line_end_stroke_width, endpoint_weight);
  stroke_side =
      mix(line_start_stroke_side, line_end_stroke_side, endpoint_weight);
  stroke_arc_length =
      mix(line_start_arc_length, line_end_arc_length, endpoint_weight);
  vertex_color = mix(line_start_color, line_end_color, float4(endpoint_weight));
  vertex_owner_ids =
      mix(line_start_owner_ids, line_end_owner_ids, float4(endpoint_weight));
  vertex_normal =
      mix(line_start_normal, line_end_normal, float3(endpoint_weight));
  stroke_layer = mix(line_start_layer, line_end_layer, endpoint_weight);
}

static inline __attribute__((always_inline)) void shade_vertex(
    thread float3& stroke_world_position_output, thread float3& vertex_position,
    thread float4& owner_ids_output, thread float4& vertex_owner_ids,
    thread float& stroke_side, thread float3& vertex_normal,
    thread float3& next_vertex_position,
    thread float3& previous_vertex_position,
    constant RibbonWideUniforms& uniforms, thread float& stroke_width,
    thread float4& clip_position, thread float& stroke_layer,
    thread float4& color_output, thread float4& vertex_color,
    thread float& clip_w_output, thread float& stroke_side_output,
    thread float& arc_length_output, thread float& stroke_arc_length,
    thread float& pixel_width_output, thread float& brush_style_output,
    thread float& edge_light_output, thread float3& tangent_output,
    thread float& radius_output) {
  stroke_world_position_output = vertex_position;
  owner_ids_output = vertex_owner_ids;
  float brush_style_flag = step(1.5, abs(stroke_side));

  if (length(vertex_normal) > 0.5) {
    float flat_stroke_side = stroke_side / (1.0 + brush_style_flag);
    float3 surface_tangent = next_vertex_position - previous_vertex_position;
    float3 normalized_surface_tangent;

    if (length(surface_tangent) > 1e-9) {
      normalized_surface_tangent = fast::normalize(surface_tangent);
    } else {
      normalized_surface_tangent = float3(1.0, 0.0, 0.0);
    }

    surface_tangent = normalized_surface_tangent;
    float3 surface_bitangent =
        cross(fast::normalize(vertex_normal), surface_tangent);
    float3 normalized_surface_bitangent;

    if (length(surface_bitangent) > 1e-6) {
      normalized_surface_bitangent = fast::normalize(surface_bitangent);
    } else {
      normalized_surface_bitangent = float3(0.0, 1.0, 0.0);
    }

    surface_bitangent = normalized_surface_bitangent;
    float4 flat_clip_position =
        uniforms.vp *
        float4(
            vertex_position +
                (((surface_bitangent * flat_stroke_side) * stroke_width) * 0.5),
            1.0);
    float flat_clip_w = fast::max(flat_clip_position.w, 1e-5);
    clip_position = float4(flat_clip_position.xy,
                           (0.9 - (1.8 * stroke_layer)) * flat_clip_position.w,
                           flat_clip_position.w);
    color_output = vertex_color;
    clip_w_output = flat_clip_position.w;
    stroke_side_output = flat_stroke_side;
    arc_length_output = stroke_arc_length;
    pixel_width_output = fast::max(
        ((stroke_width * uniforms.focal) * uniforms.half_vp.y) / flat_clip_w,
        uniforms.min_px);
    brush_style_output = brush_style_flag;
    edge_light_output = 0.0;
    tangent_output = float3(0.0);
    radius_output = 0.0;
    return;
  }

  float screen_stroke_side = stroke_side / (1.0 + brush_style_flag);
  float4 center_clip_position = uniforms.vp * float4(vertex_position, 1.0);
  float4 previous_clip_position =
      uniforms.vp * float4(previous_vertex_position, 1.0);
  float4 next_clip_position = uniforms.vp * float4(next_vertex_position, 1.0);
  float safe_clip_w = fast::max(center_clip_position.w, 1e-5);
  float2 center_screen_position =
      (center_clip_position.xy / float2(safe_clip_w)) * uniforms.half_vp;
  float2 screen_direction =
      ((next_clip_position.xy / float2(fast::max(next_clip_position.w, 1e-5))) *
       uniforms.half_vp) -
      ((previous_clip_position.xy /
        float2(fast::max(previous_clip_position.w, 1e-5))) *
       uniforms.half_vp);
  if (length(screen_direction) < 1e-6) {
    screen_direction = float2(1.0, 0.0);
  }

  float2 screen_normal =
      fast::normalize(float2(-screen_direction.y, screen_direction.x));
  float pixel_width = fast::max(
      ((stroke_width * uniforms.focal) * uniforms.half_vp.y) / safe_clip_w,
      uniforms.min_px);
  float2 extruded_pixel =
      center_screen_position +
      (((screen_normal * pixel_width) * 0.5) * screen_stroke_side);
  clip_position = float4(
      (extruded_pixel / uniforms.half_vp) * safe_clip_w,
      center_clip_position.z - (uniforms.bias0 * safe_clip_w), safe_clip_w);
  color_output = vertex_color;
  clip_w_output = center_clip_position.w;
  stroke_side_output = screen_stroke_side;
  arc_length_output = stroke_arc_length;
  pixel_width_output = pixel_width;
  brush_style_output = brush_style_flag;
  float3 stroke_tangent = next_vertex_position - previous_vertex_position;
  float3 normalized_stroke_tangent;

  if (length(stroke_tangent) > 1e-9) {
    normalized_stroke_tangent = fast::normalize(stroke_tangent);
  } else {
    normalized_stroke_tangent = float3(1.0, 0.0, 0.0);
  }

  stroke_tangent = normalized_stroke_tangent;
  tangent_output = stroke_tangent;
  radius_output = stroke_width * 0.5;
  float3 selected_view_direction;

  if (uniforms.ortho > 0.5) {
    selected_view_direction = float3(uniforms.view_back);
  } else {
    selected_view_direction = fast::normalize(uniforms.eye - vertex_position);
  }

  float3 view_direction = selected_view_direction;
  float3 view_bitangent = cross(view_direction, stroke_tangent);
  float view_bitangent_length = length(view_bitangent);
  edge_light_output = 0.0;

  if (view_bitangent_length > 1e-5) {
    view_bitangent /= float3(view_bitangent_length);
    float4 bitangent_clip_position =
        uniforms.vp *
        float4(vertex_position +
                   (view_bitangent * fast::max(stroke_width, 0.0001)),
               1.0);
    float2 bitangent_screen_offset =
        ((bitangent_clip_position.xy /
          float2(fast::max(bitangent_clip_position.w, 1e-5))) *
         uniforms.half_vp) -
        center_screen_position;
    if (dot(bitangent_screen_offset, screen_normal) < 0.0) {
      view_bitangent = -view_bitangent;
    }

    edge_light_output = dot(view_bitangent, float3(uniforms.sun_dir));
  }
}

vertex ribbon_wide_vert_out
ribbon_wide_vert(ribbon_wide_vert_in stage_input [[stage_in]],
                 constant RibbonWideUniforms& uniforms [[buffer(0)]],
                 uint vertex_index [[vertex_id]]) {
  ribbon_wide_vert_out stage_output = {};
  float start_sample_weight = 0.0;
  float3 vertex_position;
  float4 vertex_owner_ids;
  float stroke_side;
  float3 vertex_normal;
  float3 next_vertex_position;
  float3 previous_vertex_position;
  float stroke_width;
  float stroke_layer;
  float4 vertex_color;
  float stroke_arc_length;
  interpolate_line_vertex(
      start_sample_weight, vertex_position, vertex_owner_ids, stroke_side,
      vertex_normal, next_vertex_position, previous_vertex_position,
      stroke_width, stroke_layer, vertex_color, stroke_arc_length,
      stage_input.line_start_position, stage_input.line_end_position,
      stage_input.line_start_previous_position,
      stage_input.line_end_previous_position,
      stage_input.line_start_next_position, stage_input.line_end_next_position,
      stage_input.line_start_stroke_width, stage_input.line_end_stroke_width,
      stage_input.line_start_stroke_side, stage_input.line_end_stroke_side,
      stage_input.line_start_arc_length, stage_input.line_end_arc_length,
      stage_input.line_start_color, stage_input.line_end_color,
      stage_input.line_start_owner_ids, stage_input.line_end_owner_ids,
      stage_input.line_start_normal, stage_input.line_end_normal,
      stage_input.line_start_layer, stage_input.line_end_layer);
  shade_vertex(stage_output.v_wpos, vertex_position, stage_output.v_owner,
               vertex_owner_ids, stroke_side, vertex_normal,
               next_vertex_position, previous_vertex_position, uniforms,
               stroke_width, stage_output.position, stroke_layer,
               stage_output.v_color, vertex_color, stage_output.v_w,
               stage_output.v_side, stage_output.v_arc, stroke_arc_length,
               stage_output.v_px, stage_output.v_brush, stage_output.v_light,
               stage_output.v_tan, stage_output.v_rad);
  float4 start_clip_position = stage_output.position;
  float end_sample_weight = 1.0;
  interpolate_line_vertex(
      end_sample_weight, vertex_position, vertex_owner_ids, stroke_side,
      vertex_normal, next_vertex_position, previous_vertex_position,
      stroke_width, stroke_layer, vertex_color, stroke_arc_length,
      stage_input.line_start_position, stage_input.line_end_position,
      stage_input.line_start_previous_position,
      stage_input.line_end_previous_position,
      stage_input.line_start_next_position, stage_input.line_end_next_position,
      stage_input.line_start_stroke_width, stage_input.line_end_stroke_width,
      stage_input.line_start_stroke_side, stage_input.line_end_stroke_side,
      stage_input.line_start_arc_length, stage_input.line_end_arc_length,
      stage_input.line_start_color, stage_input.line_end_color,
      stage_input.line_start_owner_ids, stage_input.line_end_owner_ids,
      stage_input.line_start_normal, stage_input.line_end_normal,
      stage_input.line_start_layer, stage_input.line_end_layer);
  shade_vertex(stage_output.v_wpos, vertex_position, stage_output.v_owner,
               vertex_owner_ids, stroke_side, vertex_normal,
               next_vertex_position, previous_vertex_position, uniforms,
               stroke_width, stage_output.position, stroke_layer,
               stage_output.v_color, vertex_color, stage_output.v_w,
               stage_output.v_side, stage_output.v_arc, stroke_arc_length,
               stage_output.v_px, stage_output.v_brush, stage_output.v_light,
               stage_output.v_tan, stage_output.v_rad);
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
        clipped_start_sample_weight, vertex_position, vertex_owner_ids,
        stroke_side, vertex_normal, next_vertex_position,
        previous_vertex_position, stroke_width, stroke_layer, vertex_color,
        stroke_arc_length, stage_input.line_start_position,
        stage_input.line_end_position, stage_input.line_start_previous_position,
        stage_input.line_end_previous_position,
        stage_input.line_start_next_position,
        stage_input.line_end_next_position, stage_input.line_start_stroke_width,
        stage_input.line_end_stroke_width, stage_input.line_start_stroke_side,
        stage_input.line_end_stroke_side, stage_input.line_start_arc_length,
        stage_input.line_end_arc_length, stage_input.line_start_color,
        stage_input.line_end_color, stage_input.line_start_owner_ids,
        stage_input.line_end_owner_ids, stage_input.line_start_normal,
        stage_input.line_end_normal, stage_input.line_start_layer,
        stage_input.line_end_layer);
    shade_vertex(stage_output.v_wpos, vertex_position, stage_output.v_owner,
                 vertex_owner_ids, stroke_side, vertex_normal,
                 next_vertex_position, previous_vertex_position, uniforms,
                 stroke_width, stage_output.position, stroke_layer,
                 stage_output.v_color, vertex_color, stage_output.v_w,
                 stage_output.v_side, stage_output.v_arc, stroke_arc_length,
                 stage_output.v_px, stage_output.v_brush, stage_output.v_light,
                 stage_output.v_tan, stage_output.v_rad);
    start_clip_position = stage_output.position;
  }

  if (end_clip_weight < 1.0) {
    float clipped_end_sample_weight = end_clip_weight;
    interpolate_line_vertex(
        clipped_end_sample_weight, vertex_position, vertex_owner_ids,
        stroke_side, vertex_normal, next_vertex_position,
        previous_vertex_position, stroke_width, stroke_layer, vertex_color,
        stroke_arc_length, stage_input.line_start_position,
        stage_input.line_end_position, stage_input.line_start_previous_position,
        stage_input.line_end_previous_position,
        stage_input.line_start_next_position,
        stage_input.line_end_next_position, stage_input.line_start_stroke_width,
        stage_input.line_end_stroke_width, stage_input.line_start_stroke_side,
        stage_input.line_end_stroke_side, stage_input.line_start_arc_length,
        stage_input.line_end_arc_length, stage_input.line_start_color,
        stage_input.line_end_color, stage_input.line_start_owner_ids,
        stage_input.line_end_owner_ids, stage_input.line_start_normal,
        stage_input.line_end_normal, stage_input.line_start_layer,
        stage_input.line_end_layer);
    shade_vertex(stage_output.v_wpos, vertex_position, stage_output.v_owner,
                 vertex_owner_ids, stroke_side, vertex_normal,
                 next_vertex_position, previous_vertex_position, uniforms,
                 stroke_width, stage_output.position, stroke_layer,
                 stage_output.v_color, vertex_color, stage_output.v_w,
                 stage_output.v_side, stage_output.v_arc, stroke_arc_length,
                 stage_output.v_px, stage_output.v_brush, stage_output.v_light,
                 stage_output.v_tan, stage_output.v_rad);
    end_clip_position = stage_output.position;
  }

  int triangle_corner = signed_modulo(int(vertex_index), 6);
  float corner_sample_weight = (line_endpoint_indices[triangle_corner] == 0)
                                   ? start_clip_weight
                                   : end_clip_weight;
  interpolate_line_vertex(
      corner_sample_weight, vertex_position, vertex_owner_ids, stroke_side,
      vertex_normal, next_vertex_position, previous_vertex_position,
      stroke_width, stroke_layer, vertex_color, stroke_arc_length,
      stage_input.line_start_position, stage_input.line_end_position,
      stage_input.line_start_previous_position,
      stage_input.line_end_previous_position,
      stage_input.line_start_next_position, stage_input.line_end_next_position,
      stage_input.line_start_stroke_width, stage_input.line_end_stroke_width,
      stage_input.line_start_stroke_side, stage_input.line_end_stroke_side,
      stage_input.line_start_arc_length, stage_input.line_end_arc_length,
      stage_input.line_start_color, stage_input.line_end_color,
      stage_input.line_start_owner_ids, stage_input.line_end_owner_ids,
      stage_input.line_start_normal, stage_input.line_end_normal,
      stage_input.line_start_layer, stage_input.line_end_layer);
  shade_vertex(stage_output.v_wpos, vertex_position, stage_output.v_owner,
               vertex_owner_ids, stroke_side, vertex_normal,
               next_vertex_position, previous_vertex_position, uniforms,
               stroke_width, stage_output.position, stroke_layer,
               stage_output.v_color, vertex_color, stage_output.v_w,
               stage_output.v_side, stage_output.v_arc, stroke_arc_length,
               stage_output.v_px, stage_output.v_brush, stage_output.v_light,
               stage_output.v_tan, stage_output.v_rad);
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

static_assert(sizeof(RibbonWideUniforms) == 1904, "Uniform size");
static_assert(__builtin_offsetof(RibbonWideUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, half_vp) == 64,
              "half_vp offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, focal) == 72,
              "focal offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, min_px) == 76,
              "min_px offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, bias0) == 80,
              "bias offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, eye) == 96, "eye offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, view_back) == 112,
              "view_back offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, ortho) == 124,
              "ortho offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, sun_dir) == 128,
              "sun_dir offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, grain) == 140,
              "grain offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, fade) == 144,
              "fade offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, light_on) == 148,
              "light_on offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, sun_col) == 160,
              "sun_col offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, sky_col) == 176,
              "sky_col offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, shadow_dark) == 188,
              "shadow_dark offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, bands) == 192,
              "bands offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, shadow_on) == 196,
              "shadow_on offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, shadow_vp) == 208,
              "shadow_vp offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, shadow_bias) == 272,
              "shadow_bias offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, shadow_lift) == 276,
              "shadow_lift offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, dome) == 280,
              "dome offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, occl_on) == 284,
              "occl_on offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, xray_behind) == 288,
              "xray_behind offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, occl_tol) == 304,
              "occl_tol offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, occl_min) == 316,
              "occl_min offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, aid_pass) == 320,
              "aid_pass offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, plane_n) == 324,
              "plane_n offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, plane_o) == 336,
              "plane_o offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, plane_u) == 592,
              "plane_u offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, plane_v) == 848,
              "plane_v offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, plane_c) == 1104,
              "plane_c offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, plane_k) == 1360,
              "plane_k offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, plane_s) == 1616,
              "plane_s offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, edge_gain) == 1872,
              "edge_gain offset");
static_assert(__builtin_offsetof(RibbonWideUniforms, line_info) == 1888,
              "line_info offset");
