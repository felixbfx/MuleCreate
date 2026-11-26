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

struct MeshUniforms {
  float4x4 mvp;
  float4x4 model;
  float3 eye;
  float4 tint;
  float alpha;
  float lift;
  float light_on;
  char padding0[4];
  float3 sun_dir;
  float3 sun_col;
  packed_float3 sky_col;
  float away_dark;
  float shadow_dark;
  float bands;
  float shadow_on;
  char padding1[4];
  float4x4 shadow_vp;
  float shadow_bias;
  float shadow_lift;
  float dome;
  char padding2[4];
  float4 line_info;
};

struct mesh_frag_out {
  float4 f_color [[color(0)]];
};

struct mesh_frag_in {
  float3 v_world [[user(locn0)]];
  float3 v_color [[user(locn1)]];
};

static inline __attribute__((always_inline)) float band(
    thread const float& light_value, constant MeshUniforms& uniforms) {
  if (uniforms.bands < 1.5) {
    return light_value;
  }

  return floor((fast::clamp(light_value, 0.0, 1.0) * (uniforms.bands - 1.0)) +
               0.5) /
         (uniforms.bands - 1.0);
}

static inline __attribute__((always_inline)) float in_shadow(
    thread const float3& world_position, constant MeshUniforms& uniforms,
    texture2d<float> shadow_map, sampler shadow_sampler) {
  if (uniforms.shadow_on < 0.5) {
    return 0.0;
  }

  float4 shadow_clip_position =
      uniforms.shadow_vp * float4(world_position, 1.0);
  float3 shadow_coordinates =
      ((shadow_clip_position.xyz / float3(shadow_clip_position.w)) * 0.5) +
      float3(0.5);
  bool outside_left = shadow_coordinates.x <= 0.0;
  bool outside_horizontal_bounds;

  if (!outside_left) {
    outside_horizontal_bounds = shadow_coordinates.x >= 1.0;
  } else {
    outside_horizontal_bounds = outside_left;
  }

  bool outside_horizontal_or_below;

  if (!outside_horizontal_bounds) {
    outside_horizontal_or_below = shadow_coordinates.y <= 0.0;
  } else {
    outside_horizontal_or_below = outside_horizontal_bounds;
  }

  bool outside_xy_bounds;

  if (!outside_horizontal_or_below) {
    outside_xy_bounds = shadow_coordinates.y >= 1.0;
  } else {
    outside_xy_bounds = outside_horizontal_or_below;
  }

  bool outside_shadow_bounds;

  if (!outside_xy_bounds) {
    outside_shadow_bounds = shadow_coordinates.z >= 1.0;
  } else {
    outside_shadow_bounds = outside_xy_bounds;
  }

  if (outside_shadow_bounds) {
    return 0.0;
  }

  float2 shadow_texel_size =
      float2(1.0) /
      float2(int2(shadow_map.get_width(), shadow_map.get_height()));
  float occluded_samples = 0.0;
  for (int sample_offset_x = -1; sample_offset_x <= 1; sample_offset_x++) {
    for (int sample_offset_y = -1; sample_offset_y <= 1; sample_offset_y++) {
      occluded_samples += float(
          (shadow_coordinates.z - uniforms.shadow_bias) >
          shadow_map
              .sample(shadow_sampler,
                      (shadow_coordinates.xy +
                       (float2(float(sample_offset_x), float(sample_offset_y)) *
                        shadow_texel_size)))
              .x);
    }
  }

  return occluded_samples / 9.0;
}

static inline __attribute__((always_inline)) float3 sun_and_sky(
    thread const float3& surface_normal, thread const float& direct_light,
    thread const float3& sky_color, constant MeshUniforms& uniforms) {
  float hemisphere_factor = 0.5 + (0.5 * surface_normal.y);
  float hemisphere_light = band(hemisphere_factor, uniforms);
  return mix(float3(direct_light),
             mix(float3(1.0), sky_color, float3(0.5)) * hemisphere_light,
             float3(uniforms.dome));
}

fragment mesh_frag_out mesh_frag(mesh_frag_in stage_input [[stage_in]],
                                 constant MeshUniforms& uniforms [[buffer(0)]],
                                 texture2d<float> shadow_map [[texture(0)]],
                                 sampler shadow_sampler [[sampler(0)]]) {
  mesh_frag_out stage_output = {};
  float3 surface_normal = fast::normalize(
      cross(dfdx(stage_input.v_world), dfdy(stage_input.v_world)));
  float3 view_direction = fast::normalize(uniforms.eye - stage_input.v_world);

  if (dot(surface_normal, view_direction) < 0.0) {
    surface_normal = -surface_normal;
  }

  float3 shaded_color;

  if (uniforms.light_on > 0.5) {
    float sun_facing = dot(surface_normal, uniforms.sun_dir);
    float sun_illumination_input = smoothstep(-0.25, 0.9, sun_facing);
    float sun_illumination = band(sun_illumination_input, uniforms);
    float3 shadow_offset_normal;

    if (sun_facing >= 0.0) {
      shadow_offset_normal = surface_normal;
    } else {
      shadow_offset_normal = -surface_normal;
    }

    float3 shadow_query_position =
        (stage_input.v_world + (shadow_offset_normal * uniforms.lift)) +
        (uniforms.sun_dir * uniforms.lift);
    float shadow_sample =
        in_shadow(shadow_query_position, uniforms, shadow_map, shadow_sampler);
    float shadow_coverage = band(shadow_sample, uniforms);
    float light_amount =
        fast::min(mix(1.0 - uniforms.away_dark, 1.0, sun_illumination),
                  1.0 - (uniforms.shadow_dark * shadow_coverage));
    float3 lighting_sample_normal = surface_normal;
    float lighting_sample_amount = light_amount;
    float3 sky_sample_color = float3(uniforms.sky_col);
    shaded_color = (stage_input.v_color *
                    sun_and_sky(lighting_sample_normal, lighting_sample_amount,
                                sky_sample_color, uniforms)) *
                   mix(float3(uniforms.sky_col), uniforms.sun_col,
                       float3(sun_illumination * (1.0 - shadow_coverage)));
  } else {
    float view_illumination_input = dot(surface_normal, view_direction);
    float view_illumination = band(view_illumination_input, uniforms);
    shaded_color = stage_input.v_color * (0.72 + (0.28 * view_illumination));
  }

  stage_output.f_color =
      float4(mix(shaded_color, uniforms.tint.xyz, float3(uniforms.tint.w)),
             uniforms.alpha);
  return stage_output;
}

static_assert(sizeof(MeshUniforms) == 336, "Uniform size");
static_assert(__builtin_offsetof(MeshUniforms, mvp) == 0, "mvp offset");
static_assert(__builtin_offsetof(MeshUniforms, model) == 64, "model offset");
static_assert(__builtin_offsetof(MeshUniforms, eye) == 128, "eye offset");
static_assert(__builtin_offsetof(MeshUniforms, tint) == 144, "tint offset");
static_assert(__builtin_offsetof(MeshUniforms, alpha) == 160, "alpha offset");
static_assert(__builtin_offsetof(MeshUniforms, lift) == 164, "lift offset");
static_assert(__builtin_offsetof(MeshUniforms, light_on) == 168,
              "light_on offset");
static_assert(__builtin_offsetof(MeshUniforms, sun_dir) == 176,
              "sun_dir offset");
static_assert(__builtin_offsetof(MeshUniforms, sun_col) == 192,
              "sun_col offset");
static_assert(__builtin_offsetof(MeshUniforms, sky_col) == 208,
              "sky_col offset");
static_assert(__builtin_offsetof(MeshUniforms, away_dark) == 220,
              "away_dark offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_dark) == 224,
              "shadow_dark offset");
static_assert(__builtin_offsetof(MeshUniforms, bands) == 228, "bands offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_on) == 232,
              "shadow_on offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_vp) == 240,
              "shadow_vp offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_bias) == 304,
              "shadow_bias offset");
static_assert(__builtin_offsetof(MeshUniforms, shadow_lift) == 308,
              "shadow_lift offset");
static_assert(__builtin_offsetof(MeshUniforms, dome) == 312, "dome offset");
static_assert(__builtin_offsetof(MeshUniforms, line_info) == 320,
              "line_info offset");
