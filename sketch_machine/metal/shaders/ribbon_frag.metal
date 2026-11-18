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

struct RibbonUniforms {
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

struct ribbon_frag_out {
  float4 f_color [[color(0)]];
};

struct ribbon_frag_in {
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
  float4 v_owner [[user(locn10), flat]];
};

static inline __attribute__((always_inline)) bool occl_dense_at(
    thread const int2& pixel_coordinates, constant RibbonUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler) {
  bool density_threshold_disabled = uniforms.occl_min <= 0.0;
  bool is_dense;

  if (!density_threshold_disabled) {
    is_dense =
        (1.0 -
         exp(-occlusion_accumulation.read(uint2(pixel_coordinates), 0).x)) >=
        uniforms.occl_min;
  } else {
    is_dense = density_threshold_disabled;
  }

  return is_dense;
}

static inline __attribute__((always_inline)) float4 occl_front(
    thread int2& nearest_pixel, constant RibbonUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  int2 fragment_pixel = int2(fragment_position.xy);
  int2 occlusion_size =
      int2(occlusion_map.get_width(), occlusion_map.get_height());
  float4 nearest_occluder = occlusion_map.read(uint2(fragment_pixel), 0);
  nearest_pixel = fragment_pixel;
  for (int sample_offset_x = -1; sample_offset_x <= 1; sample_offset_x++) {
    for (int sample_offset_y = -1; sample_offset_y <= 1; sample_offset_y++) {
      int2 sample_pixel =
          clamp(fragment_pixel + int2(sample_offset_x, sample_offset_y),
                int2(0), occlusion_size - int2(1));
      float4 sample_occluder = occlusion_map.read(uint2(sample_pixel), 0);
      bool has_occluder = sample_occluder.z > 0.0;
      bool is_nearer_candidate;

      if (has_occluder) {
        bool previous_sample_empty = nearest_occluder.z <= 0.0;
        bool candidate_is_nearer;

        if (!previous_sample_empty) {
          candidate_is_nearer = sample_occluder.x < nearest_occluder.x;
        } else {
          candidate_is_nearer = previous_sample_empty;
        }

        is_nearer_candidate = candidate_is_nearer;
      } else {
        is_nearer_candidate = has_occluder;
      }

      bool candidate_is_dense;

      if (is_nearer_candidate) {
        int2 density_pixel = sample_pixel;
        candidate_is_dense =
            occl_dense_at(density_pixel, uniforms, occlusion_accumulation,
                          occlusion_accumulation_sampler);
      } else {
        candidate_is_dense = is_nearer_candidate;
      }

      if (candidate_is_dense) {
        nearest_occluder = sample_occluder;
        nearest_pixel = sample_pixel;
      }
    }
  }

  return nearest_occluder;
}

static inline __attribute__((always_inline)) float view_dist(
    thread const float3& world_position, constant RibbonUniforms& uniforms) {
  float distance_to_eye;

  if (uniforms.ortho > 0.5) {
    distance_to_eye =
        dot(uniforms.eye - world_position, float3(uniforms.view_back));
  } else {
    distance_to_eye = length(world_position - uniforms.eye);
  }

  return distance_to_eye;
}

static inline __attribute__((always_inline)) float cover_t(
    thread const float3& world_position, thread const float4& owner_ids,
    thread const float& foreign_owner_tolerance,
    constant RibbonUniforms& uniforms, texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  if (uniforms.occl_on < 0.5) {
    return 1.0;
  }

  int2 nearest_pixel;
  float4 nearest_front_occluder =
      occl_front(nearest_pixel, uniforms, occlusion_accumulation,
                 occlusion_accumulation_sampler, fragment_position,
                 occlusion_map, occlusion_sampler);
  int2 front_pixel = nearest_pixel;
  float4 front_occluder = nearest_front_occluder;
  bool front_sample_empty = front_occluder.z <= 0.0;
  bool front_sample_transparent;

  if (!front_sample_empty) {
    int2 density_pixel = front_pixel;
    front_sample_transparent =
        !occl_dense_at(density_pixel, uniforms, occlusion_accumulation,
                       occlusion_accumulation_sampler);
  } else {
    front_sample_transparent = front_sample_empty;
  }

  if (front_sample_transparent) {
    return 1.0;
  }

  float3 query_position = world_position;
  float view_distance = view_dist(query_position, uniforms);
  float depth_difference = view_distance - front_occluder.x;
  float depth_epsilon = 2e-5 * abs(view_distance);

  if (depth_difference <= depth_epsilon) {
    return 1.0;
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
      foreign_tolerance = foreign_owner_tolerance;
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

  if (within_depth_tolerance) {
    return 1.0;
  }

  return fast::max(uniforms.xray_behind,
                   exp(-occlusion_accumulation.read(uint2(front_pixel), 0).x));
}

static inline __attribute__((always_inline)) float cover(
    thread const float3& world_position, thread const float4& owner_ids,
    constant RibbonUniforms& uniforms, texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  float3 query_position = world_position;
  float4 query_owner_ids = owner_ids;
  float foreign_owner_tolerance = uniforms.occl_tol[2u];
  return cover_t(query_position, query_owner_ids, foreign_owner_tolerance,
                 uniforms, occlusion_accumulation,
                 occlusion_accumulation_sampler, fragment_position,
                 occlusion_map, occlusion_sampler);
}

static inline __attribute__((always_inline)) float hash12(
    thread const float2& noise_coordinates) {
  float3 hash_state = fract(noise_coordinates.xyx * 0.1031);
  hash_state += float3(dot(hash_state, hash_state.yzx + float3(33.33)));
  return fract((hash_state.x + hash_state.y) * hash_state.z);
}

static inline __attribute__((always_inline)) float vnoise(
    thread const float2& noise_coordinates) {
  float2 cell_origin = floor(noise_coordinates);
  float2 blend_weights = fract(noise_coordinates);
  blend_weights =
      (blend_weights * blend_weights) * (float2(3.0) - (blend_weights * 2.0));
  float2 corner00 = cell_origin;
  float2 corner10 = cell_origin + float2(1.0, 0.0);
  float2 corner01 = cell_origin + float2(0.0, 1.0);
  float2 corner11 = cell_origin + float2(1.0);
  return mix(mix(hash12(corner00), hash12(corner10), blend_weights.x),
             mix(hash12(corner01), hash12(corner11), blend_weights.x),
             blend_weights.y);
}

static inline __attribute__((always_inline)) float in_shadow(
    thread const float3& world_position, constant RibbonUniforms& uniforms,
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

static inline __attribute__((always_inline)) float band(
    thread const float& light_value, constant RibbonUniforms& uniforms) {
  if (uniforms.bands < 1.5) {
    return light_value;
  }

  return floor((fast::clamp(light_value, 0.0, 1.0) * (uniforms.bands - 1.0)) +
               0.5) /
         (uniforms.bands - 1.0);
}

static inline __attribute__((always_inline)) float3
light_edge(thread const float3& stroke_color, constant RibbonUniforms& uniforms,
           texture2d<float> shadow_map, sampler shadow_sampler,
           thread float3& stroke_world_position, thread float& edge_light,
           thread float& stroke_side, thread float& arc_length) {
  if (uniforms.light_on < 0.5) {
    return stroke_color;
  }

  float3 shadow_query_position =
      stroke_world_position + (float3(uniforms.sun_dir) * uniforms.shadow_lift);
  float shadow_sample =
      in_shadow(shadow_query_position, uniforms, shadow_map, shadow_sampler);
  float shadow_coverage = band(shadow_sample, uniforms);

  if (shadow_coverage > 0.5) {
    return (stroke_color * (1.0 - (uniforms.shadow_dark * shadow_coverage))) *
           mix(float3(1.0), float3(uniforms.sky_col), float3(shadow_coverage));
  }

  float edge_strength = smoothstep(0.12, 0.45, abs(edge_light));
  float sun_facing_side = stroke_side * sign(edge_light);
  float2 edge_noise_coordinates = floor(float2(arc_length * 2.5, 9.0));
  float edge_threshold = 0.3 + (0.18 * (hash12(edge_noise_coordinates) - 0.5));
  float3 edge_lit_color;

  if (sun_facing_side > edge_threshold) {
    edge_lit_color = fast::min(
        stroke_color * (float3(1.0) + (uniforms.sun_col *
                                       (uniforms.edge_gain * edge_strength))),
        float3(1.0));
  } else {
    edge_lit_color = stroke_color;
  }

  return edge_lit_color;
}

static inline __attribute__((always_inline)) bool on_layer(
    thread const float3& world_position, constant RibbonUniforms& uniforms,
    thread float4& fragment_position, texture2d<float> occlusion_map,
    sampler occlusion_sampler) {
  if (uniforms.occl_on < 0.5) {
    return false;
  }

  float4 front_occluder =
      occlusion_map.read(uint2(int2(fragment_position.xy)), 0);
  bool has_front_layer = front_occluder.z > 0.0;
  bool on_or_behind_front_layer;

  if (has_front_layer) {
    float3 query_position = world_position;
    on_or_behind_front_layer =
        view_dist(query_position, uniforms) >
        (front_occluder.x - (uniforms.occl_tol[0u] * front_occluder.z));
  } else {
    on_or_behind_front_layer = has_front_layer;
  }

  return on_or_behind_front_layer;
}

static inline __attribute__((always_inline)) float front_opacity(
    constant RibbonUniforms& uniforms, texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  if (uniforms.occl_on < 0.5) {
    return 0.0;
  }

  if (occlusion_map.read(uint2(int2(fragment_position.xy)), 0).z <= 0.0) {
    return 0.0;
  }

  return 1.0 -
         exp(-occlusion_accumulation.read(uint2(int2(fragment_position.xy)), 0)
                  .x);
}

static inline __attribute__((always_inline)) float3 plane_glaze(
    thread const float3& world_position, thread float3& stroke_color,
    constant RibbonUniforms& uniforms, texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, thread float4& fragment_position,
    texture2d<float> occlusion_map, sampler occlusion_sampler) {
  float3 selected_ray_origin;

  if (uniforms.ortho > 0.5) {
    selected_ray_origin =
        world_position + (float3(uniforms.view_back) * 1000000.0);
  } else {
    selected_ray_origin = uniforms.eye;
  }

  float3 ray_origin = selected_ray_origin;
  float3 ray_direction = world_position - ray_origin;
  float3 plane_intersection;
  bool plane_behind_stroke;
  for (int plane_index = 0; plane_index < uniforms.plane_n; plane_index++) {
    float3 plane_normal =
        cross(uniforms.plane_u[plane_index], uniforms.plane_v[plane_index]);
    float ray_plane_denominator = dot(plane_normal, ray_direction);

    if (abs(ray_plane_denominator) < 1e-12) {
      continue;
    }

    if (uniforms.ortho > 0.5) {
      float orthographic_distance =
          dot(plane_normal, uniforms.plane_o[plane_index] - world_position) /
          dot(plane_normal, float3(uniforms.view_back));
      plane_intersection =
          world_position + (float3(uniforms.view_back) * orthographic_distance);
      plane_behind_stroke = orthographic_distance <= 0.001;
    } else {
      float ray_parameter =
          dot(plane_normal, uniforms.plane_o[plane_index] - ray_origin) /
          ray_plane_denominator;

      if (ray_parameter <= 0.0) {
        continue;
      }

      plane_intersection = ray_origin + (ray_direction * ray_parameter);
      plane_behind_stroke = ray_parameter >= 0.9999;
    }

    bool skip_hidden_back_plane;

    if (plane_behind_stroke) {
      bool plane_has_no_back_opacity = uniforms.plane_k[plane_index].x <= 0.0;
      bool plane_not_on_front_layer;

      if (!plane_has_no_back_opacity) {
        float3 layer_query_position = plane_intersection;
        plane_not_on_front_layer =
            !on_layer(layer_query_position, uniforms, fragment_position,
                      occlusion_map, occlusion_sampler);
      } else {
        plane_not_on_front_layer = plane_has_no_back_opacity;
      }

      bool plane_not_visible;

      if (!plane_not_on_front_layer) {
        float3 coverage_query_position = plane_intersection;
        float4 plane_owner_ids = float4(1000000000.0, -1.0, -1.0, -1.0);
        plane_not_visible =
            cover(coverage_query_position, plane_owner_ids, uniforms,
                  occlusion_accumulation, occlusion_accumulation_sampler,
                  fragment_position, occlusion_map, occlusion_sampler) < 0.5;
      } else {
        plane_not_visible = plane_not_on_front_layer;
      }

      skip_hidden_back_plane = plane_not_visible;
    } else {
      skip_hidden_back_plane = plane_behind_stroke;
    }

    if (skip_hidden_back_plane) {
      continue;
    }

    float3 plane_offset = plane_intersection - uniforms.plane_o[plane_index];
    float plane_coordinate_u =
        dot(plane_offset, uniforms.plane_u[plane_index]) /
        fast::max(
            dot(uniforms.plane_u[plane_index], uniforms.plane_u[plane_index]),
            1e-12);
    float plane_coordinate_v =
        dot(plane_offset, uniforms.plane_v[plane_index]) /
        fast::max(
            dot(uniforms.plane_v[plane_index], uniforms.plane_v[plane_index]),
            1e-12);
    bool inside_plane_u = abs(plane_coordinate_u) <= 1.0;
    bool inside_plane_bounds;

    if (inside_plane_u) {
      inside_plane_bounds = abs(plane_coordinate_v) <= 1.0;
    } else {
      inside_plane_bounds = inside_plane_u;
    }

    if (inside_plane_bounds) {
      float plane_opacity = uniforms.plane_c[plane_index].w;
      bool on_front_layer;

      if (!plane_behind_stroke) {
        float3 front_layer_query_position = plane_intersection;
        on_front_layer =
            on_layer(front_layer_query_position, uniforms, fragment_position,
                     occlusion_map, occlusion_sampler);
      } else {
        on_front_layer = plane_behind_stroke;
      }

      if (on_front_layer) {
        plane_opacity = uniforms.plane_k[plane_index].x;
      } else {
        if (uniforms.plane_k[plane_index].x <= 0.0) {
          plane_opacity *=
              (1.0 - front_opacity(uniforms, occlusion_accumulation,
                                   occlusion_accumulation_sampler,
                                   fragment_position, occlusion_map,
                                   occlusion_sampler));
        } else {
          if (uniforms.plane_s[plane_index].x > 0.5) {
            plane_opacity = mix(
                plane_opacity, uniforms.plane_k[plane_index].x,
                front_opacity(uniforms, occlusion_accumulation,
                              occlusion_accumulation_sampler, fragment_position,
                              occlusion_map, occlusion_sampler));
          }
        }
      }

      stroke_color = mix(stroke_color, uniforms.plane_c[plane_index].xyz,
                         float3(plane_opacity));
    }
  }

  return stroke_color;
}

fragment ribbon_frag_out
ribbon_frag(ribbon_frag_in stage_input [[stage_in]],
            constant RibbonUniforms& uniforms [[buffer(0)]],
            texture2d<float> shadow_map [[texture(0)]],
            texture2d<float> occlusion_accumulation [[texture(1)]],
            texture2d<float> occlusion_map [[texture(2)]],
            sampler shadow_sampler [[sampler(0)]],
            sampler occlusion_accumulation_sampler [[sampler(1)]],
            sampler occlusion_sampler [[sampler(2)]],
            float4 fragment_position [[position]]) {
  ribbon_frag_out stage_output = {};

  if (stage_input.v_w <= 0.0) {
    discard_fragment();
  }

  float3 stroke_surface_position = stage_input.v_wpos;
  bool stroke_has_radius = stage_input.v_rad > 0.0;
  bool can_reconstruct_stroke_surface;

  if (stroke_has_radius) {
    can_reconstruct_stroke_surface = length(stage_input.v_tan) > 0.5;
  } else {
    can_reconstruct_stroke_surface = stroke_has_radius;
  }

  if (can_reconstruct_stroke_surface) {
    float3 selected_view_direction;

    if (uniforms.ortho > 0.5) {
      selected_view_direction = float3(uniforms.view_back);
    } else {
      selected_view_direction =
          fast::normalize(uniforms.eye - stage_input.v_wpos);
    }

    float3 view_direction = selected_view_direction;
    float3 view_normal =
        view_direction -
        (stage_input.v_tan * dot(view_direction, stage_input.v_tan));
    float view_normal_length = length(view_normal);

    if (view_normal_length > 1e-5) {
      view_normal /= float3(view_normal_length);
      float clamped_stroke_side = fast::clamp(stage_input.v_side, -1.0, 1.0);
      stroke_surface_position =
          stage_input.v_wpos +
          (((view_normal *
             sqrt(fast::max(1.0 - (clamped_stroke_side * clamped_stroke_side),
                            0.0))) +
            (cross(stage_input.v_tan, view_normal) * clamped_stroke_side)) *
           stage_input.v_rad);
    }
  }

  float3 query_position = stroke_surface_position;
  float4 query_owner_ids = stage_input.v_owner;
  float visibility =
      cover(query_position, query_owner_ids, uniforms, occlusion_accumulation,
            occlusion_accumulation_sampler, fragment_position, occlusion_map,
            occlusion_sampler);
  if (visibility <= 0.0) {
    discard_fragment();
  }

  float selected_stroke_opacity;

  if (visibility >= 1.0) {
    selected_stroke_opacity = uniforms.fade;
  } else {
    selected_stroke_opacity = visibility * uniforms.fade;
  }

  float stroke_opacity = selected_stroke_opacity;

  if (stage_input.v_brush > 0.5) {
    float brush_cross_section = (stage_input.v_side * 0.5) + 0.5;
    float2 bristle_coordinates =
        float2(brush_cross_section * 7.0, stage_input.v_arc * 0.06);
    float bristle_noise = vnoise(bristle_coordinates);
    float2 fine_grain_coordinates =
        floor(float2(brush_cross_section * 26.0, stage_input.v_arc * 0.5));
    float fine_grain_noise = hash12(fine_grain_coordinates);
    float2 fray_coordinates = float2(stage_input.v_arc * 0.9, 5.0);
    float edge_fray = smoothstep(
        0.45, 1.0,
        abs(stage_input.v_side) + (0.45 * (vnoise(fray_coordinates) - 0.5)));
    float2 dry_brush_coordinates =
        float2(stage_input.v_arc * 0.16, (brush_cross_section * 2.5) + 11.0);
    float dry_brush_noise =
        smoothstep(0.45, 0.85, vnoise(dry_brush_coordinates));
    float paint_gaps =
        (edge_fray * 0.9) + ((dry_brush_noise * 0.55) * fine_grain_noise);
    paint_gaps *= smoothstep(1.2, 3.5, stage_input.v_px);

    if (paint_gaps > (0.4 + (0.3 * fine_grain_noise))) {
      discard_fragment();
    }

    float paint_ridge =
        (0.72 + (0.45 * bristle_noise)) + (0.12 * (fine_grain_noise - 0.5));
    float3 paint_color = fast::clamp((stage_input.v_color.xyz * paint_ridge) +
                                         float3(0.06 * (bristle_noise - 0.5)),
                                     float3(0.0), float3(1.0));
    float3 paint_output_color;

    if (stage_input.v_color.w > 1.5) {
      paint_output_color = paint_color;
    } else {
      float3 paint_light_input = paint_color;
      float3 paint_glaze_position = stage_input.v_wpos;
      float3 paint_lit_color =
          light_edge(paint_light_input, uniforms, shadow_map, shadow_sampler,
                     stage_input.v_wpos, stage_input.v_light,
                     stage_input.v_side, stage_input.v_arc);
      float3 paint_glazed_color =
          plane_glaze(paint_glaze_position, paint_lit_color, uniforms,
                      occlusion_accumulation, occlusion_accumulation_sampler,
                      fragment_position, occlusion_map, occlusion_sampler);
      paint_output_color = paint_glazed_color;
    }

    stage_output.f_color =
        float4(paint_output_color,
               fast::min(stage_input.v_color.w, 1.0) * stroke_opacity);
    return stage_output;
  }

  float edge_distance = abs(stage_input.v_side);
  float2 streak_coordinates =
      floor(float2(stage_input.v_arc * 1.4, (stage_input.v_side + 1.0) * 2.5));
  float streak_noise = hash12(streak_coordinates);
  float2 tooth_coordinates =
      floor(float2(stage_input.v_arc * 5.0, (stage_input.v_side + 1.0) * 3.0)) +
      float2(31.0);
  float paper_tooth_noise = hash12(tooth_coordinates);
  float2 pressure_coordinates = float2(stage_input.v_arc * 0.35, 3.0);
  float pressure_noise = vnoise(pressure_coordinates);
  float pencil_holes = (mix(0.07, 0.38, smoothstep(0.3, 1.0, edge_distance)) +
                        (0.06 * (0.5 - pressure_noise))) *
                       uniforms.grain;
  pencil_holes *= smoothstep(1.2, 3.5, stage_input.v_px);

  if (((streak_noise * 0.55) + (paper_tooth_noise * 0.45)) < pencil_holes) {
    discard_fragment();
  }

  float pencil_shade =
      (0.94 + (0.12 * (1.0 - pressure_noise))) + (0.1 * (streak_noise - 0.5));
  float3 graphite_color =
      fast::min(stage_input.v_color.xyz * pencil_shade, float3(1.0));
  float3 graphite_output_color;

  if (stage_input.v_color.w > 1.5) {
    graphite_output_color = graphite_color;
  } else {
    float3 graphite_light_input = graphite_color;
    float3 graphite_glaze_position = stage_input.v_wpos;
    float3 graphite_lit_color =
        light_edge(graphite_light_input, uniforms, shadow_map, shadow_sampler,
                   stage_input.v_wpos, stage_input.v_light, stage_input.v_side,
                   stage_input.v_arc);
    float3 graphite_glazed_color =
        plane_glaze(graphite_glaze_position, graphite_lit_color, uniforms,
                    occlusion_accumulation, occlusion_accumulation_sampler,
                    fragment_position, occlusion_map, occlusion_sampler);
    graphite_output_color = graphite_glazed_color;
  }

  stage_output.f_color =
      float4(graphite_output_color,
             fast::min(stage_input.v_color.w, 1.0) * stroke_opacity);
  return stage_output;
}

static_assert(sizeof(RibbonUniforms) == 1904, "Uniform size");
static_assert(__builtin_offsetof(RibbonUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(RibbonUniforms, half_vp) == 64,
              "half_vp offset");
static_assert(__builtin_offsetof(RibbonUniforms, focal) == 72, "focal offset");
static_assert(__builtin_offsetof(RibbonUniforms, min_px) == 76,
              "min_px offset");
static_assert(__builtin_offsetof(RibbonUniforms, bias0) == 80, "bias offset");
static_assert(__builtin_offsetof(RibbonUniforms, eye) == 96, "eye offset");
static_assert(__builtin_offsetof(RibbonUniforms, view_back) == 112,
              "view_back offset");
static_assert(__builtin_offsetof(RibbonUniforms, ortho) == 124, "ortho offset");
static_assert(__builtin_offsetof(RibbonUniforms, sun_dir) == 128,
              "sun_dir offset");
static_assert(__builtin_offsetof(RibbonUniforms, grain) == 140, "grain offset");
static_assert(__builtin_offsetof(RibbonUniforms, fade) == 144, "fade offset");
static_assert(__builtin_offsetof(RibbonUniforms, light_on) == 148,
              "light_on offset");
static_assert(__builtin_offsetof(RibbonUniforms, sun_col) == 160,
              "sun_col offset");
static_assert(__builtin_offsetof(RibbonUniforms, sky_col) == 176,
              "sky_col offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_dark) == 188,
              "shadow_dark offset");
static_assert(__builtin_offsetof(RibbonUniforms, bands) == 192, "bands offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_on) == 196,
              "shadow_on offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_vp) == 208,
              "shadow_vp offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_bias) == 272,
              "shadow_bias offset");
static_assert(__builtin_offsetof(RibbonUniforms, shadow_lift) == 276,
              "shadow_lift offset");
static_assert(__builtin_offsetof(RibbonUniforms, dome) == 280, "dome offset");
static_assert(__builtin_offsetof(RibbonUniforms, occl_on) == 284,
              "occl_on offset");
static_assert(__builtin_offsetof(RibbonUniforms, xray_behind) == 288,
              "xray_behind offset");
static_assert(__builtin_offsetof(RibbonUniforms, occl_tol) == 304,
              "occl_tol offset");
static_assert(__builtin_offsetof(RibbonUniforms, occl_min) == 316,
              "occl_min offset");
static_assert(__builtin_offsetof(RibbonUniforms, aid_pass) == 320,
              "aid_pass offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_n) == 324,
              "plane_n offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_o) == 336,
              "plane_o offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_u) == 592,
              "plane_u offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_v) == 848,
              "plane_v offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_c) == 1104,
              "plane_c offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_k) == 1360,
              "plane_k offset");
static_assert(__builtin_offsetof(RibbonUniforms, plane_s) == 1616,
              "plane_s offset");
static_assert(__builtin_offsetof(RibbonUniforms, edge_gain) == 1872,
              "edge_gain offset");
static_assert(__builtin_offsetof(RibbonUniforms, line_info) == 1888,
              "line_info offset");
