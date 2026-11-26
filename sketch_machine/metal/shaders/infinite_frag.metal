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

struct InfiniteUniforms {
  float4x4 vp;
  float bias0;
  char padding0[12];
  float3 origin;
  float3 axis_u;
  float3 axis_v;
  float2 half_size;
  float has_core;
  char padding1[4];
  float2 tile_size;
  float2 tile_offset;
  float2 grid_step;
  char padding2[8];
  float4 fill;
  packed_float3 line;
  float grid_alpha;
  float tile_alpha;
  char padding3[12];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float reach;
  char padding4[12];
  float4 owner;
  float occl_on;
  float xray_behind;
  char padding5[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding6[12];
  float4 line_info;
};

struct infinite_frag_out {
  float4 f_color [[color(0)]];
};

struct infinite_frag_in {
  float3 v_world [[user(locn0)]];
};

static inline __attribute__((always_inline)) bool occl_dense_at(
    thread const int2& pixel_coordinates, constant InfiniteUniforms& uniforms,
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
    thread int2& nearest_pixel, constant InfiniteUniforms& uniforms,
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
    thread const float3& world_position, constant InfiniteUniforms& uniforms) {
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
    constant InfiniteUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
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
    constant InfiniteUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
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

static inline __attribute__((always_inline)) bool on_layer(
    thread const float3& world_position, constant InfiniteUniforms& uniforms,
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
    constant InfiniteUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
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

fragment infinite_frag_out
infinite_frag(infinite_frag_in stage_input [[stage_in]],
              constant InfiniteUniforms& uniforms [[buffer(0)]],
              texture2d<float> occlusion_accumulation [[texture(0)]],
              texture2d<float> occlusion_map [[texture(1)]],
              sampler occlusion_accumulation_sampler [[sampler(0)]],
              sampler occlusion_sampler [[sampler(1)]],
              float4 fragment_position [[position]]) {
  infinite_frag_out stage_output = {};
  float3 plane_offset = stage_input.v_world - uniforms.origin;
  float plane_coordinate_u = dot(plane_offset, uniforms.axis_u);
  float plane_coordinate_v = dot(plane_offset, uniforms.axis_v);
  float2 grid_coordinates =
      float2(plane_coordinate_u, plane_coordinate_v) / uniforms.grid_step;
  float2 grid_pixel_width = fast::max(fwidth(grid_coordinates), float2(1e-6));
  float2 tile_coordinates =
      (float2(plane_coordinate_u, plane_coordinate_v) + uniforms.tile_offset) /
      uniforms.tile_size;
  float2 tile_pixel_width = fast::max(fwidth(tile_coordinates), float2(1e-6));
  float3 query_position = stage_input.v_world;
  float4 query_owner_ids = uniforms.owner;
  float visibility =
      cover(query_position, query_owner_ids, uniforms, occlusion_accumulation,
            occlusion_accumulation_sampler, fragment_position, occlusion_map,
            occlusion_sampler);
  bool fully_occluded = visibility <= 0.0;
  bool discard_grid_fragment;

  if (!fully_occluded) {
    float3 layer_query_position = stage_input.v_world;
    discard_grid_fragment =
        on_layer(layer_query_position, uniforms, fragment_position,
                 occlusion_map, occlusion_sampler) != (uniforms.aid_pass < 0.5);
  } else {
    discard_grid_fragment = fully_occluded;
  }

  if (discard_grid_fragment) {
    discard_fragment();
  }

  visibility *=
      (1.0 - front_opacity(uniforms, occlusion_accumulation,
                           occlusion_accumulation_sampler, fragment_position,
                           occlusion_map, occlusion_sampler));
  bool core_enabled = uniforms.has_core > 0.5;
  bool inside_core_u;

  if (core_enabled) {
    inside_core_u = abs(plane_coordinate_u) < uniforms.half_size.x;
  } else {
    inside_core_u = core_enabled;
  }

  bool inside_core;

  if (inside_core_u) {
    inside_core = abs(plane_coordinate_v) < uniforms.half_size.y;
  } else {
    inside_core = inside_core_u;
  }

  if (inside_core) {
    discard_fragment();
  }

  float2 grid_edge_distance =
      abs(fract(grid_coordinates - float2(0.5)) - float2(0.5)) /
      grid_pixel_width;
  float grid_line_alpha =
      (1.0 -
       fast::min(fast::min(grid_edge_distance.x, grid_edge_distance.y), 1.0)) *
      (1.0 - smoothstep(0.12, 0.45,
                        fast::max(grid_pixel_width.x, grid_pixel_width.y)));
  float2 tile_edge_distance =
      abs(fract(tile_coordinates - float2(0.5)) - float2(0.5)) /
      tile_pixel_width;
  float tile_line_alpha =
      (1.0 -
       fast::min(fast::min(tile_edge_distance.x, tile_edge_distance.y), 1.0)) *
      (1.0 - smoothstep(0.12, 0.45,
                        fast::max(tile_pixel_width.x, tile_pixel_width.y)));
  float3 plane_normal =
      fast::normalize(cross(uniforms.axis_u, uniforms.axis_v));
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
        length(float2(plane_coordinate_u, plane_coordinate_v));
  } else {
    selected_view_distance = length(stage_input.v_world - uniforms.eye);
  }

  float view_distance = selected_view_distance;
  float distance_fade =
      (1.0 - smoothstep(uniforms.reach * 0.12, uniforms.reach, view_distance)) *
      smoothstep(0.02, 0.3, abs(dot(plane_normal, view_direction)));
  float line_ink = fast::max(uniforms.grid_alpha * grid_line_alpha,
                             uniforms.tile_alpha * tile_line_alpha);
  float plane_opacity =
      ((uniforms.fill.w + line_ink) * distance_fade) * visibility;

  if (plane_opacity < 0.002) {
    discard_fragment();
  }

  stage_output.f_color = float4(
      mix(uniforms.fill.xyz, float3(uniforms.line),
          float3(fast::min(
              1.0, line_ink /
                       fast::max(plane_opacity / fast::max(distance_fade, 1e-6),
                                 1e-6)))),
      plane_opacity);
  return stage_output;
}

static_assert(sizeof(InfiniteUniforms) == 352, "Uniform size");
static_assert(__builtin_offsetof(InfiniteUniforms, vp) == 0, "vp offset");
static_assert(__builtin_offsetof(InfiniteUniforms, bias0) == 64, "bias offset");
static_assert(__builtin_offsetof(InfiniteUniforms, origin) == 80,
              "origin offset");
static_assert(__builtin_offsetof(InfiniteUniforms, axis_u) == 96,
              "axis_u offset");
static_assert(__builtin_offsetof(InfiniteUniforms, axis_v) == 112,
              "axis_v offset");
static_assert(__builtin_offsetof(InfiniteUniforms, half_size) == 128,
              "half_size offset");
static_assert(__builtin_offsetof(InfiniteUniforms, has_core) == 136,
              "has_core offset");
static_assert(__builtin_offsetof(InfiniteUniforms, tile_size) == 144,
              "tile_size offset");
static_assert(__builtin_offsetof(InfiniteUniforms, tile_offset) == 152,
              "tile_offset offset");
static_assert(__builtin_offsetof(InfiniteUniforms, grid_step) == 160,
              "grid_step offset");
static_assert(__builtin_offsetof(InfiniteUniforms, fill) == 176, "fill offset");
static_assert(__builtin_offsetof(InfiniteUniforms, line) == 192, "line offset");
static_assert(__builtin_offsetof(InfiniteUniforms, grid_alpha) == 204,
              "grid_alpha offset");
static_assert(__builtin_offsetof(InfiniteUniforms, tile_alpha) == 208,
              "tile_alpha offset");
static_assert(__builtin_offsetof(InfiniteUniforms, eye) == 224, "eye offset");
static_assert(__builtin_offsetof(InfiniteUniforms, view_back) == 240,
              "view_back offset");
static_assert(__builtin_offsetof(InfiniteUniforms, ortho) == 252,
              "ortho offset");
static_assert(__builtin_offsetof(InfiniteUniforms, reach) == 256,
              "reach offset");
static_assert(__builtin_offsetof(InfiniteUniforms, owner) == 272,
              "owner offset");
static_assert(__builtin_offsetof(InfiniteUniforms, occl_on) == 288,
              "occl_on offset");
static_assert(__builtin_offsetof(InfiniteUniforms, xray_behind) == 292,
              "xray_behind offset");
static_assert(__builtin_offsetof(InfiniteUniforms, occl_tol) == 304,
              "occl_tol offset");
static_assert(__builtin_offsetof(InfiniteUniforms, occl_min) == 316,
              "occl_min offset");
static_assert(__builtin_offsetof(InfiniteUniforms, aid_pass) == 320,
              "aid_pass offset");
static_assert(__builtin_offsetof(InfiniteUniforms, line_info) == 336,
              "line_info offset");
