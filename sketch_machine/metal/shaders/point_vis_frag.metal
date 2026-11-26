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

struct PointVisUniforms {
  float count;
  char padding0[12];
  float4x4 vp;
  float2 size;
  char padding1[8];
  float3 eye;
  packed_float3 view_back;
  float ortho;
  float occl_on;
  float xray_behind;
  char padding2[8];
  packed_float3 occl_tol;
  float occl_min;
  float aid_pass;
  char padding3[12];
  float4 line_info;
};

struct point_vis_frag_out {
  float f_vis [[color(0)]];
};

struct point_vis_frag_in {
  int2 v_xy [[user(locn0)]];
  int v_front [[user(locn1)]];
  float3 v_p [[user(locn2)]];
};

static inline __attribute__((always_inline)) bool occl_dense_at(
    thread const int2& pixel_coordinates, constant PointVisUniforms& uniforms,
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

static inline __attribute__((always_inline)) float view_dist(
    thread const float3& world_position, constant PointVisUniforms& uniforms) {
  float distance_to_eye;

  if (uniforms.ortho > 0.5) {
    distance_to_eye =
        dot(uniforms.eye - world_position, float3(uniforms.view_back));
  } else {
    distance_to_eye = length(world_position - uniforms.eye);
  }

  return distance_to_eye;
}

static inline __attribute__((always_inline)) float cover_at(
    thread const float3& world_position, thread const float4& owner_ids,
    thread const int2& sample_pixel, constant PointVisUniforms& uniforms,
    texture2d<float> occlusion_accumulation,
    sampler occlusion_accumulation_sampler, texture2d<float> occlusion_map,
    sampler occlusion_sampler) {
  float4 front_occluder = occlusion_map.read(uint2(sample_pixel), 0);
  bool front_sample_empty = front_occluder.z <= 0.0;
  bool front_sample_transparent;

  if (!front_sample_empty) {
    int2 density_pixel = sample_pixel;
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
      foreign_tolerance = uniforms.occl_tol[2u];
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
                   exp(-occlusion_accumulation.read(uint2(sample_pixel), 0).x));
}

fragment point_vis_frag_out
point_vis_frag(point_vis_frag_in stage_input [[stage_in]],
               constant PointVisUniforms& uniforms [[buffer(0)]],
               texture2d<float> occlusion_accumulation [[texture(0)]],
               texture2d<float> occlusion_map [[texture(1)]],
               sampler occlusion_accumulation_sampler [[sampler(0)]],
               sampler occlusion_sampler [[sampler(1)]]) {
  point_vis_frag_out stage_output = {};
  bool occlusion_disabled = uniforms.occl_on < 0.5;
  bool skip_occlusion_test;

  if (!occlusion_disabled) {
    skip_occlusion_test = stage_input.v_front == 0;
  } else {
    skip_occlusion_test = occlusion_disabled;
  }

  if (skip_occlusion_test) {
    stage_output.f_vis = 1.0;
    return stage_output;
  }

  int2 occlusion_size =
      int2(occlusion_map.get_width(), occlusion_map.get_height());
  float visibility = 0.0;
  for (int sample_offset_x = -1; sample_offset_x <= 1; sample_offset_x++) {
    for (int sample_offset_y = -1; sample_offset_y <= 1; sample_offset_y++) {
      float3 query_position = stage_input.v_p;
      float4 query_owner_ids = float4(-1.0);
      int2 sample_pixel =
          clamp(stage_input.v_xy + int2(sample_offset_x, sample_offset_y),
                int2(0), occlusion_size - int2(1));
      visibility = fast::max(
          visibility,
          cover_at(query_position, query_owner_ids, sample_pixel, uniforms,
                   occlusion_accumulation, occlusion_accumulation_sampler,
                   occlusion_map, occlusion_sampler));
    }
  }

  stage_output.f_vis = visibility;
  return stage_output;
}

static_assert(sizeof(PointVisUniforms) == 192, "Uniform size");
static_assert(__builtin_offsetof(PointVisUniforms, count) == 0, "count offset");
static_assert(__builtin_offsetof(PointVisUniforms, vp) == 16, "vp offset");
static_assert(__builtin_offsetof(PointVisUniforms, size) == 80, "size offset");
static_assert(__builtin_offsetof(PointVisUniforms, eye) == 96, "eye offset");
static_assert(__builtin_offsetof(PointVisUniforms, view_back) == 112,
              "view_back offset");
static_assert(__builtin_offsetof(PointVisUniforms, ortho) == 124,
              "ortho offset");
static_assert(__builtin_offsetof(PointVisUniforms, occl_on) == 128,
              "occl_on offset");
static_assert(__builtin_offsetof(PointVisUniforms, xray_behind) == 132,
              "xray_behind offset");
static_assert(__builtin_offsetof(PointVisUniforms, occl_tol) == 144,
              "occl_tol offset");
static_assert(__builtin_offsetof(PointVisUniforms, occl_min) == 156,
              "occl_min offset");
static_assert(__builtin_offsetof(PointVisUniforms, aid_pass) == 160,
              "aid_pass offset");
static_assert(__builtin_offsetof(PointVisUniforms, line_info) == 176,
              "line_info offset");
