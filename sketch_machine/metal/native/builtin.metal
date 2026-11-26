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

#include <metal_raytracing>
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;
struct Vertex {
  float4 position [[position]];
  float2 uv;
};
vertex Vertex mule_vertex(uint vertex_index [[vertex_id]]) {
  float2 texture_coordinates =
      float2((vertex_index << 1) & 2, vertex_index & 2);
  return {float4(texture_coordinates * 2.0 - 1.0, 0, 1), texture_coordinates};
}

fragment float4 mule_fragment(Vertex stage_input [[stage_in]],
                              texture2d<float> scene_texture [[texture(0)]],
                              texture2d<float> overlay_texture [[texture(1)]],
                              sampler texture_sampler [[sampler(0)]]) {
  float3 scene_color =
      scene_texture.sample(texture_sampler, stage_input.uv).rgb;
  float4 overlay_color = overlay_texture.sample(
      texture_sampler, float2(stage_input.uv.x, 1 - stage_input.uv.y));
  return float4(overlay_color.rgb + scene_color * (1 - overlay_color.a), 1);
}

fragment float4 mule_copy_fragment(Vertex stage_input [[stage_in]],
                                   texture2d<float> source_texture
                                   [[texture(0)]],
                                   sampler texture_sampler [[sampler(0)]]) {
  return source_texture.sample(texture_sampler,
                               float2(stage_input.uv.x, 1 - stage_input.uv.y));
}

struct RayInput {
  packed_float3 origin;
  float minimum;
  packed_float3 direction;
  float maximum;
};
struct RayHit {
  float distance;
  uint instance;
  uint primitive;
  uint found;
};
kernel void mule_ray_hits(instance_acceleration_structure scene_acceleration
                          [[buffer(0)]],
                          const device RayInput* rays [[buffer(1)]],
                          device RayHit* hits [[buffer(2)]],
                          constant uint& ray_count [[buffer(3)]],
                          uint ray_index [[thread_position_in_grid]]) {
  if (ray_index >= ray_count)
    return;

  ray query_ray;
  query_ray.origin = float3(rays[ray_index].origin);
  query_ray.direction = float3(rays[ray_index].direction);
  query_ray.min_distance = rays[ray_index].minimum;
  query_ray.max_distance = rays[ray_index].maximum;
  intersector<triangle_data, instancing> triangle_intersector;
  triangle_intersector.assume_geometry_type(geometry_type::triangle);
  triangle_intersector.force_opacity(forced_opacity::opaque);
  auto intersection =
      triangle_intersector.intersect(query_ray, scene_acceleration, 0xff);
  hits[ray_index] = {intersection.distance, intersection.instance_id,
                     intersection.primitive_id,
                     uint(intersection.type == intersection_type::triangle)};
}
