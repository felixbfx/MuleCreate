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

struct ShadowLineUniforms {
  float4x4 svp;
  float4 line_info;
};

fragment void shadow_line_frag() {}

static_assert(sizeof(ShadowLineUniforms) == 80, "Uniform size");
static_assert(__builtin_offsetof(ShadowLineUniforms, svp) == 0, "svp offset");
static_assert(__builtin_offsetof(ShadowLineUniforms, line_info) == 64,
              "line_info offset");
