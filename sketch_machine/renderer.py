"""OpenGL (moderngl) and native Metal rendering: shaded meshes, planes with depth, GPU picking.

The viewport draws strokes and gizmos afterwards via QPainter as an overlay on top.
Picking: meshes are rendered into an invisible buffer that stores per pixel the
world position and the mesh number. Reading one pixel costs the same regardless
of the triangle count.
"""
import math
import time

import moderngl
import numpy as np


from . import theme
from . import counters, lru

PENCIL_GRAIN = 1.0  # strength of the pencil texture (0 = smooth strokes)
PICTURE_UNIT = 13  # texture unit of the projected images
SURF_UNIT = 12  # texture unit of the surface table (SURF_DATA)
SURF_COLS = 10  # texels per surface in the table
SURF_ROWS = 4096  # this many surfaces stacked, then the next column group (tests: small, checks the wrap)
# Surface table: all surfaces live in ONE buffer and are drawn with ONE call per pass. What differs per
# surface (material, coloring, state, position in the drag ...) is in a data texture, one row per surface;
# each corner carries the number of its surface (in_slot). Before: one call per surface and pass, each preceded by a
# dozen uniforms (harbor 11.11.: 944 surfaces, 3,868 calls per frame).
#   0: tint.rgb, material     1: anchor.xyz, tex_scale     2: offset.xyz, grid_step     3: state
#   4: grid_color.rgb, grid_alpha     5: lift, opacity, id, carried along?     6: coverage radius, opacity value, covers?
#   7–9: rows of the position matrix (only if carried along)
SURF_DATA = """
uniform sampler2D surf_data;
uniform int surf_rows;
vec4 surf(int slot, int col) {
    return texelFetch(surf_data, ivec2(col + %d * (slot / surf_rows), slot %% surf_rows), 0);
}
""" % SURF_COLS
PLANE_DEPTH_BIAS = 2e-5  # plane depth minimally to the back: strokes on the plane stay clearly in front
GL_MULTISAMPLE = 0x809D
# blend color normally but add up alpha: otherwise the image alpha drops below 1 and Qt
# mixes the window with the light background (semi-transparent surfaces would get lighter instead of see-through)
GL_DEPTH_BUFFER_BIT = 0x100
GL_POINT_SPRITE = 0x8861
# Coverage map: per pixel the frontmost covering layer (surfaces, meshes). Drawing aids and
# strokes compare their distance to the camera against it (world units, the same at every zoom level). Tolerance perpendicular
# to the surface in surfel radii: (own surface - e.g. drawn on it or outline line, membership unknown -
# planes, free mode, foreign surface)
OCCL_TOL = (0.5, 0.3, 0.02)
OCCL_MIN_COVER = 0.0  # without minimum opacity: what lies behind shines through according to the opacity of the layer
OCCL_ACC_UNIT = 7  # texture unit of the opacity of the frontmost layer
WIRE_DEPTH_BIAS = 2e-4  # wireframe on opaque meshes: depth offset toward the camera (fraction of the depth range)
MESH_OCCL_RADIUS = 1e-6  # meshes in the coverage map: "radius" in world units - practically none, the pixels count
SURFACE_OCCL_RADIUS = 0.02  # surfaces in the coverage map: "radius" as a fraction of their diagonal (tolerance on them)
OCCL_UNIT = 6  # texture unit of the coverage map
XRAY_BEHIND = 0.4  # X-Ray: what is hidden at this fraction of the opacity of the front layer
SHADOW_SIZE = 4096  # resolution of the shadow map
SHADOW_BIAS = 0.0015  # bias against self-shadowing (depth 0..1 over the scene)
SHADOW_LINE_PX = 2.0  # lines cast shadows at this width (pixels of the shadow map)
SHADOW_UNIT = 5  # texture unit of the shadow map
LIGHT_EDGE_GAIN = 0.25  # light edge of the lines: this much lighter than the line (subtle, no hard white fringe)
STROKE_BIAS = 3e-5  # depth lead of the strokes (they lie visibly on surfaces)
BLEND_OVER = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE)

NOISE = """
// Noise for paper and pencil structure (hash without a texture, stable on all drivers)
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), f.x),
               mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x), f.y);
}
"""

PAPER_LIB = """
// 3D noise (for the view sphere of the background)
float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y);
    float b = mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y);
    return mix(a, b, f.z);
}
// Paper on a surface, p in world units. fw = world size of one pixel: details that would be smaller
// than a pixel are faded out (otherwise it flickers when zooming out)
float paper2(vec2 p, float fw) {
    float blotch = vnoise(p * 0.7) - 0.5;
    float fibers = (vnoise(p * vec2(26.0, 5.0)) + vnoise(p * vec2(5.0, 26.0) + 41.0)) * 0.5 - 0.5;
    float tooth = vnoise(p * 70.0) - 0.5;
    fibers *= 1.0 - smoothstep(0.02, 0.08, fw);
    tooth *= 1.0 - smoothstep(0.008, 0.03, fw);
    return blotch * 0.04 + fibers * 0.6 + tooth * 0.7;  // Blotches only subliminal now
}
// triplanar: projected from three directions, blended by the surface normal
float paper_triplanar(vec3 p, vec3 n, float fw) {
    vec3 w = pow(abs(n), vec3(4.0));
    w /= max(w.x + w.y + w.z, 1e-6);
    return paper2(p.yz, fw) * w.x + paper2(p.zx + 13.0, fw) * w.y + paper2(p.xy + 29.0, fw) * w.z;
}
"""

SHADOW_LIB = """
uniform float bands;  // light steps: 0 = continuous, otherwise this many brightness steps (cel shading)
float band(float x) {
    if (bands < 1.5) return x;
    return floor(clamp(x, 0.0, 1.0) * (bands - 1.0) + 0.5) / (bands - 1.0);
}
uniform float shadow_on;
uniform mat4 shadow_vp;
uniform sampler2D shadow_map;
uniform float shadow_bias;
uniform float shadow_lift;  // world distance towards the sun by which lines are checked
float in_shadow(vec3 world) {
    // Sun shadow map: is something between this point and the sun? (3x3 averaged: soft edges)
    if (shadow_on < 0.5) return 0.0;
    vec4 sp = shadow_vp * vec4(world, 1.0);
    vec3 q = sp.xyz / sp.w * 0.5 + 0.5;
    if (q.x <= 0.0 || q.x >= 1.0 || q.y <= 0.0 || q.y >= 1.0 || q.z >= 1.0) return 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    float hit = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            hit += q.z - shadow_bias > texture(shadow_map, q.xy + vec2(i, j) * texel).r ? 1.0 : 0.0;
    return hit / 9.0;
}
uniform float dome;  // share of the sky light (dome light): shapes stay readable in shadow too
vec3 sun_and_sky(vec3 n, float sun_part, vec3 sky) {
    // sun_part: sunlight including shadow; plus light from the whole sky - full at the top, less towards the bottom, slightly blue
    float up = band(0.5 + 0.5 * n.y);
    return mix(vec3(sun_part), up * mix(vec3(1.0), sky, 0.5), dome);
}
"""

MESH_VS = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_world;
out vec3 v_color;
void main() {
    v_world = (model * vec4(in_pos, 1.0)).xyz;
    v_color = in_color;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

MESH_FS = """
#version 330
uniform vec3 eye;
uniform vec4 tint;
uniform float alpha;
uniform float lift;  // shadow test point this far away from the surface (world units)
uniform float light_on;
uniform vec3 sun_dir;
uniform vec3 sun_col;
uniform vec3 sky_col;
uniform float away_dark;
uniform float shadow_dark;
in vec3 v_world;
in vec3 v_color;
out vec4 f_color;
""" + SHADOW_LIB + """
void main() {
    // Surface normal from screen derivatives: no normals needed, flat shaded
    vec3 n = normalize(cross(dFdx(v_world), dFdy(v_world)));
    vec3 l = normalize(eye - v_world);
    if (dot(n, l) < 0.0) n = -n;  // the side that is seen
    vec3 c;
    if (light_on > 0.5) {
        // Light mode: sun, far side and shadow as with the surfaces
        float nl = dot(n, sun_dir);
        float k = band(smoothstep(-0.25, 0.9, nl));
        float sh = band(in_shadow(v_world + (nl >= 0.0 ? n : -n) * lift + sun_dir * lift));
        float lit = min(mix(1.0 - away_dark, 1.0, k), 1.0 - shadow_dark * sh);
        c = v_color * sun_and_sky(n, lit, sky_col) * mix(sky_col, sun_col, k * (1.0 - sh));
    } else {
        float d = band(dot(n, l));
        c = v_color * (0.72 + 0.28 * d);  // imported meshes: mid grey, flat shaded (pencil stays readable)
    }
    f_color = vec4(mix(c, tint.rgb, tint.a), alpha);
}
"""

WIRE_VS = """
#version 330
uniform mat4 mvp;
uniform float bias;  // share of the depth: edges lie slightly in front of their surface
in vec3 in_pos;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    gl_Position.z -= bias * gl_Position.w;
}
"""

WIRE_FS = """
#version 330
uniform vec4 color;  // wireframe of the meshes: edge colour with the opacity of the slider
out vec4 f_color;
void main() {
    f_color = color;
}
"""

OCCL_VIEW = """
uniform vec3 eye;
uniform vec3 view_back;
uniform float ortho;
"""

OCCL_LIB = """
uniform sampler2D occl_map;  // per pixel: distance of the front-most covering layer, |cos| view/normal, radius, surface id
uniform float occl_on;
uniform float xray_behind;   // hidden: 0 = invisible, X-Ray: soft (opacity)
uniform vec3 occl_tol;       // tolerance in radii: own surface, unknown, other surface
uniform sampler2D occl_acc;  // opacity of the front-most layer per pixel (summed up as -log(1 - a))
uniform float occl_min;      // from this opacity on it covers (0: do not check opacity)
bool occl_dense_at(ivec2 q) {
    return occl_min <= 0.0 || 1.0 - exp(-texelFetch(occl_acc, q, 0).r) >= occl_min;
}
bool occl_dense() { return occl_dense_at(ivec2(gl_FragCoord.xy)); }
// front-most covering layer in the 3x3 neighbourhood (at the outline of a surface the pixel centre often lies just beside it - without
// the neighbourhood hidden things would show through there as dots); returns the map there and writes its position to q
vec4 occl_front(out ivec2 q) {
    ivec2 c = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(occl_map, 0);
    vec4 best = texelFetch(occl_map, c, 0);
    q = c;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            ivec2 t = clamp(c + ivec2(i, j), ivec2(0), size - 1);
            vec4 m = texelFetch(occl_map, t, 0);
            if (m.b > 0.0 && (best.b <= 0.0 || m.r < best.r) && occl_dense_at(t)) {
                best = m;
                q = t;
            }
        }
    return best;
}
float view_dist(vec3 p) { return ortho > 0.5 ? dot(eye - p, view_back) : length(p - eye); }
// 1.0: in front or on the front-most surface; otherwise xray_behind. owners: surface ids p belongs to (-1 = none),
// other: tolerance towards other surfaces (radii)
float cover_t(vec3 p, vec4 owners, float other) {
    if (occl_on < 0.5) return 1.0;
    ivec2 q;
    vec4 m = occl_front(q);
    if (m.b <= 0.0 || !occl_dense_at(q)) return 1.0;  // nothing covers here (or only thinly)
    float d = view_dist(p);
    float diff = d - m.r;
    float eps = 2e-5 * abs(d);
    if (diff <= eps) return 1.0;
    bool own = m.a >= 0.0 && any(equal(owners, vec4(m.a)));
    bool known = m.a >= 0.0 && max(max(owners.x, owners.y), max(owners.z, owners.w)) >= 0.0;
    float tol = own ? occl_tol.x : (known ? other : occl_tol.y);
    // measure perpendicular to the surface (view angle does not matter), along the view at most a few radii
    if (diff * m.g <= tol * m.b + eps && diff <= (own ? 4.0 : 2.0) * m.b) return 1.0;
    // behind: as far as the front-most layer lets through (covering surface: nothing; X-Ray: soft)
    return max(xray_behind, exp(-texelFetch(occl_acc, q, 0).r));
}
float cover(vec3 p, vec4 owners) { return cover_t(p, owners, occl_tol.z); }
// Opacity pass: does p belong to the front-most layer exactly at its pixel? Not in the 3x3 neighbourhood like cover() - at
// every depth edge (a nearer part in the neighbouring pixel) the rear side would otherwise drop out, have no opacity there,
// and what lies behind would show through as speckles (dense meshes: edges everywhere)
bool front_here(vec3 p, vec4 owners) {
    if (occl_on < 0.5) return true;
    vec4 m = texelFetch(occl_map, ivec2(gl_FragCoord.xy), 0);
    if (m.b <= 0.0) return true;
    float d = view_dist(p);
    float diff = d - m.r;
    float eps = 2e-5 * abs(d);
    if (diff <= eps) return true;
    bool own = m.a >= 0.0 && any(equal(owners, vec4(m.a)));
    bool known = m.a >= 0.0 && max(max(owners.x, owners.y), max(owners.z, owners.w)) >= 0.0;
    float tol = own ? occl_tol.x : (known ? occl_tol.z : occl_tol.y);
    return diff * m.g <= tol * m.b + eps && diff <= (own ? 4.0 : 2.0) * m.b;
}
// Surfaces among each other: visible if it is in front at any pixel of the 3x3 neighbourhood - at the outline of a surface in front
// of it the depth test per sample decides (anti-aliasing) instead of paper showing through there
float cover_at(vec3 p, vec4 owners, ivec2 q) {
    vec4 m = texelFetch(occl_map, q, 0);
    if (m.b <= 0.0 || !occl_dense_at(q)) return 1.0;
    float d = view_dist(p);
    float diff = d - m.r;
    float eps = 2e-5 * abs(d);
    if (diff <= eps) return 1.0;
    bool own = m.a >= 0.0 && any(equal(owners, vec4(m.a)));
    bool known = m.a >= 0.0 && max(max(owners.x, owners.y), max(owners.z, owners.w)) >= 0.0;
    float tol = own ? occl_tol.x : (known ? occl_tol.z : occl_tol.y);
    if (diff * m.g <= tol * m.b + eps && diff <= (own ? 4.0 : 2.0) * m.b) return 1.0;
    return max(xray_behind, exp(-texelFetch(occl_acc, q, 0).r));
}
float cover_edge(vec3 p, vec4 owners) {
    if (occl_on < 0.5) return 1.0;
    ivec2 c = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(occl_map, 0);
    float v = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            v = max(v, cover_at(p, owners, clamp(c + ivec2(i, j), ivec2(0), size - 1)));
    return v;
}
// Drawing aids: does p lie on or behind the front-most surface? Then it is drawn in front of the surface
// and covered by it (soft at the edges, invisible inside)
uniform float aid_pass;  // 0: pass under the surfaces, 1: above them (in front of or without a surface, glaze of the active plane)
// Opacity of the front-most layer here (0 = no surface): inactive drawing aids in front fade out by that much
float front_opacity() {
    if (occl_on < 0.5) return 0.0;
    if (texelFetch(occl_map, ivec2(gl_FragCoord.xy), 0).b <= 0.0) return 0.0;
    return 1.0 - exp(-texelFetch(occl_acc, ivec2(gl_FragCoord.xy), 0).r);
}
bool on_layer(vec3 p) {
    if (occl_on < 0.5) return false;
    vec4 m = texelFetch(occl_map, ivec2(gl_FragCoord.xy), 0);
    return m.b > 0.0 && view_dist(p) > m.r - occl_tol.x * m.b;
}
"""

SHADOW_LINE_VS = """
#version 330
uniform mat4 svp;
in vec3 in_pos;
void main() {
    gl_Position = svp * vec4(in_pos, 1.0);
}
"""

SHADOW_LINE_FS = """
#version 330
void main() {
}
"""

PICK_VS = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
out vec3 v_world;
void main() {
    v_world = (model * vec4(in_pos, 1.0)).xyz;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

PICK_GATHER_VS = """
#version 330
uniform float count;
in vec2 in_xy;          // pixel in the position buffer
flat out ivec2 v_xy;
void main() {
    v_xy = ivec2(in_xy);
    gl_PointSize = 1.0;
    gl_Position = vec4((float(gl_VertexID) + 0.5) / count * 2.0 - 1.0, 0.0, 0.0, 1.0);  // pixel k of the row
}
"""

PICK_GATHER_FS = """
#version 330
uniform sampler2D pick_tex;
uniform sampler2D prim_tex;
flat in ivec2 v_xy;
layout(location = 0) out vec4 f_out;
layout(location = 1) out float f_prim;
void main() {
    ivec2 xy = clamp(v_xy, ivec2(0), textureSize(pick_tex, 0) - 1);
    f_out = texelFetch(pick_tex, xy, 0);
    f_prim = texelFetch(prim_tex, xy, 0).r;
}
"""

# visibility of individual points (point symbols in edit): one pixel of a row per point, there the same occlusion rule
# as for strokes (coverage map, 3x3 neighborhood) - only this row is read
POINT_VIS_VS = """
#version 330
uniform float count;
uniform mat4 vp;
uniform vec2 size;
in vec3 in_pos;
flat out ivec2 v_xy;
flat out int v_front;
out vec3 v_p;
void main() {
    vec4 c = vp * vec4(in_pos, 1.0);
    v_front = c.w > 0.0 ? 1 : 0;
    vec2 ndc = c.xy / max(abs(c.w), 1e-12);
    v_xy = ivec2(floor((ndc * 0.5 + 0.5) * size));
    v_p = in_pos;
    gl_PointSize = 1.0;
    gl_Position = vec4((float(gl_VertexID) + 0.5) / count * 2.0 - 1.0, 0.0, 0.0, 1.0);  // pixel k of the row
}
"""

POINT_VIS_FS = """
#version 330
""" + OCCL_VIEW + OCCL_LIB + """
flat in ivec2 v_xy;
flat in int v_front;
in vec3 v_p;
out float f_vis;
void main() {
    if (occl_on < 0.5 || v_front == 0) { f_vis = 1.0; return; }
    ivec2 size = textureSize(occl_map, 0);
    float v = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            v = max(v, cover_at(v_p, vec4(-1.0), clamp(v_xy + ivec2(i, j), ivec2(0), size - 1)));
    f_vis = v;
}
"""

PICK_FS = """
#version 330
uniform float mesh_index;
in vec3 v_world;
layout(location = 0) out vec4 f_out;
layout(location = 1) out float f_prim;  // number of the triangle hit (exact intersection only with this one)
void main() {
    f_out = vec4(v_world, mesh_index);
    f_prim = float(gl_PrimitiveID);
}
"""

PICTURE_FS = """
#version 330
uniform mat4 picture_m;
uniform sampler2D picture;
uniform vec3 eye;
uniform vec3 view_back;
uniform float ortho;
uniform vec4 owner;
uniform float under;
in vec3 v_world;
out vec4 f_color;
""" + "%OCCL%" + """
void main() {
    vec4 c = picture_m * vec4(v_world, 1.0);
    if (c.w <= 0.0) discard;
    vec2 uv = c.xy / c.w * 0.5 + 0.5;
    vec4 tex = texture(picture, uv);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) discard;
    float vis = cover(v_world, owner);
    if (vis <= 0.0) discard;
    f_color = vec4(tex.rgb, tex.a * vis);
}
"""

FLAT_VS = """
#version 330
uniform mat4 vp;
uniform float bias;  // > 0: minimally backwards (depth pass of the planes)
in vec3 in_pos;
out vec3 v_world;
void main() {
    v_world = in_pos;
    vec4 c = vp * vec4(in_pos, 1.0);
    gl_Position = vec4(c.xy, c.z + bias * c.w, c.w);
}
"""

FLAT_FS = """
#version 330
uniform vec4 color;
uniform float grain;  // tracing paper: paper structure in the fill (fixed on the plane)
uniform vec3 eye;
uniform vec3 view_back;
uniform float ortho;
uniform float milk;   // milkier under a flat view (like real tracing paper)
uniform vec4 owner;   // plane/drawing surface (for the coverage map)
uniform float under;  // on a surface: this opacity (glaze of the plane, 0 = gone), < 0 = unchanged
uniform float soft;   // 1: uninvolved plane (neither drawing plane nor selected) - also IN FRONT of a surface only the glaze
in vec3 v_world;
out vec4 f_color;
""" + NOISE + PAPER_LIB + OCCL_LIB + """
void main() {
    // Derivatives before every discard (afterwards they are undefined in the neighbouring pixels: black pixels at the surface edge)
    vec3 cr = cross(dFdx(v_world), dFdy(v_world));
    vec3 n = length(cr) > 1e-20 ? normalize(cr) : vec3(0.0, 1.0, 0.0);
    float fw_w = length(fwidth(v_world));
    float vis = cover(v_world, owner);  // drawing aids behind a surface: gone (X-Ray: soft)
    if (vis <= 0.0) discard;
    float tex = 0.0;
    float a = color.a;
    if (grain > 0.0) {
        tex = paper_triplanar(v_world, n, fw_w);
        vec3 view = ortho > 0.5 ? view_back : normalize(eye - v_world);
        a = min(a * (1.0 + milk * (1.0 - abs(dot(n, view)))), 1.0);
    }
    bool low = on_layer(v_world);
    if (aid_pass < 0.5) {
        if (!low) discard;  // under the surfaces: only what lies on/behind them
    } else if (low) {
        if (under <= 0.0) discard;  // above a surface only the glaze of the plane
        a = under;
    } else if (under <= 0.0) {
        a *= 1.0 - front_opacity();  // without glaze in front of a surface: does not cover it
    } else if (soft > 0.5) {
        a = mix(a, under, front_opacity());  // uninvolved in front of a surface: only the glaze, in front of the paper full
    }
    if (a <= 0.0) discard;
    f_color = vec4(color.rgb * (1.0 + grain * 0.06 * tex), a * vis);
}
"""

INFINITE_FS = """
#version 330
uniform vec3 origin;
uniform vec3 axis_u;
uniform vec3 axis_v;
uniform vec2 half_size;   // core (the visible rectangle): the normal plane draws there
uniform float has_core;   // 0: base grid without core (everywhere)
uniform vec2 tile_size;   // tile size
uniform vec2 tile_offset; // tile borders: edges of the core (plane) or through the origin (base grid)
uniform vec2 grid_step;
uniform vec4 fill;        // milk: colour + opacity
uniform vec3 line;
uniform float grid_alpha;
uniform float tile_alpha; // tiles in the size of the plane (rectangle repeats)
uniform vec3 eye;
uniform vec3 view_back;
uniform float ortho;
uniform float reach;      // fade out up to this distance
uniform vec4 owner;
in vec3 v_world;
out vec4 f_color;
""" + OCCL_LIB + """
void main() {
    vec3 d = v_world - origin;
    float s = dot(d, axis_u), t = dot(d, axis_v);
    vec2 g = vec2(s, t) / grid_step;
    vec2 fw = max(fwidth(g), vec2(1e-6));  // derivatives before every discard
    vec2 k = (vec2(s, t) + tile_offset) / tile_size;
    vec2 kw = max(fwidth(k), vec2(1e-6));
    float vis = cover(v_world, owner);
    if (vis <= 0.0 || on_layer(v_world) != (aid_pass < 0.5)) discard;
    vis *= 1.0 - front_opacity();  // in front of a surface: does not cover it
    if (has_core > 0.5 && abs(s) < half_size.x && abs(t) < half_size.y) discard;
    vec2 f = abs(fract(g - 0.5) - 0.5) / fw;
    float lines = (1.0 - min(min(f.x, f.y), 1.0)) * (1.0 - smoothstep(0.12, 0.45, max(fw.x, fw.y)));
    vec2 kf = abs(fract(k - 0.5) - 0.5) / kw;
    float tiles = (1.0 - min(min(kf.x, kf.y), 1.0)) * (1.0 - smoothstep(0.12, 0.45, max(kw.x, kw.y)));
    vec3 n = normalize(cross(axis_u, axis_v));
    vec3 view = ortho > 0.5 ? view_back : normalize(eye - v_world);
    float dist = ortho > 0.5 ? length(vec2(s, t)) : length(v_world - eye);
    float fade = (1.0 - smoothstep(reach * 0.12, reach, dist)) * smoothstep(0.02, 0.3, abs(dot(n, view)));
    float ink = max(grid_alpha * lines, tile_alpha * tiles);
    float a = (fill.a + ink) * fade * vis;
    if (a < 0.002) discard;
    f_color = vec4(mix(fill.rgb, line, min(1.0, ink / max(a / max(fade, 1e-6), 1e-6))), a);
}
"""

PAPER_VS = """
#version 330
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);  // one triangle over the whole screen
    gl_Position = vec4(p * 2.0 - 1.0, 0.999, 1.0);
}
"""

PAPER_FS = """
#version 330
uniform vec3 paper;
uniform vec2 size;
uniform float stain;
uniform float grain;
uniform float vignette;
uniform vec3 cam_right;   // camera basis: the paper lies on a sphere around the camera
uniform vec3 cam_up;
uniform vec3 cam_back;
uniform float view_tan;   // tan(half the opening angle), also orthographic (then it only turns along)
out vec4 f_color;
""" + NOISE + PAPER_LIB + """
void main() {
    vec2 ndc = gl_FragCoord.xy / size * 2.0 - 1.0;
    vec3 dir = normalize(-cam_back + ndc.x * view_tan * (size.x / size.y) * cam_right + ndc.y * view_tan * cam_up);
    // Structure on the view sphere: 3D noise on the direction, without seams and poles
    float blotch = vnoise3(dir * 3.0) * 0.6 + vnoise3(dir * 7.0 + 5.0) * 0.4 - 0.5;   // large, soft blotches
    float fibers = (vnoise3(dir * vec3(380.0, 70.0, 380.0)) + vnoise3(dir * vec3(70.0, 380.0, 70.0) + 9.0)) * 0.5 - 0.5;
    float tooth = vnoise3(dir * 1300.0) - 0.5;                                        // grain
    vec2 uv = gl_FragCoord.xy / size - 0.5;
    float shade = 1.0 + stain * 2.0 * blotch + grain * (0.9 * tooth + 0.6 * fibers);
    shade *= 1.0 - vignette * dot(uv, uv) * 1.6;
    f_color = vec4(paper * shade, 1.0);
}
"""

RIBBON_VS = """
#version 330
uniform mat4 vp;
uniform vec2 half_vp;   // half framebuffer size in pixels
uniform float focal;    // 1 / tan(fov/2), orthographic 1 / half image height (w is then 1)
uniform float min_px;   // minimum width so that distant strokes do not vanish
uniform float bias;     // pull slightly towards the camera so that strokes on surfaces do not flicker
in vec3 in_pos;
in vec3 in_prev;
in vec3 in_next;
in float in_width;
in float in_side;
in float in_arc;   // arc length in stroke widths (texture sticks to the stroke)
in vec4 in_color;
uniform vec3 eye;        // light mode: view direction per point (perspective) ...
uniform vec3 view_back;  // ... or orthographic
uniform float ortho;
uniform vec3 sun_dir;    // direction to the sun
out vec4 v_color;
out float v_w;
out float v_side;
out float v_arc;
out float v_px;
out float v_brush;
out float v_light;       // > 0: side +1 faces the sun, < 0: side -1; magnitude: how clearly
out vec3 v_wpos;
out vec3 v_tan;           // stroke direction in space (tube for the occlusion, 0: paint stroke)
out float v_rad;          // half stroke width in world units (0: flat)
in vec4 in_owner;         // drawing surfaces the stroke belongs to (-1 = none)
flat out vec4 v_owner;
in vec3 in_normal;        // paint stroke: surface normal (otherwise 0)
in float in_layer;        // paint stroke: order 0..1 (newer = higher = on top)
void main() {
    v_wpos = in_pos;
    v_owner = in_owner;
    float brush = step(1.5, abs(in_side));  // brush: side +-2
    if (length(in_normal) > 0.5) {
        // Paint stroke: flat in the surface (width across the stroke in the surface plane), becomes narrower under a flat view
        // like paint on paper; depth = paint order (the newest is on top, hiding is handled by the
        // coverage map)
        float fside = in_side / (1.0 + brush);
        vec3 T = in_next - in_prev;
        T = length(T) > 1e-9 ? normalize(T) : vec3(1.0, 0.0, 0.0);
        vec3 B = cross(normalize(in_normal), T);
        B = length(B) > 1e-6 ? normalize(B) : vec3(0.0, 1.0, 0.0);
        vec4 cf = vp * vec4(in_pos + B * fside * in_width * 0.5, 1.0);
        float wf = max(cf.w, 1e-5);
        gl_Position = vec4(cf.xy, (0.9 - 1.8 * in_layer) * cf.w, cf.w);
        v_color = in_color;
        v_w = cf.w;
        v_side = fside;
        v_arc = in_arc;
        v_px = max(in_width * focal * half_vp.y / wf, min_px);
        v_brush = brush;
        v_light = 0.0;  // no painted light edge (lies on the surface); shadow applies
        v_tan = vec3(0.0);
        v_rad = 0.0;
        return;
    }
    float side = in_side / (1.0 + brush);
    vec4 c = vp * vec4(in_pos, 1.0);
    vec4 a = vp * vec4(in_prev, 1.0);
    vec4 b = vp * vec4(in_next, 1.0);
    float w = max(c.w, 1e-5);
    vec2 sc = c.xy / w * half_vp;
    vec2 d = b.xy / max(b.w, 1e-5) * half_vp - a.xy / max(a.w, 1e-5) * half_vp;
    if (length(d) < 1e-6) d = vec2(1.0, 0.0);
    vec2 n = normalize(vec2(-d.y, d.x));
    // World width -> pixels (perspective makes distant strokes thinner), with minimum width
    float px = max(in_width * focal * half_vp.y / w, min_px);
    vec2 s = sc + n * px * 0.5 * side;
    gl_Position = vec4(s / half_vp * w, c.z - bias * w, w);
    v_color = in_color;
    v_w = c.w;
    v_side = side;
    v_arc = in_arc;
    v_px = px;
    v_brush = brush;
    // Cross direction of the stroke in space (perpendicular to stroke and view), sign like the screen side
    vec3 T = in_next - in_prev;
    T = length(T) > 1e-9 ? normalize(T) : vec3(1.0, 0.0, 0.0);
    v_tan = T;
    v_rad = in_width * 0.5;
    vec3 V = ortho > 0.5 ? view_back : normalize(eye - in_pos);
    vec3 B = cross(V, T);
    float bl = length(B);
    v_light = 0.0;
    if (bl > 1e-5) {
        B /= bl;
        vec4 cb = vp * vec4(in_pos + B * max(in_width, 1e-4), 1.0);
        vec2 sb = cb.xy / max(cb.w, 1e-5) * half_vp - sc;
        if (dot(sb, n) < 0.0) B = -B;
        v_light = dot(B, sun_dir);
    }
}
"""

OCCL_ACC_MESH_FS = """
#version 330
""" + OCCL_VIEW + OCCL_LIB + """
uniform vec4 owner;
uniform float acc_value;  // -log(1 - opacity): meshes full (20), surfaces according to their slider
in vec3 v_world;
out float f_acc;
void main() {
    if (!front_here(v_world, owner)) discard;
    f_acc = acc_value;
}
"""

OCCL_MESH_FS = """
#version 330
""" + OCCL_VIEW + """
uniform float radius;
uniform float mesh_id;
const float OCCL_MIN_PX = 3.0;  // tolerance at least this many pixels
in vec3 v_world;
out vec4 f_cover;
void main() {
    vec3 n = normalize(cross(dFdx(v_world), dFdy(v_world)));
    vec3 view = ortho > 0.5 ? view_back : normalize(eye - v_world);
    float d = ortho > 0.5 ? dot(eye - v_world, view_back) : length(v_world - eye);
    // Tolerance at least a few pixels (in world units here): otherwise a small or distant surface covers itself
    // in the 3x3 neighbourhood and vanishes when zooming out
    float px = length(fwidth(v_world));
    f_cover = vec4(d, abs(dot(n, view)), max(radius, OCCL_MIN_PX * px), mesh_id);
}
"""

# the same passes for surfaces: radius, id and opacity per surface from the surface table
OCCL_SURF_FS = OCCL_MESH_FS.replace("uniform float radius;\nuniform float mesh_id;\n", SURF_DATA + "flat in int v_slot;\n") \
    .replace("void main() {\n", "void main() {\n    float radius = surf(v_slot, 6).x;\n    float mesh_id = surf(v_slot, 5).z;\n")
OCCL_ACC_SURF_FS = OCCL_ACC_MESH_FS.replace("uniform vec4 owner;\nuniform float acc_value;", SURF_DATA + "flat in int v_slot;") \
    .replace("void main() {\n", "void main() {\n    vec4 owner = vec4(surf(v_slot, 5).z, -1.0, -1.0, -1.0);\n"
             "    float acc_value = surf(v_slot, 6).y;\n")
assert "v_slot" in OCCL_SURF_FS and "acc_value = surf" in OCCL_ACC_SURF_FS

# --- Mesh block: all imported meshes in one buffer (position included), per point the number of its part ---
# Per part three texels in the block table: 0 tint (rgba) · 1 (id, drawn, -, -) · 2 (picking number, triangle
# start high, low, -). Highlighting and hiding change only the table, not the buffer.
BLOCK_UNIT = 11  # texture unit of the block table
BLOCK_TAB = """
uniform sampler2D block_tab;
vec4 tab(int s, int j) { int k = s * 3 + j; return texelFetch(block_tab, ivec2(k % 1024, k / 1024), 0); }
"""
BLOCK_VS = """
#version 330
uniform mat4 mvp;
uniform int vis_col;  // 1: drawn (image, coverage, shadow) - 2: in this picking
in vec3 in_pos;
in vec3 in_color;
in float in_slot;
out vec3 v_world;
out vec3 v_color;
flat out int v_slot;
""" + BLOCK_TAB + """
void main() {
    v_slot = int(in_slot + 0.5);
    v_world = in_pos;
    v_color = in_color;
    bool shown = vis_col == 2 ? tab(v_slot, 2).x > 0.5 : tab(v_slot, 1).y > 0.5;
    gl_Position = shown ? mvp * vec4(in_pos, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);  // hidden: outside, discarded
}
"""
BLOCK_MESH_FS = MESH_FS.replace("uniform vec4 tint;\n", BLOCK_TAB + "flat in int v_slot;\n") \
    .replace("void main() {\n", "void main() {\n    vec4 tint = tab(v_slot, 0);\n", 1)
BLOCK_PICK_FS = PICK_FS.replace("uniform float mesh_index;\n", BLOCK_TAB + "flat in int v_slot;\n") \
    .replace("    f_out = vec4(v_world, mesh_index);\n    f_prim = float(gl_PrimitiveID);\n",
             "    vec4 t = tab(v_slot, 2);\n    f_out = vec4(v_world, t.x);\n"
             "    f_prim = float(gl_PrimitiveID - (int(t.y) * 65536 + int(t.z)));  // triangle in the own mesh\n")
BLOCK_OCCL_FS = OCCL_MESH_FS.replace("uniform float mesh_id;\n", BLOCK_TAB + "flat in int v_slot;\n") \
    .replace("void main() {\n", "void main() {\n    float mesh_id = tab(v_slot, 1).x;\n", 1)
BLOCK_OCCL_ACC_FS = OCCL_ACC_MESH_FS.replace("uniform vec4 owner;\n", BLOCK_TAB + "flat in int v_slot;\n") \
    .replace("void main() {\n", "void main() {\n    vec4 owner = vec4(tab(v_slot, 1).x, -1.0, -1.0, -1.0);\n", 1)
assert "tab(v_slot, 0)" in BLOCK_MESH_FS and "t.x" in BLOCK_PICK_FS and "tab(v_slot, 1).x" in BLOCK_OCCL_FS \
    and "vec4 owner = vec4(tab" in BLOCK_OCCL_ACC_FS
BLOCK_MIN = 64  # from this many imported meshes a block is drawn instead of one per mesh
BLOCK_REBUILD = 64  # this many meshes outside the block (new, moved) -> rebuild as soon as nothing is being dragged

RIBBON_FS = """
#version 330
uniform float grain;
uniform float fade;  // < 1: hidden strokes (behind hidden drawing surfaces) weakened
in vec4 v_color;
in float v_w;
in float v_side;
in float v_arc;
in float v_px;
in float v_brush;
in float v_light;
in vec3 v_wpos;
in vec3 v_tan;
in float v_rad;
uniform float light_on;
uniform vec3 sun_dir;
uniform vec3 sun_col;
uniform vec3 sky_col;
uniform float shadow_dark;
flat in vec4 v_owner;
out vec4 f_color;
""" + NOISE + SHADOW_LIB + OCCL_VIEW + OCCL_LIB + """
// Helper planes tint what lies behind them: planes cover nothing and are drawn before the strokes - without this
// strokes and painting behind them would show through untinted (harbour test 09.11.). Per plane: does it lie between eye and p
// (inside its rectangle), the colour is mixed with its fill colour (opacity like the plane itself).
uniform int plane_n;
uniform vec3 plane_o[16];
uniform vec3 plane_u[16];  // half edge in u (world)
uniform vec3 plane_v[16];  // half edge in v (world)
uniform vec4 plane_c[16];  // fill colour + opacity
uniform float plane_k[16];  // glaze of the plane on a surface, 0: none
uniform float plane_s[16];  // 1: uninvolved plane - also in front of a surface only the glaze
vec3 plane_glaze(vec3 p, vec3 rgb) {
    vec3 e = ortho > 0.5 ? p + view_back * 1.0e6 : eye;
    vec3 d = p - e;
    for (int i = 0; i < plane_n; i++) {
        vec3 nrm = cross(plane_u[i], plane_v[i]);
        float den = dot(nrm, d);
        if (abs(den) < 1e-12) continue;
        vec3 q;       // where the view ray hits the plane
        bool behind;  // the plane lies behind the point here
        if (ortho > 0.5) {
            // Eye at infinity: distance along the view direction in world units. (Computed via the distant substitute eye
            // every plane nearer than 100 units in front of the stroke counted as "not in front" - in orthographic
            // views planes practically never tinted the strokes behind them.)
            float s = dot(nrm, plane_o[i] - p) / dot(nrm, view_back);
            q = p + view_back * s;
            behind = s <= 1e-3;
        } else {
            float t = dot(nrm, plane_o[i] - e) / den;
            if (t <= 0.0) continue;  // plane lies behind the eye
            q = e + d * t;
            behind = t >= 1.0 - 1e-4;
        }
        // ... then it only counts where it is nevertheless drawn over the surface: if a plane dips into a surface,
        // it still glazes the surface a bit (tolerance of the coverage map). Painting and strokes on this
        // surface lie IN FRONT of the plane there and would stay untinted - they ate into the edge of the plane as blotches
        // (artist 12.11.).
        // (asked like the plane itself: as a foreign drawing aid with an id that no surface carries - with "no id"
        // the wide tolerance for unknown things would apply, and painting would be tinted even above a deeply buried plane)
        if (behind && (plane_k[i] <= 0.0 || !on_layer(q) || cover(q, vec4(1.0e9, -1.0, -1.0, -1.0)) < 0.5)) continue;
        vec3 x = q - plane_o[i];
        float a = dot(x, plane_u[i]) / max(dot(plane_u[i], plane_u[i]), 1e-12);
        float b = dot(x, plane_v[i]) / max(dot(plane_v[i], plane_v[i]), 1e-12);
        if (abs(a) <= 1.0 && abs(b) <= 1.0) {
            // as opaque as the plane is itself drawn at this spot (FLAT_FS): if it lies on a
            // surface, only its glaze; in front fully - unless it does not glaze (neither drawing plane nor selected), then
            // it yields to the surface. Otherwise painting behind the plane would be tinted differently from the surface it
            // lies on (blotchy, "eaten" edge), and an uninvolved plane would tint only the strokes around.
            float alpha = plane_c[i].a;
            if (behind || on_layer(q)) alpha = plane_k[i];
            else if (plane_k[i] <= 0.0) alpha *= 1.0 - front_opacity();
            else if (plane_s[i] > 0.5) alpha = mix(alpha, plane_k[i], front_opacity());
            rgb = mix(rgb, plane_c[i].rgb, alpha);
        }
    }
    return rgb;
}

uniform float edge_gain;  // light edge: line colour brighter by this much (tinted in sun colour)
vec3 light_edge(vec3 rgb) {
    // Light mode: painted light edge - a narrow, slightly broken bright rim on the side facing the sun;
    // the stroke itself stays flat as drawn
    if (light_on < 0.5) return rgb;
    float sh = band(in_shadow(v_wpos + sun_dir * shadow_lift));  // a bit towards the sun: lines on surfaces are not shadowed by themselves
    if (sh > 0.5) return rgb * (1.0 - shadow_dark * sh) * mix(vec3(1.0), sky_col, sh);  // in shadow: no edge
    float strength = smoothstep(0.12, 0.45, abs(v_light));
    float toward = v_side * sign(v_light);
    float thr = 0.3 + 0.18 * (hash12(floor(vec2(v_arc * 2.5, 9.0))) - 0.5);
    return toward > thr ? min(rgb * (1.0 + edge_gain * strength * sun_col), vec3(1.0)) : rgb;
}
void main() {
    if (v_w <= 0.0) discard;  // behind the camera
    // Occlusion at the surface of an imagined tube instead of at the centre line: a thick stroke through a surface
    // is cut round (as an ellipse), not at a straight edge across the band (harbour test 09.11.)
    vec3 p = v_wpos;
    if (v_rad > 0.0 && length(v_tan) > 0.5) {
        vec3 V = ortho > 0.5 ? view_back : normalize(eye - v_wpos);
        vec3 np = V - dot(V, v_tan) * v_tan;
        float nl = length(np);
        if (nl > 1e-5) {
            np /= nl;
            float s = clamp(v_side, -1.0, 1.0);
            p = v_wpos + v_rad * (sqrt(max(1.0 - s * s, 0.0)) * np + s * cross(v_tan, np));
        }
    }
    float vis = cover(p, v_owner);  // behind surfaces/meshes: gone (X-Ray: soft)
    if (vis <= 0.0) discard;
    float k = vis >= 1.0 ? fade : vis * fade;
    if (v_brush > 0.5) {
        // Oil brush: bristle stripes along the stroke, fraying edges, colour sometimes rich, sometimes drier
        float across = v_side * 0.5 + 0.5;
        float bristle = vnoise(vec2(across * 7.0, v_arc * 0.06));          // bristle grooves along the stroke
        float fine = hash12(floor(vec2(across * 26.0, v_arc * 0.5)));       // single bristles
        float fray = smoothstep(0.45, 1.0, abs(v_side) + 0.45 * (vnoise(vec2(v_arc * 0.9, 5.0)) - 0.5));
        float dry = smoothstep(0.45, 0.85, vnoise(vec2(v_arc * 0.16, across * 2.5 + 11.0)));  // dry brush
        float gaps = fray * 0.9 + dry * 0.55 * fine;
        gaps *= smoothstep(1.2, 3.5, v_px);  // do not perforate very thin strokes
        if (gaps > 0.4 + 0.3 * fine) discard;
        float ridge = 0.72 + 0.45 * bristle + 0.12 * (fine - 0.5);       // colour darker in the grooves, lighter on the ridges
        vec3 oil = clamp(v_color.rgb * ridge + 0.06 * (bristle - 0.5), 0.0, 1.0);
        // Opacity > 1 = colour of the surface (selection, hover): always at full brightness - neither light and
        // shadow nor the tint of a helper plane in front may darken it (otherwise the selected is not recognisable)
        f_color = vec4(v_color.a > 1.5 ? oil : plane_glaze(v_wpos, light_edge(oil)), min(v_color.a, 1.0) * k);
        return;
    }
    // Pencil: graphite only sticks to the "teeth" of the paper. Grain in stroke coordinates
    // (along: arc length, across: side), more patchy towards the edges. Gaps instead of transparency:
    // so crossings become denser and darker, and drawing the strokes a second time changes nothing.
    float edge = abs(v_side);
    // Graphite stripes along the stroke direction: elongated cells (long along, narrow across)
    float streak = hash12(floor(vec2(v_arc * 1.4, (v_side + 1.0) * 2.5)));
    float tooth = hash12(floor(vec2(v_arc * 5.0, (v_side + 1.0) * 3.0)) + 31.0);
    float pressure = vnoise(vec2(v_arc * 0.35, 3.0));  // pressure varies along the stroke
    float holes = (mix(0.07, 0.38, smoothstep(0.3, 1.0, edge)) + 0.06 * (0.5 - pressure)) * grain;
    holes *= smoothstep(1.2, 3.5, v_px);  // do not perforate very thin strokes
    if (streak * 0.55 + tooth * 0.45 < holes) discard;
    float shade = 0.94 + 0.12 * (1.0 - pressure) + 0.1 * (streak - 0.5);
    vec3 graphite = min(v_color.rgb * shade, vec3(1.0));
    f_color = vec4(v_color.a > 1.5 ? graphite : plane_glaze(v_wpos, light_edge(graphite)), min(v_color.a, 1.0) * k);
}
"""


SURFACE_VS = """
#version 330
uniform mat4 vp;
uniform float bias;  // < 0: minimally backwards (depth pass)
uniform int occluders_only;  // 1 (coverage map): helper surfaces cover nothing - their corners drop out of the image
""" + SURF_DATA + """
in vec3 in_pos;
in vec3 in_normal;  // smoothed normal (milkier under a flat view)
in vec3 in_calm;    // strongly smoothed normal: only for light with light steps (small waves without step edges)
in float in_slot;   // number of the surface in the surface table
out vec3 v_world;
out vec3 v_normal;
out vec3 v_calm;
flat out int v_slot;
void main() {
    int slot = int(in_slot + 0.5);
    v_slot = slot;
    vec3 p = in_pos;
    v_normal = in_normal;
    v_calm = in_calm;
    if (occluders_only == 1 && surf(slot, 6).z < 0.5) {
        v_world = p;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    if (surf(slot, 5).w > 0.5) {  // carried along rigidly (gizmo drag): net from the start of the drag times position matrix
        // only rotation/uniform scaling: rotate the normals along, length as before
        mat4 model = transpose(mat4(surf(slot, 7), surf(slot, 8), surf(slot, 9), vec4(0.0, 0.0, 0.0, 1.0)));
        mat3 r = mat3(model);
        p = (model * vec4(in_pos, 1.0)).xyz;
        v_normal = normalize(r * in_normal) * length(in_normal);
        v_calm = normalize(r * in_calm) * length(in_calm);
    }
    v_world = p;
    vec4 c = vp * vec4(p, 1.0);
    gl_Position = vec4(c.xy, c.z - bias * c.w, c.w);
}
"""

CAP_VS = """
#version 330
uniform mat4 vp;
uniform vec2 half_vp;
uniform float focal;
uniform float min_px;
uniform float bias;
in vec3 in_pos;
in float in_width;
in vec4 in_color;
out vec4 v_color;
out float v_w;
out float v_seed;
out float v_px;
out vec3 v_wpos;
in vec4 in_owner;
flat out vec4 v_owner;
in float in_style;     // 2: brush
flat out float v_brush;
in vec3 in_normal;     // paint stroke: surface normal (otherwise 0)
in float in_layer;
flat out float v_flat;
flat out mat2 v_inv;   // paint stroke: screen pixel -> coordinates of the flat disc
flat out float v_size;
void main() {
    v_wpos = in_pos;
    v_owner = in_owner;
    v_brush = step(1.5, in_style);
    vec4 c = vp * vec4(in_pos, 1.0);
    float w = max(c.w, 1e-5);
    v_flat = 0.0;
    v_inv = mat2(1.0);
    gl_PointSize = max(in_width * focal * half_vp.y / w, min_px);
    gl_Position = vec4(c.xy, c.z - bias * c.w, c.w);
    if (length(in_normal) > 0.5) {
        // Paint stroke: round end flat in the surface - in the image an ellipse (point sprite covers it)
        vec3 N = normalize(in_normal);
        vec3 U = normalize(cross(N, abs(N.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
        vec3 W = cross(N, U);
        float r = in_width * 0.5;
        vec2 s0 = c.xy / w * half_vp;
        vec4 cu = vp * vec4(in_pos + U * r, 1.0);
        vec4 cw = vp * vec4(in_pos + W * r, 1.0);
        vec2 a = cu.xy / max(cu.w, 1e-5) * half_vp - s0;
        vec2 b = cw.xy / max(cw.w, 1e-5) * half_vp - s0;
        mat2 m = mat2(a, b);
        v_flat = abs(determinant(m)) > 1e-6 ? 1.0 : 0.0;
        v_inv = v_flat > 0.5 ? inverse(m) : mat2(1.0);
        gl_PointSize = 2.0 * sqrt(dot(a, a) + dot(b, b)) + 2.0;
        gl_Position = vec4(c.xy, (0.9 - 1.8 * in_layer) * c.w, c.w);
    }
    v_size = gl_PointSize;
    v_color = in_color;
    v_w = c.w;
    v_seed = fract(dot(in_pos, vec3(12.9898, 78.233, 37.719)));
    v_px = gl_PointSize;
}
"""

CAP_FS = """
#version 330
uniform float grain;
uniform float fade;  // < 1: hidden strokes (behind hidden drawing surfaces) weakened
in vec4 v_color;
in float v_w;
in float v_seed;
in float v_px;
in vec3 v_wpos;
flat in vec4 v_owner;
flat in float v_brush;
flat in float v_flat;
flat in mat2 v_inv;
flat in float v_size;
out vec4 f_color;
uniform float light_on;
uniform vec3 sun_dir;
uniform vec3 sky_col;
uniform float shadow_dark;
""" + NOISE + SHADOW_LIB + OCCL_VIEW + OCCL_LIB + """
// Helper planes tint what lies behind them: planes cover nothing and are drawn before the strokes - without this
// strokes and painting behind them would show through untinted (harbour test 09.11.). Per plane: does it lie between eye and p
// (inside its rectangle), the colour is mixed with its fill colour (opacity like the plane itself).
uniform int plane_n;
uniform vec3 plane_o[16];
uniform vec3 plane_u[16];  // half edge in u (world)
uniform vec3 plane_v[16];  // half edge in v (world)
uniform vec4 plane_c[16];  // fill colour + opacity
uniform float plane_k[16];  // glaze of the plane on a surface, 0: none
uniform float plane_s[16];  // 1: uninvolved plane - also in front of a surface only the glaze
vec3 plane_glaze(vec3 p, vec3 rgb) {
    vec3 e = ortho > 0.5 ? p + view_back * 1.0e6 : eye;
    vec3 d = p - e;
    for (int i = 0; i < plane_n; i++) {
        vec3 nrm = cross(plane_u[i], plane_v[i]);
        float den = dot(nrm, d);
        if (abs(den) < 1e-12) continue;
        vec3 q;       // where the view ray hits the plane
        bool behind;  // the plane lies behind the point here
        if (ortho > 0.5) {
            // Eye at infinity: distance along the view direction in world units. (Computed via the distant substitute eye
            // every plane nearer than 100 units in front of the stroke counted as "not in front" - in orthographic
            // views planes practically never tinted the strokes behind them.)
            float s = dot(nrm, plane_o[i] - p) / dot(nrm, view_back);
            q = p + view_back * s;
            behind = s <= 1e-3;
        } else {
            float t = dot(nrm, plane_o[i] - e) / den;
            if (t <= 0.0) continue;  // plane lies behind the eye
            q = e + d * t;
            behind = t >= 1.0 - 1e-4;
        }
        // ... then it only counts where it is nevertheless drawn over the surface: if a plane dips into a surface,
        // it still glazes the surface a bit (tolerance of the coverage map). Painting and strokes on this
        // surface lie IN FRONT of the plane there and would stay untinted - they ate into the edge of the plane as blotches
        // (artist 12.11.).
        // (asked like the plane itself: as a foreign drawing aid with an id that no surface carries - with "no id"
        // the wide tolerance for unknown things would apply, and painting would be tinted even above a deeply buried plane)
        if (behind && (plane_k[i] <= 0.0 || !on_layer(q) || cover(q, vec4(1.0e9, -1.0, -1.0, -1.0)) < 0.5)) continue;
        vec3 x = q - plane_o[i];
        float a = dot(x, plane_u[i]) / max(dot(plane_u[i], plane_u[i]), 1e-12);
        float b = dot(x, plane_v[i]) / max(dot(plane_v[i], plane_v[i]), 1e-12);
        if (abs(a) <= 1.0 && abs(b) <= 1.0) {
            // as opaque as the plane is itself drawn at this spot (FLAT_FS): if it lies on a
            // surface, only its glaze; in front fully - unless it does not glaze (neither drawing plane nor selected), then
            // it yields to the surface. Otherwise painting behind the plane would be tinted differently from the surface it
            // lies on (blotchy, "eaten" edge), and an uninvolved plane would tint only the strokes around.
            float alpha = plane_c[i].a;
            if (behind || on_layer(q)) alpha = plane_k[i];
            else if (plane_k[i] <= 0.0) alpha *= 1.0 - front_opacity();
            else if (plane_s[i] > 0.5) alpha = mix(alpha, plane_k[i], front_opacity());
            rgb = mix(rgb, plane_c[i].rgb, alpha);
        }
    }
    return rgb;
}

vec3 cap_light(vec3 rgb) {
    // Light mode: shadow exactly like the band at the last line point (v_wpos is this point) - the cap has
    // the same brightness as the line piece next to it, no matter how it faces the camera
    if (light_on < 0.5) return rgb;
    float sh = band(in_shadow(v_wpos + sun_dir * shadow_lift));
    if (sh > 0.5) return rgb * (1.0 - shadow_dark * sh) * mix(vec3(1.0), sky_col, sh);
    return rgb;
}
void main() {
    float vis = cover(v_wpos, v_owner);
    if (vis <= 0.0) discard;
    // round stroke ends: circle in the point sprite (paint stroke: flat disc in the surface), edge grained like the stroke
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    if (v_flat > 0.5) {
        vec2 px = (gl_PointCoord - 0.5) * v_size;
        q = v_inv * vec2(px.x, -px.y);  // point sprite counts y downwards
    }
    float r = length(q);
    if (r > 1.0 || v_w <= 0.0) discard;
    if (v_brush > 0.5) {
        // Brush end: round as with the pen, but with fraying edge and bristle structure like the stroke
        float ang = atan(q.y, q.x);
        float fray = vnoise(vec2(ang * 2.2 + v_seed * 40.0, 3.0));
        if (r > 0.72 + 0.2 * fray * smoothstep(1.2, 3.5, v_px)) discard;  // as wide as the fraying stroke
        float bristle = vnoise(q * 3.5 + v_seed * 17.0);
        vec3 rgb = clamp(v_color.rgb * (0.78 + 0.4 * bristle), 0.0, 1.0);
        f_color = vec4(v_color.a > 1.5 ? rgb : plane_glaze(v_wpos, cap_light(rgb)),
                       min(v_color.a, 1.0) * (vis >= 1.0 ? fade : vis * fade));
        return;
    }
    float fine = hash12(floor(vec2(q.x * 2.5, (q.y + 1.0) * 3.0)) + v_seed * 97.0);  // grain as fine as in the stroke
    float holes = mix(0.07, 0.38, smoothstep(0.3, 1.0, r)) * grain * smoothstep(1.2, 3.5, v_px);
    if (fine < holes) discard;
    f_color = vec4(v_color.a > 1.5 ? v_color.rgb : plane_glaze(v_wpos, cap_light(v_color.rgb)),
                   min(v_color.a, 1.0) * (vis >= 1.0 ? fade : vis * fade));
}
"""

MATERIAL_LIB = """
// Materials of the surfaces (display only): seamless and procedural, projected triplanar. p in texture units,
// fw = size of one pixel in them (details below one pixel are faded out, otherwise it flickers). All bright/whitish.
float fbm2(vec2 p) {
    return vnoise(p) * 0.5 + vnoise(p * 2.03 + 7.1) * 0.25 + vnoise(p * 4.01 + 3.7) * 0.125 + vnoise(p * 8.07 + 1.3) * 0.0625;
}
float fbm3(vec3 p) {
    return vnoise3(p) * 0.5 + vnoise3(p * 2.03 + 7.1) * 0.25 + vnoise3(p * 4.01 + 3.7) * 0.125 + vnoise3(p * 8.07 + 1.3) * 0.0625;
}
float fade_px(float feature, float fw) { return 1.0 - smoothstep(0.35, 1.0, fw / max(feature, 1e-6)); }
// Paper (grainy watercolour paper): soft bumps, lit from top left, fine tooth
vec3 m_papier(vec2 p, float fw) {
    float s = 22.0;
    float e = 0.35 / s;
    float h0 = fbm2(p * s), hx = fbm2((p + vec2(e, 0.0)) * s), hy = fbm2((p + vec2(0.0, e)) * s);
    float relief = ((h0 - hx) + (h0 - hy)) * 2.2 * fade_px(1.0 / s, fw);
    float tooth = (vnoise(p * 140.0) - 0.5) * fade_px(1.0 / 140.0, fw);
    float t = (0.968 + relief * 0.05 + tooth * 0.015 + (vnoise(p * 1.1) - 0.5) * 0.01) * 0.91;  // slightly darker than white (artist: 70 % back from 0.7)
    return vec3(t, t * 0.997, t * 0.99);
}
// Cloth (linen): warp and weft threads in plain weave, single threads thicker or thinner (slubs)
vec3 m_stoff(vec2 p, float fw) {
    float f = 45.0;
    vec2 q = p * f;
    float k = fade_px(1.0 / f, fw);
    vec2 cell = floor(q), fr = fract(q);
    float thick_u = 0.75 + 0.5 * vnoise(vec2(cell.x * 2.3, p.y * 2.5));   // thread uneven over its length
    float thick_v = 0.75 + 0.5 * vnoise(vec2(p.x * 2.5, cell.y * 1.9 + 50.0));
    float warp = 1.0 - pow(abs(fr.x - 0.5) * 2.0, 2.0 / thick_u);          // thread centre bright, joint dark
    float weft = 1.0 - pow(abs(fr.y - 0.5) * 2.0, 2.0 / thick_v);
    float over = mod(cell.x + cell.y, 2.0);                                  // warp and weft lie on top alternately
    float thread = mix(warp * (0.85 + 0.15 * weft), weft * (0.85 + 0.15 * warp), over);
    float slub = (vnoise(vec2(cell.x * 0.7, p.y * 0.8)) + vnoise(vec2(p.x * 0.8, cell.y * 0.7 + 9.0))) * 0.5 - 0.5;
    float t = 0.968 - (1.0 - thread) * 0.06 * k + slub * 0.03 * k + (fbm2(p * 1.5) - 0.5) * 0.015;
    return vec3(t, t, t * 0.994);
}
// Finnish cardboard: bright, slightly brownish, short curly fibres, cloudy, occasional dark inclusions
vec3 m_pappe(vec2 p, float fw) {
    // Fibres: noise whose direction itself wanders slowly (no visible hatching)
    float ang = fbm2(p * 6.0) * 6.2832;
    vec2 d = vec2(cos(ang), sin(ang));
    vec2 r = vec2(dot(p, d), dot(p, vec2(-d.y, d.x)));
    float fib = (vnoise(r * vec2(150.0, 22.0)) - 0.5) * fade_px(1.0 / 150.0, fw);
    float fine = (vnoise(p * 220.0) - 0.5) * fade_px(1.0 / 220.0, fw);
    float cloud = fbm2(p * 2.5) - 0.5;
    vec2 cell = floor(p * 30.0);
    vec2 jitter = vec2(hash12(cell + 3.1), hash12(cell + 7.7)) - 0.5;
    float speck = step(0.993, hash12(cell)) * (1.0 - smoothstep(0.05, 0.16, length(fract(p * 30.0) - 0.5 - jitter * 0.6)));
    speck *= fade_px(1.0 / 30.0, fw);
    float light_fib = smoothstep(0.7, 0.9, vnoise(r.yx * vec2(120.0, 16.0) + 3.0)) * fade_px(1.0 / 120.0, fw);
    float t = 1.0 + fib * 0.13 + fine * 0.05 + light_fib * 0.035 + cloud * 0.05 - speck * 0.3;
    return vec3(0.94, 0.905, 0.85) * t;
}
// Wood: bright, fine dense grain lengthwise (slightly wavy), darker narrow lines, pores
vec3 m_holz(vec2 p, float fw) {
    float warp = fbm2(p * vec2(0.35, 0.08)) * 2.5 + fbm2(p * vec2(1.5, 0.3)) * 0.35;
    float x = p.x * 16.0 + warp * 4.0;
    float line = 1.0 - smoothstep(0.0, 0.09, abs(fract(x) - 0.5) * 2.0 - 0.0);   // narrow line per annual ring
    line *= 0.5 + 0.5 * vnoise(vec2(floor(x) * 3.1, p.y * 1.5));                // sometimes clearer, sometimes barely
    float pores = (vnoise(p * vec2(160.0, 7.0)) - 0.5) * fade_px(1.0 / 160.0, fw);
    float band = fbm2(vec2(x * 0.15, p.y * 0.1)) - 0.5;                           // wide bright/dark zones
    float t = 1.0 - line * 0.07 * fade_px(1.0 / 16.0, fw) + pores * 0.04 + band * 0.04;
    return vec3(0.955, 0.915, 0.855) * t;
}
// Spray paint, painterly: frayed paint islands and speckles with hard edges, in few tone steps (stylised,
// hardly any gradients), plus rough grain. The edges are restless like painted via a distorted grid
vec3 m_spruehfarbe(vec2 p, float fw) {
    vec2 warp = vec2(fbm2(p * 3.1), fbm2(p * 3.1 + 17.3)) - 0.5;
    vec2 q = p + warp * 0.2;
    float edge = clamp(fw * 6.0, 0.004, 0.08);                    // edge width: hard, but without stair steps
    float islands = smoothstep(0.5 - edge, 0.5 + edge, fbm2(q * 1.8));              // large rich/thin islands
    float patches = smoothstep(0.56 - edge, 0.56 + edge, fbm2(q * 6.5 + 5.0));      // medium patches
    float k1 = fade_px(1.0 / 48.0, fw), k2 = fade_px(1.0 / 110.0, fw);
    float speck = smoothstep(0.71 - edge, 0.71 + edge, vnoise(q * 48.0 + 2.0)) * k1;   // speckles
    float fine = smoothstep(0.78 - edge, 0.78 + edge, vnoise(q * 110.0 + 9.0)) * k2;   // fine speckles
    float tone = islands * 0.3 + patches * 0.3 + speck * 0.25 + fine * 0.15;
    tone = floor(tone * 4.0 + 0.5) / 4.0;                         // four tone steps: stylised instead of gradient
    float rough = (vnoise(q * vec2(150.0, 38.0)) + vnoise(q * vec2(38.0, 150.0) + 4.0) - 1.0) * k2;
    float t = 0.95 - tone * 0.17 + rough * 0.025;
    return vec3(t * 0.995, t, t * 1.005);
}

// Size of the structures per material (paper = 1): the others coarser so that all live in the same order of magnitude
const float MAT_SCALE[5] = float[5](1.0, 0.45, 0.4, 0.5, 0.35);
// Brightness per material: on average all have the same luminance as the paper (measured, artist)
const float MAT_GAIN[5] = float[5](1.0, 0.9325, 0.9664, 0.957, 0.976);  // spray paint re-measured after the tone steps
// Blotches per material: only paper, cardboard and spray paint (not cloth, not wood)
const float MAT_STAIN[5] = float[5](1.0, 0.0, 1.0, 0.0, 1.0);
vec3 material2(int m, vec2 p, float fw) {
    int i = clamp(m, 0, 4);
    float k = MAT_SCALE[i];
    p *= k;
    fw *= k;
    vec3 c;
    if (m == 1) c = m_stoff(p, fw);
    else if (m == 2) c = m_pappe(p, fw);
    else if (m == 3) c = m_holz(p, fw);
    else if (m == 4) c = m_spruehfarbe(p, fw);
    else c = m_papier(p, fw);
    return c * MAT_GAIN[i];
}
// Tone steps (artist): the structure of the materials looks stepped like the light steps, not a fine gradient -
// the brightness jumps in steps of TONE_STEP
const float TONE_STEP = 0.022;
vec3 tone_steps(vec3 c) {
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float q = (floor(l / TONE_STEP) + 0.5) * TONE_STEP;
    return c * (q / max(l, 1e-4));
}
vec3 material_triplanar(int m, vec3 q, vec3 n, float fw) {
    vec3 w = pow(abs(n), vec3(6.0));
    w /= max(w.x + w.y + w.z, 1e-6);
    vec3 c = vec3(0.0);
    if (w.x > 0.01) c += material2(m, q.zy, fw) * w.x;
    if (w.y > 0.01) c += material2(m, q.xz + 13.7, fw) * w.y;
    if (w.z > 0.01) c += material2(m, q.xy + 29.3, fw) * w.z;
    return c / max(w.x * float(w.x > 0.01) + w.y * float(w.y > 0.01) + w.z * float(w.z > 0.01), 1e-6);
}
// Blotches over all materials (like water/coffee stains): large-area, with a darker rim. 3D noise in the
// texture space of the surface: every surface its own piece (offset), no seams
vec3 stains(vec3 q, float strength) {
    if (strength <= 0.0) return vec3(1.0);
    float s = fbm3(q * 0.3);
    float body = smoothstep(0.5225, 0.6075, s);  // area share ~19 % (30 % less than the previous 28 %)
    float rim = smoothstep(0.5025, 0.5275, s) * (1.0 - smoothstep(0.5325, 0.5825, s));  // rim somewhat darker (dry edge)
    float mottled = fbm3(q * 1.3 + 31.0);
    float k = (body * (0.5 + 0.5 * mottled) * 0.12 + rim * 0.09) * strength;
    return mix(vec3(1.0), vec3(0.80, 0.70, 0.55), k);
}
"""

MATERIAL_FS = """
#version 330
// per surface, from the surface table (load_surface, first thing in main):
int material;
vec3 tint;          // tint (white = material as it is)
vec3 anchor;        // reference point of the surface: if it is moved, the texture moves along
vec3 offset;        // random offset per surface (texture units)
float tex_scale;    // world units per texture unit (global, somewhat with the surface size)
vec4 state;         // state: colour + strength (active, selected, surface mode)
float grid_step;    // grid of the active drawing surface (world units, 0 = none)
float grid_alpha;
vec3 grid_color;
float opacity;      // opacity of the surface (slider in the outliner)
vec4 owner;
float lift;         // shadow test point this far away from the surface (world units)
flat in int v_slot;
uniform float lift_min;     // ... at least this far (a few texels of the shadow map)
uniform float stain;        // strength of the blotches
uniform float layer;        // 0: hidden (X-Ray), 1: front-most layer
uniform float front_alpha;  // opacity of the front-most layer (1, X-Ray: slider)
uniform float mono;
uniform float light_on;
uniform vec3 sun_dir;
uniform vec3 sun_col;
uniform vec3 sky_col;
uniform float away_dark;
uniform float shadow_dark;
uniform float calm;         // share of the strongly smoothed normal in the light (with light steps, otherwise 0)
in vec3 v_world;
in vec3 v_normal;
in vec3 v_calm;
out vec4 f_color;
""" + SURF_DATA + NOISE + PAPER_LIB + MATERIAL_LIB + SHADOW_LIB + OCCL_VIEW + OCCL_LIB + """
void load_surface() {
    vec4 a = surf(v_slot, 0), b = surf(v_slot, 1), c = surf(v_slot, 2), d = surf(v_slot, 4), e = surf(v_slot, 5);
    tint = a.rgb; material = int(a.a + 0.5);
    anchor = b.xyz; tex_scale = b.w;
    offset = c.xyz; grid_step = c.w;
    state = surf(v_slot, 3);
    grid_color = d.rgb; grid_alpha = d.a;
    lift = max(e.x, lift_min); opacity = e.y;
    owner = vec4(e.z, -1.0, -1.0, -1.0);
}
void main() {
    load_surface();
    // Derivatives before every discard (afterwards they are undefined in the neighbouring pixels); if the normal degenerates at the edge
    // (length 0), the smoothed one applies
    vec3 vn = length(v_normal) > 1e-9 ? normalize(v_normal) : vec3(0.0, 1.0, 0.0);
    vec3 cr = cross(dFdx(v_world), dFdy(v_world));
    vec3 n_geo = length(cr) > 1e-20 ? normalize(cr) : vn;
    vec3 q = (v_world - anchor) / tex_scale + offset;
    float fw = min(length(fwidth(q)), 1e3);
    vec3 g = v_world / max(grid_step, 1e-9);
    vec3 gw = max(fwidth(g), vec3(1e-6));
    float vis = cover_edge(v_world, owner);  // behind: gone (X-Ray: soft); at the outline in front: depth test
    bool front = vis >= 1.0;
    if (front != (layer > 0.5) || vis <= 0.0) discard;
    vec3 view = ortho > 0.5 ? view_back : normalize(eye - v_world);
    vec3 c = tone_steps(material_triplanar(material, q, n_geo, fw) * stains(q, stain * MAT_STAIN[clamp(material, 0, 4)]))
             * tint;
    vec3 nrm = vn;
    if (dot(nrm, view) < 0.0) nrm = -nrm;  // the side that is seen
    float facing = abs(dot(nrm, view));
    if (light_on > 0.5) {
        // with light steps: shade the large forms, not every small wave of the net surface (otherwise camouflage pattern)
        vec3 cn = length(v_calm) > 1e-9 ? normalize(v_calm) : nrm;
        if (dot(cn, nrm) < 0.0) cn = -cn;
        vec3 ln = normalize(mix(nrm, cn, calm));
        float nl = dot(ln, sun_dir);
        float k = band(smoothstep(-0.25, 0.9, nl));
        float sh = band(in_shadow(v_world + (nl >= 0.0 ? ln : -ln) * lift + sun_dir * lift));
        float lit = min(mix(1.0 - away_dark, 1.0, k), 1.0 - shadow_dark * sh);
        c *= sun_and_sky(ln, lit, sky_col) * mix(sky_col, sun_col, k * (1.0 - sh));
    } else {
        c *= 0.9 + 0.1 * facing;  // subtle: surfaces seen at a slant somewhat darker (form stays readable)
    }
    if (grid_step > 0.0 && grid_alpha > 0.0) {  // active drawing surface: grid
        vec3 dist = abs(fract(g - 0.5) - 0.5) / gw;
        vec3 w = pow(abs(n_geo), vec3(4.0));
        w /= max(w.x + w.y + w.z, 1e-6);
        float lx = 1.0 - min(min(dist.y, dist.z), 1.0);
        float ly = 1.0 - min(min(dist.x, dist.z), 1.0);
        float lz = 1.0 - min(min(dist.x, dist.y), 1.0);
        c = mix(c, grid_color, (w.x * lx + w.y * ly + w.z * lz) * grid_alpha);
    }
    c = mix(c, state.rgb, state.a);
    vec3 lw = vec3(0.2126, 0.7152, 0.0722);
    c = mix(c, vec3(dot(c, lw)), mono);
    f_color = vec4(c, (front ? front_alpha : vis) * opacity);
}
"""


PROGRAMS = {
    'mesh': (MESH_VS, MESH_FS),
    'pick': (PICK_VS, PICK_FS),
    'wire': (WIRE_VS, WIRE_FS),
    'flat': (FLAT_VS, FLAT_FS),
    'infinite': (FLAT_VS, INFINITE_FS),
    'ribbon': (RIBBON_VS, RIBBON_FS),
    'material': (SURFACE_VS, MATERIAL_FS),
    'occl_surface': (SURFACE_VS, OCCL_SURF_FS),
    'occl_acc_surface': (SURFACE_VS, OCCL_ACC_SURF_FS),
    'shadow_surface': (SURFACE_VS, SHADOW_LINE_FS),
    'occl_mesh': (PICK_VS, OCCL_MESH_FS),
    'occl_acc_mesh': (PICK_VS, OCCL_ACC_MESH_FS),
    'block_shade': (BLOCK_VS, BLOCK_MESH_FS),
    'block_pick': (BLOCK_VS, BLOCK_PICK_FS),
    'block_occl': (BLOCK_VS, BLOCK_OCCL_FS),
    'block_occl_acc': (BLOCK_VS, BLOCK_OCCL_ACC_FS),
    'picture': (FLAT_VS, PICTURE_FS.replace('%OCCL%', OCCL_LIB)),
    'shadow_line': (SHADOW_LINE_VS, SHADOW_LINE_FS),
    'paper': (PAPER_VS, PAPER_FS),
    'cap': (CAP_VS, CAP_FS),
    'pick_gather': (PICK_GATHER_VS, PICK_GATHER_FS),
    'point_vis': (POINT_VIS_VS, POINT_VIS_FS),
}


def program(ctx, name):
    """Keep the OpenGL shader path; Metal loads the equivalent precompiled program by name."""
    if getattr(ctx, "native_metal", False):
        return ctx.program(name)
    vs, fs = PROGRAMS[name]
    return ctx.program(vertex_shader=vs, fragment_shader=fs)


_NORMALS_CACHE = {}  # id(vertices) -> (vertices, normals): drawing surfaces come cached


def _smooth_normals(verts, tris):
    """Smoothed normals per point (mean of the adjacent triangles) - curved drawing surfaces
    become evenly milkier under a flat view instead of faceted."""
    hit = _NORMALS_CACHE.get(id(verts))
    if hit is not None and hit[0] is verts:
        return hit[1]
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(verts, dtype=float)
    for k in range(3):
        np.add.at(normals, tris[:, k], face)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    lru.trim(_NORMALS_CACHE, 8192)
    _NORMALS_CACHE[id(verts)] = (verts, normals)
    return normals


def _vertex_normals(verts, tris):
    """Simply averaged normal per point (area-weighted) - fast, for surfaces currently deforming in a drag."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face = np.cross(b - a, c - a)
    flat = tris.reshape(-1)
    normals = np.empty((len(verts), 3))
    for ax in range(3):
        normals[:, ax] = np.bincount(flat, weights=np.repeat(face[:, ax], 3), minlength=len(verts))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals


from .components import SHARP_FROM_DEG as CREASE_FROM_DEG, SHARP_FULL_DEG as CREASE_FULL_DEG  # noqa: E402 – like the hardness


def _rigid_rotation(old, new, tol=1e-5):
    """Rotation R if new = s·R·old + t for ALL points (rigid motion, uniform scaling, no mirroring),
    otherwise None. Fast (one matrix fit, one check over all points) - cheaper than recomputing normals."""
    old, new = np.asarray(old, float), np.asarray(new, float)
    if old.shape != new.shape or len(old) < 3:
        return None
    co, cn = old.mean(axis=0), new.mean(axis=0)
    a, b = old - co, new - cn
    size = float(np.sqrt((a * a).sum(axis=1).max()))
    if size < 1e-12:
        return None
    u, sv, vt = np.linalg.svd(b.T @ a)
    if sv[1] <= 1e-9 * max(sv[0], 1e-30):
        return None  # points on a line: rotation about the line undetermined - recompute
    d = np.ones(3)
    d[2] = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0  # Kabsch: never mirror (flat shapes: normal unambiguous)
    rot = (u * d) @ vt
    s = float((sv * d).sum() / max((a * a).sum(), 1e-30))
    if s <= 0:
        return None
    err = np.abs(b - s * (a @ rot.T)).max()  # mirrored shapes fall through here (winding) and are recomputed
    return rot if err <= tol * size * max(s, 1.0) else None


PARALLEL_MIN = 4  # from this many new shapes the thread pool pays off
_POOL = []


def _pool():
    """Shared thread pool for independent numpy computations per shape (created once)."""
    if not _POOL:
        import os
        from concurrent.futures import ThreadPoolExecutor
        _POOL.append(ThreadPoolExecutor(max_workers=max(2, min(16, (os.cpu_count() or 4)))))
    return _POOL[0]


def _corner_normals(verts, tris):
    """Normals per triangle corner (T, 3, 3) and how strongly the corner lies on a fold (T, 3): only the
    triangles at the point that face a similar direction are averaged - fully up to 15° kink, not at all from 45°, in between
    proportionally (like hardness). A hard fold thus gets its own light on each side (one averaged normal per point
    smeared it over a cell width); a smooth surface stays smooth."""
    key = ("corner", id(verts))
    hit = _NORMALS_CACHE.get(key)
    if hit is not None and hit[0] is verts:
        return hit[1]
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face = np.cross(b - a, c - a)  # length = double the area (weight)
    unit = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-18)
    # neighborhood (which triangles at which corner) depends only on the meshing: when deforming in a drag it stays,
    # only the positions change - remembered like the smoothing matrix (_CALM_OPS)
    tkey = (len(verts), hash(np.ascontiguousarray(tris).tobytes()))
    topo = _CORNER_TOPO.get(tkey)
    if topo is None:
        flat = tris.reshape(-1)
        order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=len(verts))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        deg = int(counts.max()) if len(counts) else 0
        # per point the triangles on it (padded with -1)
        pad = -np.ones((len(verts), max(deg, 1)), int)
        rank = np.arange(len(flat)) - np.repeat(starts, counts)
        pad[flat[order], rank] = order // 3
        nb = pad[tris]  # (T, 3, D)
        valid = nb >= 0
        lru.trim(_CORNER_TOPO, 256)
        topo = _CORNER_TOPO[tkey] = (valid, np.where(valid, nb, 0))
    valid, nbc = topo
    cos = np.einsum("tkdi,ti->tkd", unit[nbc], unit)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    w = np.clip((CREASE_FULL_DEG - ang) / (CREASE_FULL_DEG - CREASE_FROM_DEG), 0.0, 1.0) * valid
    area = np.linalg.norm(face, axis=1)[nbc] * valid
    normals = np.einsum("tkd,tkdi->tki", w, face[nbc])
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-18)
    split = 1.0 - (w * area).sum(axis=2) / np.maximum(area.sum(axis=2), 1e-18)
    out = (normals, split)
    lru.trim(_NORMALS_CACHE, 8192)
    _NORMALS_CACHE[key] = (verts, out)
    return out


_CORNER_TOPO = {}  # meshing -> (valid, neighbor triangles per corner) for _corner_normals
CALM_TIME = 400.0  # smoothing of the light normals: diffusion time (in neighbor steps) - small waves go, shapes stay
CALM_FULL, CALM_NONE = 5.0, 30.0  # light steps: up to 5 steps fully smoothed, from 30 not at all (in between proportionally)


def calm_for_bands(bands):
    """Fraction of the smoothed light normals: the fewer light steps, the stronger (flowing light: none)."""
    if bands < 1.5:
        return 0.0
    return float(min(max((CALM_NONE - bands) / (CALM_NONE - CALM_FULL), 0.0), 1.0))
_CALM_OPS = {}  # meshing -> factorized smoothing matrix (when deforming only the positions change, not the meshing)


def _calm_normals(verts, tris):
    """Strongly smoothed normals per point: implicit diffusion (I + t L) x = n over the neighborhood of the points. For
    light with light steps - small waves of the net surfaces should not get step edges (blotches), large shapes
    should. Without scipy: the simply smoothed normals."""
    key = ("calm", id(verts))
    hit = _NORMALS_CACHE.get(key)
    if hit is not None and hit[0] is verts:
        return hit[1]
    normals = _smooth_normals(verts, tris)
    import scipy.sparse as sparse
    from scipy.sparse.linalg import splu
    n = len(verts)
    tkey = (n, hash(np.ascontiguousarray(tris).tobytes()))
    lu = _CALM_OPS.get(tkey)
    if lu is None:
        edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        edges = np.concatenate([edges, edges[:, ::-1]])
        a = sparse.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        a.data[:] = 1.0  # double edges count once
        deg = np.asarray(a.sum(axis=1)).ravel()
        lap = sparse.diags(deg) - a
        m = (sparse.identity(n) + CALM_TIME * sparse.diags(1.0 / np.maximum(deg, 1.0)) @ lap).tocsc()
        lru.trim(_CALM_OPS, 128)  # (before 64 slots and everything gone on overflow; discard only the older half)
        lu = _CALM_OPS[tkey] = splu(m)
    out = lu.solve(np.ascontiguousarray(normals, dtype=float))
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    _NORMALS_CACHE[key] = (verts, out)
    return out


def _stroke_arrays(curves):
    """All strokes with at least two points strung together: (points, widths, rgba per point,
    point count per stroke). One pass in numpy instead of a loop per stroke (large selections)."""
    curves = [c for c in curves if len(c[0]) >= 2]
    if not curves:
        return None
    counts = np.array([len(c[0]) for c in curves])
    pts = np.concatenate([np.asarray(c[0], dtype="f4") for c in curves])
    widths = np.concatenate([_fit(np.asarray(c[1], dtype="f4"), len(c[0])) for c in curves])
    colors = np.repeat(np.asarray([c[2] for c in curves], dtype="f4"), counts, axis=0)
    styles = np.repeat(np.asarray([c[3] if len(c) > 3 else 1.0 for c in curves], dtype="f4"), counts)
    owners = np.repeat(np.asarray([_owner4(c[4] if len(c) > 4 else ()) for c in curves], dtype="f4"), counts, axis=0)
    # painting strokes: (normals per point, order 0..1); otherwise normal 0
    normals = np.concatenate([np.asarray(c[5][0], dtype="f4") if len(c) > 5 and c[5] is not None
                              else np.zeros((len(c[0]), 3), "f4") for c in curves])
    layers = np.repeat(np.asarray([c[5][1] if len(c) > 5 and c[5] is not None else 0.0 for c in curves], dtype="f4"),
                       counts)
    return pts, widths, colors, counts, styles, owners, normals, layers


def _fit(values, n):
    """Safety net: widths matching the point count (otherwise drawing aborts - colorful boxes in the viewport)."""
    if len(values) == n:
        return values
    if not len(values):
        return np.full(n, 0.02, "f4")
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values).astype("f4")


def _owner4(ids):
    """Up to four surface ids, padded with -1 (membership of a stroke for the coverage map)."""
    ids = [float(i) for i in ids][:4]
    return ids + [-1.0] * (4 - len(ids))


def stroke_buffers(curves):
    """(vertex data, indices, ends) of the strokes - ribbons and ends from ONE processing of the entries."""
    arrays = _stroke_arrays(curves)
    if arrays is None:
        return None, None, None
    return (*ribbon_vertices(curves, arrays), cap_vertices(curves, arrays))


def ribbon_vertices(curves, arrays=None):
    """[(points (N,3), widths (N,), rgba)] -> (vertex data, indices) for camera ribbons.
    Two vertices per point (left/right), two triangles per segment."""
    arrays = _stroke_arrays(curves) if arrays is None else arrays
    if arrays is None:
        return None, None
    pts, widths, colors, counts, styles, owners, normals, layers = arrays
    last = np.cumsum(counts) - 1
    first = last - counts + 1
    # arc length per stroke in stroke widths: coordinate of the pencil grain along the stroke
    # sum in double precision: the sum runs over ALL strokes - in single precision the grain of a
    # stroke depended on how many precede it in the buffer (and the partial upload of single strokes would deviate)
    seg = np.linalg.norm(np.diff(pts.astype("f8"), axis=0), axis=1)
    seg[last[:-1]] = 0.0  # no jump from the end of one stroke to the next
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    arc -= np.repeat(arc[first], counts)
    arc /= np.maximum(np.repeat(np.add.reduceat(widths, first) / counts, counts), 1e-6)
    prev = np.roll(pts, 1, axis=0)
    prev[first] = pts[first]
    nxt = np.roll(pts, -1, axis=0)
    nxt[last] = pts[last]
    n = len(pts)
    per_point = np.hstack([pts, prev, nxt, widths[:, None]])
    side = (np.tile(np.array([-1.0, 1.0], dtype="f4"), n) * np.repeat(styles, 2))[:, None]  # brush: ±2
    verts = np.hstack([np.repeat(per_point, 2, axis=0), side, np.repeat(arc[:, None], 2, axis=0),
                       np.repeat(colors, 2, axis=0), np.repeat(owners, 2, axis=0), np.repeat(normals, 2, axis=0),
                       np.repeat(layers[:, None], 2, axis=0)])
    seg = np.ones(n, bool)
    seg[last] = False  # no segment from the end of one stroke to the start of the next
    base = 2 * np.flatnonzero(seg)
    idx = np.stack([base, base + 1, base + 2, base + 1, base + 3, base + 2], axis=1).reshape(-1)
    return np.ascontiguousarray(verts, dtype="f4"), idx.astype("i4")


def _stroke_same(a, b):
    """Two stroke entries with the same content? Arrays by identity (they are never changed in place; the
    remembered entry holds them, so an id cannot have been reassigned), the rest by value."""
    if a is b:
        return True
    if a[0] is not b[0] or a[1] is not b[1] or len(a) != len(b) or tuple(a[2]) != tuple(b[2]):
        return False
    if len(a) > 3 and a[3] != b[3]:
        return False
    if len(a) > 4 and list(a[4]) != list(b[4]):
        return False
    if len(a) > 5:
        if a[5] is None or b[5] is None:
            return a[5] is b[5]
        return a[5][0] is b[5][0] and a[5][1] == b[5][1]
    return True


def _stroke_diff(old, new):
    """Indices of the entries that have changed - or None if the layout is different (different number of
    strokes, different point count, painting/line swapped): then everything must be reprocessed."""
    if len(old) != len(new):
        return None
    changed = []
    for k, (a, b) in enumerate(zip(old, new)):
        if _stroke_same(a, b):
            continue
        if len(a[0]) != len(b[0]) or len(a[1]) != len(b[1]) or \
                (len(a) > 5 and a[5] is not None) != (len(b) > 5 and b[5] is not None):
            return None
        changed.append(k)
    return changed


def cap_vertices(curves, arrays=None):
    """End points of all strokes as (position, width, rgba) for the round ends."""
    arrays = _stroke_arrays(curves) if arrays is None else arrays
    if arrays is None:
        return None
    pts, widths, colors, counts, styles, owners, normals, layers = arrays
    last = np.cumsum(counts) - 1
    ends = np.stack([last - counts + 1, last], axis=1).reshape(-1)
    return np.ascontiguousarray(np.hstack([pts[ends], widths[ends, None], colors[ends], owners[ends],
                                           styles[ends, None], normals[ends], layers[ends, None]]), dtype="f4")


def _mat_bytes(m):
    return np.asarray(m, dtype="f4").T.tobytes()  # GLSL expects column-major


CULL_PX = 1.0  # meshes whose bounding box on screen is smaller are not drawn (overview shots of large scenes)


def view_matrix(cam):
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = cam.right, cam.up, cam.back
    m[:3, 3] = -m[:3, :3] @ cam.eye
    return m


def projection_matrix(fov, aspect, near, far):
    f = 1.0 / math.tan(fov / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ])


def ortho_matrix(half_height, aspect, depth):
    """Orthographic, visible from -depth to +depth around the camera (also "behind" it)."""
    return np.array([
        [1 / (half_height * aspect), 0, 0, 0],
        [0, 1 / half_height, 0, 0],
        [0, 0, -1 / depth, 0],
        [0, 0, 0, 1],
    ])


# vertex/index data of the meshes are shared by all viewports (the GL contexts share their buffers,
# see AA_ShareOpenGLContexts); only the vertex arrays belong to each context itself.
_SHARED_MESH_DATA = {}  # id(Geometry) -> (vbo, cbo, ibo, geo)


def _mesh_data(ctx, geo):
    data = _SHARED_MESH_DATA.get(id(geo))
    if data is None or data[3] is not geo:
        data = _SHARED_MESH_DATA[id(geo)] = (ctx.buffer(geo.vertices.tobytes()), ctx.buffer(geo.colors.tobytes()),
                                             ctx.buffer(geo.faces.tobytes()), geo)
    return data


def _release_mesh_data(alive):
    for key in [k for k in _SHARED_MESH_DATA if k not in alive]:
        for res in _SHARED_MESH_DATA.pop(key)[:3]:
            res.release()


def shadow_ribbons(lines, sun, min_w, r):
    """Triangles of the lines for the shadow map: per segment a ribbon across the sun at the stroke width (min. min_w),
    extended by half the width at both ends (no gaps at kinks) - thick strokes cast thick shadows.
    The ribbon lies half the width behind the line (away from the sun, like the back of a tube): this way
    a thick stroke does not shadow itself. lines: [(points, widths)] - per point or one number."""
    # all lines in one go (before, one by one per line: harbor scene 12.11., 9,000 strokes, ~0.5 s per change with light)
    sun = np.asarray(sun, float)
    lines = [(np.asarray(pts, float), w) for pts, w in lines if len(pts) >= 2]
    if not lines:
        return np.zeros((0, 3), "f4")
    counts = np.array([len(pts) for pts, _ in lines])
    pts = np.concatenate([pts for pts, _ in lines])
    hw = np.concatenate([np.broadcast_to(np.asarray(w, float), (len(p),)) for p, w in lines])
    hw = np.maximum(hw, min_w) * 0.5
    pts = pts - sun * hw[:, None]
    seg = np.ones(len(pts) - 1, bool)
    seg[np.cumsum(counts)[:-1] - 1] = False  # no segment between two lines
    i = np.flatnonzero(seg)
    a, b, ha, hb = pts[i], pts[i + 1], hw[i, None], hw[i + 1, None]
    t = b - a
    lt = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(lt, 1e-12)
    a, b = a - t * ha, b + t * hb
    side = np.cross(t, sun)
    ln = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(ln > 1e-9, side / np.maximum(ln, 1e-12), r)
    sa, sb = side * ha, side * hb
    q = [a - sa, a + sa, b + sb, b - sb]
    return np.stack([q[0], q[1], q[2], q[0], q[2], q[3]], axis=1).reshape(-1, 3).astype("f4")

class _MeshBuffers:
    def __init__(self, ctx, geo, shade_prog, pick_prog, occl_prog, occl_acc_prog):
        # moderngl checks context membership: bind the shared GL buffers here only as foreign buffers
        vbo, cbo, ibo = (b if b.ctx is ctx else ctx.external_buffer(b.glo, b.size) for b in _mesh_data(ctx, geo)[:3])
        attrs = [(vbo, "3f", "in_pos"), (cbo, "3f", "in_color")]
        self.shade = ctx.vertex_array(shade_prog, attrs, index_buffer=ibo, index_element_size=4)
        self.pick = ctx.vertex_array(pick_prog, [(vbo, "3f", "in_pos")], index_buffer=ibo, index_element_size=4)
        self.occl = ctx.vertex_array(occl_prog, [(vbo, "3f", "in_pos")], index_buffer=ibo, index_element_size=4)
        self.occl_acc = ctx.vertex_array(occl_acc_prog, [(vbo, "3f", "in_pos")], index_buffer=ibo, index_element_size=4)
        self._ctx, self._vbo, self._wire, self._edge_buf = ctx, vbo, None, None

    def wire(self, prog, geo):
        """Edges for the wireframe - created only on first switch-on."""
        if self._wire is None:
            self._edge_buf = self._ctx.buffer(geo.edges.tobytes())
            self._wire = self._ctx.vertex_array(prog, [(self._vbo, "3f", "in_pos")], index_buffer=self._edge_buf,
                                                index_element_size=4)
        return self._wire

    def release(self):
        for res in (self.shade, self.pick, self.occl, self.occl_acc, self._wire, self._edge_buf):
            if res is not None:
                res.release()


class _MeshBlock:
    """All (at build time) imported meshes in one buffer: points in world coordinates, per point the number
    (slot) of its mesh. An entry (block, None) in the renderer's mesh lists stands for all parts in it."""

    def __init__(self, ctx, meshes, progs):
        self.meshes = list(meshes)
        self.slot = {id(m): k for k, m in enumerate(self.meshes)}
        self.baked = [(m.geo, m.xform.tobytes()) for m in self.meshes]
        nv = np.array([len(m.geo.vertices) for m in self.meshes], np.int64)
        nt = np.array([m.geo.triangle_count for m in self.meshes], np.int64)
        voff, toff = np.concatenate([[0], np.cumsum(nv)]), np.concatenate([[0], np.cumsum(nt)])
        pos = np.empty((voff[-1], 3), "f4")
        col = np.empty((voff[-1], 3), "f4")
        idx = np.empty((toff[-1], 3), "u4")
        for k, m in enumerate(self.meshes):
            a = np.asarray(m.xform, float)
            pos[voff[k]:voff[k + 1]] = m.geo.vertices @ a[:3, :3].T.astype("f4") + a[:3, 3].astype("f4")
            col[voff[k]:voff[k + 1]] = m.geo.colors
            idx[toff[k]:toff[k + 1]] = m.geo.faces + np.uint32(voff[k])
        slot = np.repeat(np.arange(len(self.meshes), dtype="f4"), nv)
        self.lo, self.hi = pos.min(axis=0).astype(float), pos.max(axis=0).astype(float)
        self.triangle_count = int(toff[-1])
        self.geo = self  # (the mesh lists ask geo.triangle_count)
        self.xform = np.eye(4)
        self.vis_rev = 0  # counts changes, what is drawn (shadow map)
        self._bufs = [ctx.buffer(pos.tobytes()), ctx.buffer(col.tobytes()), ctx.buffer(slot.tobytes()),
                      ctx.buffer(idx.tobytes())]
        self.vaos = {}
        for name, prog in progs.items():
            attrs = [(self._bufs[0], "3f", "in_pos")]
            if "in_color" in prog:
                attrs.append((self._bufs[1], "3f", "in_color"))
            attrs.append((self._bufs[2], "1f", "in_slot"))
            self.vaos[name] = ctx.vertex_array(prog, attrs, index_buffer=self._bufs[3], index_element_size=4)
        n = len(self.meshes)
        self.table = np.zeros((max(-(-n * 3 // 1024), 1) * 1024, 4), "f4")
        rows = self.table[:n * 3].reshape(n, 3, 4)
        rows[:, 1, 0] = [float(m.id) for m in self.meshes]
        rows[:, 2, 1], rows[:, 2, 2] = toff[:-1] // 65536, toff[:-1] % 65536
        self.tex = ctx.texture((1024, len(self.table) // 1024), 4, dtype="f4")
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._sent = None

    def rows(self):
        return self.table[:len(self.meshes) * 3].reshape(len(self.meshes), 3, 4)

    def current(self, m):
        """Slot of m if m sits unchanged in the block (same geometry and matrix), otherwise None."""
        k = self.slot.get(id(m))
        if k is None or self.meshes[k] is not m:
            return None
        geo, xf = self.baked[k]
        return k if geo is m.geo and xf == m.xform.tobytes() else None

    def world_bounds(self):
        return self.lo, self.hi

    def bind(self, prog, vis_col):
        if self._sent is None or not np.array_equal(self._sent, self.table):
            self.tex.write(self.table.tobytes())
            self._sent = self.table.copy()
        self.tex.use(location=BLOCK_UNIT)
        prog["block_tab"].value = BLOCK_UNIT
        prog["vis_col"].value = vis_col

    def release(self):
        for res in list(self.vaos.values()) + self._bufs + [self.tex]:
            res.release()
        self.vaos, self._bufs = {}, []


class Renderer:
    MSAA_SAMPLES = 8

    def __init__(self, ctx, gl):
        """gl: QOpenGLFunctions of the Qt context, for states moderngl does not manage
        (glDepthMask, GL_MULTISAMPLE)."""
        self.ctx = ctx
        self._gl = gl
        self._depth_mask = gl.glDepthMask
        self._msaa_fbo, self._msaa_size = None, None
        self.mesh_prog = program(ctx, 'mesh')
        self.pick_prog = program(ctx, 'pick')
        self.wire_prog = program(ctx, 'wire')
        self.flat_prog = program(ctx, 'flat')
        self.inf_prog = program(ctx, 'infinite')
        self._infinite = []  # infinite planes of this frame (base grid, active plane), see viewport
        self.ribbon_prog = program(ctx, 'ribbon')
        self.material_prog = program(ctx, 'material')
        self.occl_surf_prog = program(ctx, 'occl_surface')
        self.occl_acc_surf_prog = program(ctx, 'occl_acc_surface')
        self.shadow_surf_prog = program(ctx, 'shadow_surface')
        self._surface_vaos = {}  # program -> vao on the surface buffer
        self._surfaces = []  # surfaces of this frame (dicts, see render)
        self.material_scale = 1.0  # global material scale (viewport)
        # hold the shadow map fixed (viewport: while a gizmo drag runs). Re-rendering it costs per frame with the
        # scene (all strokes, all surfaces); during a drag the frame rate counts, the shadows follow on release.
        self.shadow_hold = False
        self.occl_mesh_prog = program(ctx, 'occl_mesh')
        self.occl_acc_mesh_prog = program(ctx, 'occl_acc_mesh')
        self.block_progs = {name: program(ctx, "block_" + name)
                            for name in ("shade", "pick", "occl", "occl_acc")}
        self._block = None  # _MeshBlock of the imported meshes (or None)
        self._block_memo = None  # (input list, list with block entry) - same list when rotating
        self._occl = None  # (fbo, coverage texture, size, depth buffer)
        self._occl_on = False
        self._xray_front = 1.0  # opacity of the frontmost layer (X-Ray < 1)
        self._xray_behind = 0.0  # opacity of hidden things (0 = invisible)
        self._xray_on = False
        self._aid_pass = 1.0  # drawing aids: 0 below the surfaces, 1 above them
        self.pictures = []  # projected images of this frame (pictures.draw_list)
        self._picture_textures = {}
        self.picture_prog = program(ctx, 'picture')
        self._occl_dense_on = True  # check the opacity of the frontmost layer (off while it is being created)
        self.shadow_line_prog = program(ctx, 'shadow_line')
        self.light = None  # light mode: dict(dir, col, sky, away, shadow) or None
        self._shadow = None  # (fbo, depth texture, size)
        self._shadow_key = None
        self._shadow_vp = np.eye(4)
        self._shadow_bias = 0.002
        self._shadow_lines = None  # (vbo, vao, number of vertices, key)
        self._stroke_rev = 0  # counts every re-upload of the strokes
        # ... and only those where position or width of a stroke changed: the shadow map depends on that. A
        # pure color change (hover, selection) casts no different shadow and does not re-render it.
        self._stroke_geo_rev = 0
        self.paper_prog = program(ctx, 'paper')
        self._paper_vao = ctx.vertex_array(self.paper_prog, [])
        self.cap_prog = program(ctx, 'cap')
        self._caps = None  # (vbo, vao, capacity)
        self._surface_buf = None  # (vbo, vao, capacity)
        self._slot_buf = None  # (vbo, capacity, corner counts per surface): number of the surface per corner
        self._surf_tex = None  # surface table (SURF_DATA)
        self._surf_table = None  # (surface list, scale, values, ...) of the last uploaded table
        self._stroke_state = None  # last uploaded strokes: entries, vertices, ends (partial upload, _patch_strokes)
        self._surface_ranges = []  # (first vertex, count) per drawing surface of this frame
        self._line_buf = None  # (vbo, vao, capacity) for outlines of the drawing surfaces
        self._ribbon = None  # (vbo, ibo, vao, capacity vbo, capacity ibo)
        self._live = (None, None, (0, 0))  # live strokes (running stroke): (ribbon, caps, counts) as above, small
        # flipbook drawings (Timing) of all frames: own buffer, rebuilt only when drawings change; per frame
        # only the ranges of the current drawings are drawn (a frame change uploads nothing)
        self._flip_buf = (None, None)  # (ribbon, caps)
        self._flip_key = None
        self._flip_ranges = {}  # (plane, frame) -> (first index, indices, first end, ends)
        self._flip_jobs = []  # [(range, opacity)] of this frame
        self._meshes = {}  # id(Geometry) -> _MeshBuffers
        self._flat_buf, self._flat_vao, self._flat_cap = None, None, 0
        self._pick_fbo, self._pick_size, self._pick_key = None, None, None
        self._pick_meshes = []

    # --- Resources ---
    def _mesh_buffers(self, geo):
        buf = self._meshes.get(id(geo))
        if buf is None:
            buf = self._meshes[id(geo)] = _MeshBuffers(self.ctx, geo, self.mesh_prog, self.pick_prog, self.occl_mesh_prog,
                                                          self.occl_acc_mesh_prog)
            buf.geo = geo  # keeps the id stable as long as the buffer lives
        return buf

    def release_unused(self, geos):
        alive = {id(g) for g in geos}
        for key in [k for k in self._meshes if k not in alive]:
            self._meshes.pop(key).release()
        _release_mesh_data(alive)

    def _flat_vertices(self, data):
        data = np.ascontiguousarray(data, dtype="f4")
        if data.nbytes > self._flat_cap:
            if self._flat_buf is not None:
                self._flat_vao.release()
                self._flat_buf.release()
            self._flat_cap = max(data.nbytes, 2 * self._flat_cap, 4096)
            self._flat_buf = self.ctx.buffer(reserve=self._flat_cap, dynamic=True)
            self._flat_vao = self.ctx.vertex_array(self.flat_prog, [(self._flat_buf, "3f", "in_pos")])
            self._inf_vao = self.ctx.vertex_array(self.inf_prog, [(self._flat_buf, "3f", "in_pos")])
        self._flat_buf.write(data.tobytes())

    # --- Visible image ---
    def render(self, fbo, cam, proj, focal, light_eye, meshes, planes, strokes, opacity, min_px=1.0, surfaces=(),
               view_tan=0.414, infinite=(), xray=None, wire=None, live=(), flip=None):
        """meshes: [(MeshObject, tint_rgba)], planes: [(Plane, fill_rgba, line_rgba, highlighted)]
        (sorted back to front), strokes: [(points, widths, rgba, side, surface ids)], opacity: opacity
        of the meshes, xray: None or opacity of the front layer in X-Ray (hidden: XRAY_BEHIND of it; meshes
        show each of their layers), wire: None or opacity of the wireframe (edges of the meshes added, only in
        X-Ray - then all; opaque meshes without X-Ray stay without edges).
        proj: projection matrix, focal: pixel scale of the strokes (see RIBBON_VS), light_eye: light source of the
        mesh shading.

        surfaces: surfaces with material (dicts, see _upload_surfaces).
        live: strokes like strokes that change with every mouse movement (the running stroke) - own small
        buffer so that strokes need not be reprocessed each time while drawing.
        flip: flipbook drawings (Timing) as (entries like painting strokes, fingerprint, {group: (first entry,
        count)}, [(group, opacity)]) - own buffer, only the named groups are drawn.

        What is in front is decided by the coverage map (surfaces + meshes, _update_occlusion) - not by the order
        in which things were drawn. Order in the image: meshes -> drawing aids behind -> surfaces opaque with material
        (only the frontmost layer) -> drawing aids in front (hide nothing) -> painting -> strokes on top (on
        their surface always, behind it not)."""
        ctx = self.ctx
        w, h = fbo.size
        target = self._msaa_target((w, h))  # Qt provides no MSAA -> own multisample buffer
        target.use()
        ctx.viewport = (0, 0, w, h)
        ctx.scissor = None
        self._gl.glEnable(GL_MULTISAMPLE)  # Qt's painter turns it off, moderngl does not turn it back on
        self._depth_mask(True)
        target.clear(*theme.rgb(theme.PAPER), 1.0, depth=1.0)
        vp = proj @ view_matrix(cam)
        self._view = (cam.eye, cam.back, getattr(cam, "ortho", False))
        self._frame_vp, self._frame_size = vp, (w, h)  # for points_visible after the frame
        self._draw_paper((w, h), cam, view_tan)

        self._upload_surfaces(surfaces)
        meshes = self._blockify(meshes, build=not self.shadow_hold)  # imported meshes: one block instead of one per mesh
        self._update_shadow(strokes, meshes)  # light mode: shadow map (only on changes)
        meshes = self._in_view(vp, meshes, (w, h))  # everything after this sees only this camera: do not draw meshes outside
        self._xray_front = 1.0 if xray is None else float(xray)
        self._xray_behind = 0.0 if xray is None else float(xray) * XRAY_BEHIND
        self._xray_on = xray is not None
        mesh_alpha = opacity * self._xray_front
        # semi-transparent meshes (slider) cover only in X-Ray - otherwise one sees through them
        occluding = mesh_alpha >= 0.999 or xray is not None
        self._update_occlusion(vp, (w, h), meshes if occluding else (), focal, self._surfaces)
        self._frame_meshes = meshes  # for id_map after the frame
        target.use()
        ctx.viewport = (0, 0, w, h)
        self._upload_strokes(strokes)
        self._upload_live(live)
        self._upload_flip(flip)
        self._infinite = list(infinite or ())

        # 1. Meshes - among themselves via depth buffer, semi-transparent only with their frontmost layer; X-Ray: every
        # layer see-through on top of each other (denser where more layers lie). Wireframe is added only in X-Ray
        # (overlay like in Blender, all edges); opaque meshes without X-Ray stay without edges
        wires = meshes and wire is not None and wire > 0.001
        if meshes and mesh_alpha <= 0.001 and wires:  # meshes fully see-through (slider): only the edges
            ctx.enable_only(moderngl.BLEND)
            ctx.blend_func = BLEND_OVER
            self._draw_wires(vp, meshes, wire)
        elif meshes and mesh_alpha > 0.001:
            ctx.enable_only(moderngl.DEPTH_TEST)
            self._depth_mask(True)
            if xray is not None and mesh_alpha < 0.999:
                ctx.enable_only(moderngl.BLEND)
                ctx.blend_func = BLEND_OVER
                # per layer so much that a closed body (front + back) looks like the front layer
                # with the opacity of the slider
                self._draw_meshes(vp, light_eye, meshes, alpha=1.0 - math.sqrt(max(1.0 - mesh_alpha, 0.0)))
            elif mesh_alpha >= 0.999:
                self._draw_meshes(vp, light_eye, meshes, alpha=1.0)
            else:
                target.color_mask = (False, False, False, False)
                target.use()  # moderngl applies the mask only on activation
                self._draw_meshes(vp, light_eye, meshes, alpha=1.0)
                target.color_mask = (True, True, True, True)
                target.use()
                ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)
                ctx.blend_func = BLEND_OVER
                ctx.depth_func = "<="
                self._draw_meshes(vp, light_eye, meshes, alpha=mesh_alpha)
                ctx.depth_func = "<"
            if wires and xray is not None:
                xray_wire = xray is not None and mesh_alpha < 0.999  # X-Ray: no depth buffer, all edges
                ctx.enable_only(moderngl.BLEND if xray_wire else moderngl.BLEND | moderngl.DEPTH_TEST)
                ctx.blend_func = BLEND_OVER
                self._depth_mask(False)
                ctx.depth_func = "<="
                self._draw_wires(vp, meshes, wire, bias=0.0 if xray_wire else WIRE_DEPTH_BIAS)
                ctx.depth_func = "<"
                self._depth_mask(True)
            self._gl.glClear(GL_DEPTH_BUFFER_BIT)  # what lies in front of/behind the meshes is decided from here on by the coverage map
        # 2. Drawing aids lying on or behind a surface: below - it covers them
        ctx.enable_only(moderngl.BLEND)
        ctx.blend_func = BLEND_OVER
        self._aid_pass = 0.0
        self._glaze_planes = [(np.asarray(pl.origin, float), np.asarray(pl.axis_u, float) * pl.size_u / 2,
                               np.asarray(pl.axis_v, float) * pl.size_v / 2, tuple(float(x) for x in fill),
                               (self._plane_under(em),), (0.0 if int(em) else 1.0,))
                              for pl, fill, _line, em in planes if len(fill) > 3 and fill[3] > 0.01]
        self._draw_planes(vp, planes)
        self._draw_pictures(vp)
        # 3. Surfaces opaque with their material (only the frontmost layer; X-Ray: hidden things softly)
        self._draw_materials(vp)
        # 4. Drawing aids in front - hide nothing; outlines of the surfaces
        self._aid_pass = 1.0
        self._draw_planes(vp, planes)
        self._draw_pictures(vp)
        self._draw_lines(vp, [(o, s["line"], s["emphasis"], s["id"]) for s in self._surfaces for o in s["outlines"]])
        # 5. Painting: in painting order (depth = order, the newest lies on top), then the strokes on top
        # (depth only among themselves: round ends not blended twice)
        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = BLEND_OVER
        self._depth_mask(True)
        if getattr(self, "_paint_counts", (0, 0))[0]:
            self._gl.glClear(GL_DEPTH_BUFFER_BIT)
            self._draw_strokes(vp, (w, h), focal, min_px, part="paint")
            self._gl.glClear(GL_DEPTH_BUFFER_BIT)
        if self._flip_jobs:  # flipbook: per plane the current drawing (onion skin faint in front)
            ribbon, capbuf = self._flip_buf
            for (f_idx, n_idx, f_cap, n_cap), fade in self._flip_jobs:
                self._gl.glClear(GL_DEPTH_BUFFER_BIT)
                self._draw_strokes(vp, (w, h), focal, min_px, fade=fade,
                                   only=[(ribbon, capbuf, f_idx, n_idx, f_cap, n_cap)])
            self._gl.glClear(GL_DEPTH_BUFFER_BIT)
        self._draw_strokes(vp, (w, h), focal, min_px)
        ctx.enable_only(moderngl.NOTHING)
        ctx.copy_framebuffer(fbo, target)  # resolve multisample
        fbo.use()

    def _draw_paper(self, size, cam, view_tan):
        """Drawing paper as background: blotches, fibers, grain on a sphere around the camera
        (rotates with the view like a sky), plus slightly darker corners."""
        ctx = self.ctx
        ctx.enable_only(moderngl.NOTHING)
        prog = self.paper_prog
        prog["paper"].value = theme.rgb(theme.PAPER)
        prog["size"].value = (float(size[0]), float(size[1]))
        prog["stain"].value = theme.PAPER_STAIN
        prog["grain"].value = theme.PAPER_GRAIN
        prog["vignette"].value = theme.PAPER_VIGNETTE
        prog["cam_right"].value = tuple(float(x) for x in cam.right)
        prog["cam_up"].value = tuple(float(x) for x in cam.up)
        prog["cam_back"].value = tuple(float(x) for x in cam.back)
        prog["view_tan"].value = float(view_tan)
        self._paper_vao.render(moderngl.TRIANGLES, vertices=3)

    def _occl_uniforms(self, prog, owner=None, under=None):
        """Coverage map to a program: owner = surface id (drawing aids), under = opacity on a surface."""
        if "occl_on" not in prog:
            return
        prog["occl_on"].value = 1.0 if self._occl_on else 0.0
        if "xray_behind" in prog:  # not in the opacity pass (front_here)
            prog["xray_behind"].value = self._xray_behind
        prog["occl_tol"].value = OCCL_TOL
        if "aid_pass" in prog:
            prog["aid_pass"].value = self._aid_pass
        if "occl_min" in prog:
            prog["occl_min"].value = OCCL_MIN_COVER if self._occl_dense_on else 0.0
        eye, back, ortho = self._view
        for name, value in (("eye", tuple(float(x) for x in eye)), ("view_back", tuple(float(x) for x in back)),
                            ("ortho", 1.0 if ortho else 0.0)):
            if name in prog:
                prog[name].value = value
        if self._occl_on:
            self._occl[1].use(location=OCCL_UNIT)
            prog["occl_map"].value = OCCL_UNIT
            if "occl_acc" in prog:
                # while the opacity is being created, its texture is the target - then bind a different one (not read)
                (self._occl[4] if self._occl_dense_on else self._occl[1]).use(location=OCCL_ACC_UNIT)
                prog["occl_acc"].value = OCCL_ACC_UNIT
        self._occl_owner(prog, owner, under)

    @staticmethod
    def _occl_owner(prog, owner=None, under=None):
        """Per drawing aid: its id and the opacity on a surface (the rest stays set by the pass)."""
        if owner is not None and "owner" in prog:
            prog["owner"].value = (float(owner), -1.0, -1.0, -1.0)
        if under is not None and "under" in prog:
            prog["under"].value = float(under)

    def _surface_vao(self, prog):
        """Vertex array of a program on the shared surface buffer (position, normal)."""
        vbo, slots = self._surface_buf[0], self._slot_buf[0]
        cached = self._surface_vaos.get(id(prog))
        if cached is None or cached[0] is not vbo or cached[2] is not slots:
            slot = [(slots, "1f", "in_slot")]
            if "in_calm" in prog:
                vao = self.ctx.vertex_array(prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_calm")] + slot)
            elif "in_normal" in prog:
                vao = self.ctx.vertex_array(prog, [(vbo, "3f 3f 12x", "in_pos", "in_normal")] + slot)
            else:
                vao = self.ctx.vertex_array(prog, [(vbo, "3f 24x", "in_pos")] + slot)
            cached = self._surface_vaos[id(prog)] = (vbo, vao, slots)
        if "surf_data" in prog:
            self._surf_tex.use(location=SURF_UNIT)
            prog["surf_data"].value = SURF_UNIT
            prog["surf_rows"].value = SURF_ROWS
        if "occluders_only" in prog:
            prog["occluders_only"].value = 0
        return cached[1]

    def _surface_rows(self, surfaces):
        """Values of the surface table (N, SURF_COLS, 4) - see SURF_DATA."""
        rows = np.zeros((len(surfaces), SURF_COLS, 4), "f4")
        for k, s in enumerate(surfaces):
            size = max(s["size"], 1e-6)
            # grain: 90 % global, 10 % by surface size (large surfaces somewhat coarser)
            scale = self.material_scale * (size / theme.MATERIAL_SIZE_REF) ** theme.MATERIAL_SIZE_WEIGHT
            seed = float(s["seed"])
            opacity = float(s.get("opacity", 1.0))
            m = s.get("model")
            r = rows[k]
            r[0, :3], r[0, 3] = s["tint"], int(s["material"])
            r[1, :3], r[1, 3] = s["anchor"], scale
            r[2, :3], r[2, 3] = (seed * 7.31 % 97.0, seed * 3.17 % 89.0, seed * 5.53 % 83.0), s["grid_step"]
            r[3] = s["state"]
            r[4, :3], r[4, 3] = s["grid_color"], s["grid_alpha"]
            # (shadow check point: in the shader at least a few texels of the shadow map away from the surface, lift_min)
            r[5] = (size * 0.004, opacity, float(s["id"]), 0.0 if m is None else 1.0)
            r[6, :3] = (max(s["size"] * SURFACE_OCCL_RADIUS / min(s.get("resolution", 1.0), 1.0), 1e-6),
                        float(-np.log(1.0 - min(max(opacity, 0.0), 0.9999))),
                        1.0 if s["count"] and s.get("occlude", True) else 0.0)
            if m is not None:
                r[7:10] = np.asarray(m, "f4")[:3]
        return rows

    def _upload_surface_table(self, source):
        """Surface table and surface numbers per corner for this frame. source: the list the viewport passed -
        the same list as in the last frame (camera move) means the same values."""
        surfaces = self._surfaces
        counts = [s["count"] for s in surfaces]
        if self._slot_buf is None or self._slot_buf[2] != counts:
            data = np.repeat(np.arange(len(counts), dtype="f4"), counts)
            if self._slot_buf is None or data.nbytes > self._slot_buf[1]:
                if self._slot_buf is not None:
                    for _vbo, vao, _slots in self._surface_vaos.values():  # hang on the old buffer
                        vao.release()
                    self._surface_vaos.clear()
                    self._slot_buf[0].release()
                cap = max(data.nbytes * 2, 1 << 14)
                self._slot_buf = (self.ctx.buffer(reserve=cap, dynamic=True), cap, counts)
            else:
                self._slot_buf = (self._slot_buf[0], self._slot_buf[1], counts)
            if data.nbytes:
                self._slot_buf[0].write(data)
        extra = (theme.MATERIAL_SIZE_REF, theme.MATERIAL_SIZE_WEIGHT, SURF_ROWS)
        last = self._surf_table
        same = last is not None and last[0] is source and last[1] == self.material_scale and last[3] == extra
        if same and not self.VERIFY_STROKES:
            return
        rows = self._surface_rows(surfaces)
        if same:  # test check mode: the same list must yield the same values
            assert rows.shape == last[2].shape and np.array_equal(rows, last[2]), \
                "surface table: same list as in the last frame, but different values"
            return
        n = len(surfaces)
        blocks = (n + SURF_ROWS - 1) // SURF_ROWS
        height = SURF_ROWS if blocks > 1 else max(64, 1 << int(np.ceil(np.log2(max(n, 1)))))
        size = (SURF_COLS * blocks, height)
        if self._surf_tex is None or self._surf_tex.size != size:
            if self._surf_tex is not None:
                self._surf_tex.release()
            self._surf_tex = self.ctx.texture(size, 4, dtype="f4")
            self._surf_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        grid = np.zeros((height, SURF_COLS * blocks, 4), "f4")
        for b in range(blocks):
            part = rows[b * SURF_ROWS:(b + 1) * SURF_ROWS]
            grid[:len(part), b * SURF_COLS:(b + 1) * SURF_COLS] = part
        self._surf_tex.write(grid.tobytes())
        self._surf_table = (source, self.material_scale, rows, extra)

    def _upload_surfaces(self, surfaces):
        """Upload the surfaces of this frame (only if one has changed: same arrays = same shape).
        surfaces: [dict(verts, tris, id, material, tint, seed, state, grid_step, grid_alpha, grid_color, outlines, line,
        emphasis)]."""
        self._surfaces = []
        if not surfaces:
            self._surface_key = None
            return
        self._upload_surface_blocks(surfaces)
        self._upload_surface_table(surfaces)

    def _upload_surface_blocks(self, surfaces):
        # draft (surface currently deforming in a gizmo drag): simple normals - belongs to the state, so that after
        # release the same shape is uploaded once more with exact normals
        key = tuple((id(s["verts"]), id(s["tris"]), bool(s.get("draft"))) for s in surfaces)
        first = 0
        chunks = [] if key != getattr(self, "_surface_key", None) or self._surface_buf is None else None
        bounds = getattr(self, "_bounds_cache", {})
        self._bounds_cache = {}
        # per shape (arrays): normals per corner and finished block for uploading - unchanged ones not again. Only shift around
        # when actually uploading (otherwise they would be empty after a frame without changes, and the next change
        # would recompute all surfaces)
        normals = getattr(self, "_normal_cache", {})
        blocks = getattr(self, "_block_cache", {})
        if chunks is not None:
            counters.C["flaechen_upload"] += len(surfaces)
            self._normal_cache, self._block_cache = {}, {}
            self._prefetch_normals(surfaces, normals, blocks)
        for s in surfaces:
            verts, tris = s["verts"], s["tris"]
            n = tris.size
            hit = bounds.get(id(verts))
            if hit is None or hit[0] is not verts:  # compute circumference only for new shapes (otherwise every surface per frame)
                hit = (verts, verts.min(axis=0), verts.max(axis=0))
            self._bounds_cache[id(verts)] = hit
            lo, hi = hit[1], hit[2]
            if s.get("model") is not None:  # carried along: box of the moved net
                m = s["model"]
                box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                box = box @ m[:3, :3].T + m[:3, 3]
                lo, hi = box.min(axis=0), box.max(axis=0)
            self._surfaces.append(dict(s, first=first, count=n, lo=lo, hi=hi, anchor=(lo + hi) / 2,
                                       size=float(np.linalg.norm(hi - lo))))
            if chunks is not None:
                bk = (id(verts), id(tris))
                draft = bool(s.get("draft"))
                hit = blocks.get(bk) or self._block_cache.get(bk)
                if hit is not None and hit[0] is verts and hit[1] is tris and (draft or not hit[3]):
                    self._block_cache[bk] = hit
                    chunks.append(hit[2])
                    first += n
                    continue
                flat = tris.reshape(-1)
                basis = s.get("basis")  # copy of a generator (only offset): normals of the original, rotated along
                if basis is not None:
                    ov, ot, m3, flip = basis
                    corner, calm, rough = self._corner_data(ov, ot, normals, draft=draft)
                    corner, calm = corner @ m3.T, calm @ m3.T
                    if flip:  # mirrored: winding reversed (the corners of each triangle in new order)
                        corner = corner.reshape(-1, 3, 3)[:, [0, 2, 1]].reshape(-1, 3)
                        calm = calm.reshape(-1, 3, 3)[:, [0, 2, 1]].reshape(-1, 3)
                else:
                    corner, calm, rough = self._corner_data(verts, tris, normals, sid=s.get("id"), draft=draft)
                block = np.empty((len(flat), 9), "f4")
                block[:, :3], block[:, 3:6], block[:, 6:] = verts[flat], corner, calm
                self._block_cache[bk] = (verts, tris, block, rough)
                chunks.append(block)
            first += n
        if chunks is None:
            return
        self._surface_keep = [(s["verts"], s["tris"]) for s in surfaces]  # keeps the ids valid
        old = getattr(self, "_surface_blocks", None)
        self._surface_blocks = chunks
        if (self._surface_buf is not None and old is not None and len(old) == len(chunks)
                and all(a.shape == b.shape for a, b in zip(old, chunks))):
            # same layout: write only the changed surfaces into their place (instead of copying everything anew)
            offset = 0
            for a, b in zip(old, chunks):
                if a is not b:
                    self._surface_buf[0].write(b, offset=offset)
                offset += b.nbytes
            self._surface_key = key
            return
        data = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if self._surface_buf is None or data.nbytes > self._surface_buf[2]:
            if self._surface_buf is not None:
                self._surface_buf[1].release()
                self._surface_buf[0].release()
            cap = max(data.nbytes * 2, 1 << 16)
            vbo = self.ctx.buffer(reserve=cap, dynamic=True)
            vao = self.ctx.vertex_array(self.material_prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_calm")])
            self._surface_buf = (vbo, vao, cap)
        self._surface_buf[0].write(data)
        self._surface_key = key

    def _prefetch_normals(self, surfaces, previous, blocks):
        """Normals of all shapes that would have to be recomputed shortly, in advance and in parallel (independent per shape, numpy releases
        the Python lock on large computations) - e.g. after extruding many surfaces: hundreds of new shapes.
        Result lands in previous; _corner_data finds it there. Rotated copies (rigid) stay when rotating along."""
        jobs = []
        by_sid = self.__dict__.get("_normal_by_sid", {})
        for s in surfaces:
            verts, tris = s["verts"], s["tris"]
            if s.get("basis") is not None or s.get("draft"):
                continue
            bk = (id(verts), id(tris))
            hit = blocks.get(bk)
            if hit is not None and hit[0] is verts and hit[1] is tris and not hit[3]:
                continue
            hit = previous.get(bk) or self._normal_cache.get(bk)
            if hit is not None and hit[0] is verts and hit[1] is tris and not hit[4]:
                continue
            old = by_sid.get(s.get("id"))
            if old is not None and old[1].shape == tris.shape and _rigid_rotation(old[0], verts) is not None:
                continue
            jobs.append((verts, tris))
        if len(jobs) < PARALLEL_MIN:
            return
        def work(job):
            verts, tris = job
            flat = tris.reshape(-1)
            corner, split = _corner_normals(verts, tris)
            corner, split = corner.reshape(-1, 3), split.reshape(-1, 1)
            calm = _calm_normals(verts, tris)[flat] * (1 - split) + corner * split
            return (verts, tris, corner, calm, False)
        for hit in _pool().map(work, jobs):
            previous[(id(hit[0]), id(hit[1]))] = hit

    def _corner_data(self, verts, tris, previous, sid=None, draft=False):
        """(corner normals, smoothed normals, only approximate?) per triangle corner (N*3, 3) of a shape - taken over from previous
        (last upload) if the same arrays, otherwise computed. Light normals per corner (folds stay
        hard); the strongly smoothed ones (light steps) yield to the corner normal at a fold, otherwise the
        smoothing would blur the fold. draft (surface deforming in a drag): only the simply averaged normals per point - folds
        and the smoothing for light steps come after release (exactly computed ones are also used during the drag, approximate ones
        never for the finished image)."""
        k = (id(verts), id(tris))
        hit = previous.get(k) or self._normal_cache.get(k)
        if hit is not None and hit[4] and not draft:
            hit = None  # only remembered approximately: now exactly
        if hit is None or hit[0] is not verts or hit[1] is not tris:
            by_sid = self.__dict__.setdefault("_normal_by_sid", {})
            old = by_sid.get(sid) if sid is not None else None
            rot = _rigid_rotation(old[0], verts) if old is not None and (
                old[1] is tris or old[1].shape == tris.shape and np.array_equal(old[1], tris)) else None
            if rot is not None and (draft or not old[4]):
                # same shape, only rotated/shifted/uniformly scaled (gizmo drag of a group): rotate normals along
                hit = (verts, tris, old[2] @ rot.T, old[3] @ rot.T, old[4])
            elif draft:
                rough = _vertex_normals(verts, tris)[tris.reshape(-1)]
                hit = (verts, tris, rough, rough, True)
            else:
                flat = tris.reshape(-1)
                corner, split = _corner_normals(verts, tris)
                corner, split = corner.reshape(-1, 3), split.reshape(-1, 1)
                calm = _calm_normals(verts, tris)[flat] * (1 - split) + corner * split
                hit = (verts, tris, corner, calm, False)
        if sid is not None:
            self.__dict__.setdefault("_normal_by_sid", {})[sid] = hit
        self._normal_cache[k] = hit
        return hit[2], hit[3], hit[4]

    def _draw_materials(self, vp):
        """Surfaces opaque with their material. What is in front is told by the coverage map; X-Ray: first the hidden things softly,
        above them the frontmost layer with the opacity of the slider."""
        if not self._surfaces:
            return
        prog = self.material_prog
        prog["vp"].write(_mat_bytes(vp))
        prog["bias"].value = 0.0
        prog["stain"].value = float(theme.STAIN_STRENGTH)
        prog["mono"].value = 0.0 if theme.COLOR_VIEW else 1.0
        prog["front_alpha"].value = self._xray_front
        self._light_uniforms(prog, surface=True)
        bands = float(self.light.get("bands", 0)) if self.light else 0.0
        prog["calm"].value = calm_for_bands(bands)
        self._occl_uniforms(prog)
        prog["lift_min"].value = 3.0 * getattr(self, "_shadow_texel", 0.0)
        vao = self._surface_vao(prog)
        total = sum(s["count"] for s in self._surfaces)
        # first hidden things (shine through see-through surfaces or in X-Ray), above them the frontmost layer - the one with
        # depth test: where two surfaces both count as front at the outline, it decides per sample (smooth edges)
        for layer in (0.0, 1.0):
            prog["layer"].value = layer
            if layer > 0.5:
                self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)
                self._depth_mask(True)
                self._gl.glClear(GL_DEPTH_BUFFER_BIT)
            vao.render(moderngl.TRIANGLES, vertices=total)  # all surfaces at once (surface table)
            counters.C["zeichenaufrufe"] += 1
        self._gl.glClear(GL_DEPTH_BUFFER_BIT)  # the strokes afterwards bring their own depth
        self.ctx.enable_only(moderngl.BLEND)

    def _draw_lines(self, vp, lines):
        """Thin lines (outlines of the drawing surfaces): [(polyline (N,3), rgba, highlighted, surface id)]."""
        lines = [l for l in lines if len(l[0]) >= 2]
        if not lines:
            return
        counts = [2 * (len(pts) - 1) for pts, *_ in lines]
        key = tuple((id(l[0]), len(l[0])) for l in lines)
        if key != getattr(self, "_line_key", None) or self._line_buf is None:  # upload only on changes
            segs = [np.stack([pts[:-1], pts[1:]], axis=1).reshape(-1, 3) for pts, *_ in lines]
            data = np.ascontiguousarray(np.vstack(segs), dtype="f4")
            if self._line_buf is None or data.nbytes > self._line_buf[2]:
                if self._line_buf is not None:
                    self._line_buf[1].release()
                    self._line_buf[0].release()
                cap = max(data.nbytes * 2, 1 << 15)
                vbo = self.ctx.buffer(reserve=cap, dynamic=True)
                vao = self.ctx.vertex_array(self.flat_prog, [(vbo, "3f", "in_pos")])
                self._line_buf = (vbo, vao, cap)
            self._line_buf[0].write(data.tobytes())
            self._line_key = key
            self._line_keep = [l[0] for l in lines]  # keeps the ids valid
        vbo, vao = self._line_buf[:2]
        prog = self.flat_prog
        prog["vp"].write(_mat_bytes(vp))
        prog["bias"].value = 0.0
        prog["grain"].value = 0.0
        self._occl_uniforms(prog, under=-1.0)
        first = 0
        for (pts, rgba, highlighted, sid), n in zip(lines, counts):
            self._occl_owner(prog, owner=sid)
            _, border_a, width = theme.PLANE_EMPHASIS[int(highlighted)]
            prog["color"].value = (*rgba[:3], border_a)
            self.ctx.line_width = width
            vao.render(moderngl.LINES, vertices=n, first=first)
            first += n
        self.ctx.line_width = 1.0

    def _upload_strokes(self, strokes):
        """Upload ribbon and end-point data; drawn twice if needed. Unchanged strokes
        (same fingerprint as in the last frame, e.g. when rotating the camera) are not
        reprocessed - with thousands of lines the biggest part of the frame time."""
        if strokes is getattr(self, "_stroke_list", None) and self._ribbon is not None:
            return  # same list as in the last frame (viewport kept it, e.g. when rotating): nothing to do
        self._stroke_list = strokes
        ents = [s for s in strokes if len(s[0])]
        old = self._stroke_state
        if old is not None and (self._ribbon is not None or not ents):
            changed = _stroke_diff(old["ents"], ents)
            if changed is not None and not changed:
                old["ents"] = ents  # same content: no new state (empty stays empty)
                return
            if changed is not None and self._ribbon is not None and self._caps is not None:
                # same layout (number of strokes, points per stroke, painting in front): reprocess only the changed strokes
                # and write them to their place in the buffer - when dragging, usually a small part changes,
                # but the processing costs per stroke (harbor scene: 24 ms per frame for 1,300 strokes)
                self._stroke_rev += 1
                olds = old["ents"]
                if any(olds[i][0] is not ents[i][0] or olds[i][1] is not ents[i][1] for i in changed):
                    self._stroke_geo_rev += 1
                counters.C["strich_upload"] += len(changed)
                self._patch_strokes(old, ents, changed)
                if self.VERIFY_STROKES:
                    self._verify_strokes(old)
                return
        self._stroke_rev += 1
        self._stroke_geo_rev += 1
        counters.C["strich_upload"] += len(ents)
        verts, idx, caps = stroke_buffers(ents)
        self._stroke_counts = (0 if idx is None else len(idx), 0 if caps is None else len(caps))
        # painting strokes are at the front of the list: their share of indices and ends (own pass)
        lead = []
        for s in ents:
            if len(s) > 5 and s[5] is not None:
                if len(s[0]) >= 2:
                    lead.append(len(s[0]))
            else:
                break
        self._paint_counts = (sum(6 * (n - 1) for n in lead), 2 * len(lead))
        # remember state (holds the arrays of the entries - comparison by identity stays valid)
        self._stroke_state = {"ents": ents, "verts": verts, "caps": caps}
        if verts is None:
            self._stroke_counts = (0, 0)
            return
        self._ribbon, self._caps = self._fill_stroke_buffers(verts, idx, caps, self._ribbon, self._caps)

    VERIFY_STROKES = False  # tests: after every partial upload check against the full processing

    def _patch_strokes(self, state, ents, changed):
        """Reprocess only the strokes changed (indices in ents): write their vertices and ends into the remembered arrays
        and - in contiguous ranges - into the GPU buffers. The layout is the same as at the last full
        upload (_stroke_diff), so indices and all positions remain correct."""
        verts, caps = state["verts"], state["caps"]
        if "at" not in state:  # position of each stroke: first vertex (2 per point), first end (2 per stroke with >= 2 points)
            n = np.array([len(e[0]) if len(e[0]) >= 2 else 0 for e in ents], np.int64)
            state["at"] = np.concatenate([[0], np.cumsum(2 * n)])
            state["cap_at"] = np.concatenate([[0], np.cumsum(2 * (n > 0))])
        at, cap_at = state["at"], state["cap_at"]
        state["ents"] = ents
        todo = [k for k in changed if len(ents[k][0]) >= 2]
        if not todo:
            return
        part = [ents[k] for k in todo]
        new_v, _, new_c = stroke_buffers(part)
        v_at = c_at = 0
        spans = []  # (first vertex, after the last) of the changed ranges
        for k in todo:
            a, b = int(at[k]), int(at[k + 1])
            verts[a:b] = new_v[v_at:v_at + (b - a)]
            v_at += b - a
            ca = int(cap_at[k])
            caps[ca:ca + 2] = new_c[c_at:c_at + 2]
            c_at += 2
            if spans and a - spans[-1][1] <= 4096:  # small gap: upload along with it instead of another write
                spans[-1][1] = b
            else:
                spans.append([a, b])
        row = verts.shape[1] * 4
        if sum(b - a for a, b in spans) * 2 > len(verts):
            self._ribbon[0].write(verts.tobytes())
        else:
            for a, b in spans:
                self._ribbon[0].write(verts[a:b].tobytes(), offset=a * row)
        self._caps[0].write(caps.tobytes())

    def _verify_strokes(self, state):
        verts, _ = ribbon_vertices(state["ents"])
        caps = cap_vertices(state["ents"])
        same = lambda x, y: x.shape == y.shape and np.allclose(x, y, rtol=1e-5, atol=1e-5)  # arc length: rounding
        assert same(verts, state["verts"]), "partial stroke upload differs"
        assert same(caps, state["caps"]), "stroke ends in the partial upload differ"
        gpu = np.frombuffer(self._ribbon[0].read(size=verts.nbytes), "f4").reshape(verts.shape)
        assert np.array_equal(gpu, state["verts"]), "GPU buffer of the strokes differs"
        gpu = np.frombuffer(self._caps[0].read(size=caps.nbytes), "f4").reshape(caps.shape)
        assert np.array_equal(gpu, state["caps"]), "GPU buffer of the stroke ends differs"

    def _fill_stroke_buffers(self, verts, idx, caps, ribbon, capbuf):
        """Write ribbon and end-point data into (growing) GPU buffers -> (ribbon, caps)."""
        ctx = self.ctx
        if ribbon is None or verts.nbytes > ribbon[3] or idx.nbytes > ribbon[4]:
            if ribbon is not None:
                for res in ribbon[:3]:
                    res.release()
            vcap, icap = max(verts.nbytes * 2, 1 << 16), max(idx.nbytes * 2, 1 << 15)
            vbo = ctx.buffer(reserve=vcap, dynamic=True)
            ibo = ctx.buffer(reserve=icap, dynamic=True)
            vao = ctx.vertex_array(self.ribbon_prog, [
                (vbo, "3f 3f 3f 1f 1f 1f 4f 4f 3f 1f", "in_pos", "in_prev", "in_next", "in_width", "in_side", "in_arc",
                 "in_color", "in_owner", "in_normal", "in_layer")],
                index_buffer=ibo, index_element_size=4)
            ribbon = (vbo, ibo, vao, vcap, icap)
        ribbon[0].write(verts.tobytes())
        ribbon[1].write(idx.tobytes())
        if capbuf is None or caps.nbytes > capbuf[2]:
            if capbuf is not None:
                capbuf[1].release()
                capbuf[0].release()
            cap = max(caps.nbytes * 2, 1 << 14)
            vbo = ctx.buffer(reserve=cap, dynamic=True)
            vao = ctx.vertex_array(self.cap_prog, [(vbo, "3f 1f 4f 4f 1f 3f 1f", "in_pos", "in_width", "in_color",
                                                    "in_owner", "in_style", "in_normal", "in_layer")])
            capbuf = (vbo, vao, cap)
        capbuf[0].write(caps.tobytes())
        return ribbon, capbuf

    def _upload_live(self, live):
        """Live strokes (running stroke, previews) into their own small buffer - new every frame, they are short."""
        ribbon, capbuf, _ = self._live
        live = [s for s in live if len(s[0])]
        verts, idx = ribbon_vertices(live) if live else (None, None)
        if verts is None:
            self._live = (ribbon, capbuf, (0, 0))
            return
        caps = cap_vertices(live)
        ribbon, capbuf = self._fill_stroke_buffers(verts, idx, caps, ribbon, capbuf)
        self._live = (ribbon, capbuf, (len(idx), len(caps)))

    def _upload_flip(self, flip):
        """Flipbook buffer: reprocessed only when the fingerprint of the drawings changes (it holds the arrays
        via the entries - ids stay valid). Per frame only the ranges of the named groups."""
        if not flip:
            self._flip_jobs = []
            return
        entries, key, groups, jobs = flip
        if key != self._flip_key or self._flip_buf[0] is None:
            verts, idx = ribbon_vertices(entries)
            if verts is None:
                self._flip_jobs, self._flip_key = [], key
                return
            caps = cap_vertices(entries)
            self._flip_buf = self._fill_stroke_buffers(verts, idx, caps, *self._flip_buf)
            # per entry (all with >= 2 points): 6 (n-1) indices and 2 ends
            n_idx = np.array([6 * (len(e[0]) - 1) for e in entries], np.int64)
            start = np.concatenate([[0], np.cumsum(n_idx)])
            self._flip_ranges = {g: (int(start[i0]), int(start[i0 + n] - start[i0]), 2 * i0, 2 * n)
                                 for g, (i0, n) in groups.items()}
            self._flip_key, self._flip_keep = key, entries
        self._flip_jobs = [(self._flip_ranges[g], fade) for g, fade in jobs if g in self._flip_ranges]

    @staticmethod
    def _plane_under(highlighted):
        """Glaze of a plane over a surface it lies on. Every visible plane glazes (artist 12.11.: an
        uninvolved plane was invisible over the geometry and was overlooked). The difference lies IN FRONT of a
        surface: drawing plane (2) and selected plane (1) cover there with their full fill, an uninvolved one (0)
        also there only with the glaze (soft / plane_s)."""
        return theme.ACTIVE_GLAZE

    def _plane_glaze_uniforms(self, prog):
        """Visible helper planes (render: planes) for plane_glaze in the stroke shader: center, half edges, fill color."""
        if "plane_n" not in prog:
            return
        planes = getattr(self, "_glaze_planes", [])[:16]
        prog["plane_n"].value = len(planes)
        if not planes:
            return
        pad = lambda rows, k: np.asarray(list(rows) + [rows[-1]] * (16 - len(rows)), "f4").reshape(16, k)
        for name, k, idx in (("plane_o", 3, 0), ("plane_u", 3, 1), ("plane_v", 3, 2), ("plane_c", 4, 3),
                             ("plane_k", 1, 4), ("plane_s", 1, 5)):
            if name in prog:
                prog[name].write(pad([p[idx] for p in planes], k).tobytes())

    def _draw_strokes(self, vp, size, focal, min_px, fade=1.0, part="lines", only=None):
        """Strokes as camera ribbons plus round ends (point sprites). Where the ribbon already lies,
        the end point loses the depth test - so nothing is blended twice.
        part: "paint" = only the painting strokes (at the front of the buffer), "lines" = the rest. only: exactly these ranges
        [(ribbon, caps, first index, indices, first end, ends)] (flipbook)."""
        if only is not None:
            jobs = [j for j in only if j[0] is not None and j[3]]
        else:
            n_all, n_all_caps = getattr(self, "_stroke_counts", (0, 0))
            p_idx, p_caps = getattr(self, "_paint_counts", (0, 0))
            if part == "paint":
                first_idx, n_idx, first_cap, n_caps = 0, p_idx, 0, p_caps
            else:
                first_idx, n_idx, first_cap, n_caps = p_idx, n_all - p_idx, p_caps, n_all_caps - p_caps
            jobs = [(self._ribbon, self._caps, first_idx, n_idx, first_cap, n_caps)] if n_idx and self._ribbon else []
            if part == "lines" and self._live[2][0]:  # the running stroke: on top, from its own buffer
                jobs.append((self._live[0], self._live[1], 0, self._live[2][0], 0, self._live[2][1]))
        if not jobs:
            return
        self._light_uniforms(self.ribbon_prog)
        self._light_uniforms(self.cap_prog)  # ends: same shadow as the adjoining line piece
        for prog in (self.ribbon_prog, self.cap_prog):
            self._occl_uniforms(prog)
            prog["vp"].write(_mat_bytes(vp))
            prog["half_vp"].value = (size[0] / 2, size[1] / 2)
            prog["focal"].value = focal
            prog["min_px"].value = min_px
            prog["bias"].value = STROKE_BIAS
            prog["grain"].value = PENCIL_GRAIN * theme.STROKE_GRAIN
            prog["fade"].value = fade
            self._plane_glaze_uniforms(prog)
        for ribbon, capbuf, f_idx, c_idx, f_cap, c_cap in jobs:
            ribbon[2].render(moderngl.TRIANGLES, vertices=c_idx, first=f_idx)
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            # Qt gives a compatibility context: there point sprites (gl_PointCoord) need this switch, otherwise
            # the shader discards everything and the strokes end square
            self._gl.glEnable(GL_POINT_SPRITE)
            capbuf[1].render(moderngl.POINTS, vertices=c_cap, first=f_cap)
            self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)

    def _light_uniforms(self, prog, surface=False):
        """Pass light mode and view direction to a program (lines, stroke ends or surfaces) - only what
        the program also uses."""
        def put(name, value):
            if name in prog:
                prog[name].value = value
        eye, back, ortho = self._view
        put("eye", tuple(float(x) for x in eye))
        put("view_back", tuple(float(x) for x in back))
        put("ortho", 1.0 if ortho else 0.0)
        light = self.light
        put("light_on", 1.0 if light else 0.0)
        put("sun_dir", tuple(float(x) for x in (light["dir"] if light else (0.0, 1.0, 0.0))))
        put("sun_col", tuple(float(x) for x in (light["col"] if light else (1.0, 1.0, 1.0))))
        put("sky_col", tuple(float(x) for x in (light["sky"] if light else (1.0, 1.0, 1.0))))
        if surface:
            put("away_dark", float(light["away"]) if light else 0.0)
        shadow = bool(light and light.get("shadow") and self._shadow is not None)
        put("shadow_on", 1.0 if shadow else 0.0)
        put("shadow_dark", float(light["shadow"]) if shadow else 0.0)
        if "shadow_vp" in prog:
            prog["shadow_vp"].write(_mat_bytes(self._shadow_vp))
        put("shadow_bias", self._shadow_bias)
        put("dome", float(light.get("dome", 0.0)) if light else 0.0)
        put("bands", float(light.get("bands", 0)) if light else 0.0)
        put("edge_gain", LIGHT_EDGE_GAIN)
        put("shadow_lift", getattr(self, "_shadow_lift", 0.02))
        if shadow and "shadow_map" in prog:
            self._shadow[1].use(location=SHADOW_UNIT)
            prog["shadow_map"].value = SHADOW_UNIT

    def _blockify(self, meshes, build=True):
        """Mesh list [(mesh, tint)] -> the same list with one entry (block, None) instead of the imported meshes
        that sit unchanged in the block (tint and visibility into its table). New or moved meshes
        stay individual until there are BLOCK_REBUILD of them (then rebuild, except build=False: gizmo drag)."""
        memo = self._block_memo
        if memo is not None and memo[0] is meshes and memo[2] is self._block:
            return memo[1]
        from .scene import MeshObject
        eligible = [(m, t) for m, t in meshes if type(m) is MeshObject and not m.missing and m.geo.triangle_count]
        block = self._block
        if len(eligible) < BLOCK_MIN:
            if block is not None:
                block.release()
                self._block = None
            self._block_memo = (meshes, meshes, None)
            return meshes
        if build:
            outside = sum(1 for m, _ in eligible if block is None or block.current(m) is None)
            stale = block is not None and len(block.meshes) - (len(eligible) - outside) > max(len(block.meshes) // 2, BLOCK_REBUILD)
            if block is None or outside >= BLOCK_REBUILD or stale:
                if block is not None:
                    block.release()
                t0 = time.perf_counter()
                block = self._block = _MeshBlock(self.ctx, [m for m, _ in eligible], self.block_progs)
                counters.C["block_neu"] += 1
                self.block_seconds = time.perf_counter() - t0
        if block is None:
            self._block_memo = (meshes, meshes, None)
            return meshes
        rows = block.rows()
        shown_before = rows[:, 1, 1].copy()
        rows[:, 1, 1] = 0.0
        out = [(block, None)]
        for m, t in meshes:
            k = block.current(m) if type(m) is MeshObject else None
            if k is None:
                out.append((m, t))
            else:
                rows[k, 0] = t
                rows[k, 1, 1] = 1.0
        if not np.array_equal(shown_before, rows[:, 1, 1]):
            block.vis_rev += 1
        self._block_memo = (meshes, out, block)
        return out

    def _in_view(self, vp, meshes, size=None):
        """Only the meshes whose bounding box touches the field of view and is at least CULL_PX large on screen
        (large USD hierarchies: thousands of meshes, per frame several passes with one draw call per mesh). The corners in world coordinates stay remembered as long as
        the same list comes (rotating, zooming: the viewport lists are not rebuilt)."""
        if len(meshes) < 64:
            return meshes
        memo = getattr(self, "_cull_memo", None)
        if memo is None or memo[0] is not meshes:
            corners = np.empty((len(meshes), 8, 4))
            for k, (m, _) in enumerate(meshes):
                lo, hi = (m.lo, m.hi) if isinstance(m, _MeshBlock) else m.geo.bbox
                c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                corners[k] = c @ np.asarray(m.xform, float).T
            memo = self._cull_memo = (meshes, corners)
        clip = memo[1] @ np.asarray(vp, float).T  # (N, 8, 4)
        w = clip[..., 3]
        out = np.zeros(len(meshes), bool)
        for axis in range(3):
            out |= np.all(clip[..., axis] > w, axis=1) | np.all(clip[..., axis] < -w, axis=1)
        if size is not None:  # tiny: entirely in front of the camera and smaller than CULL_PX (extent of the bounding box)
            front = np.all(w > 1e-9, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ndc = clip[..., :2] / np.where(front[:, None], w, 1.0)[..., None]
            ext = (ndc.max(axis=1) - ndc.min(axis=1)) * (np.array(size, float) / 2)
            out |= front & (ext.max(axis=1) < CULL_PX)
        out &= np.array([not isinstance(m, _MeshBlock) for m, _ in meshes])  # the block: the GPU discards by itself
        if not out.any():
            return meshes
        return [e for e, o in zip(meshes, out) if not o]

    def _update_occlusion(self, vp, size, meshes=(), focal=1.0, surfaces=()):
        """Coverage map: per pixel the frontmost covering layer - surfaces and meshes - with distance to the camera
        (world units), |cos| between view and normal, radius (tolerance) and surface id. Surfaces, drawing aids
        and strokes compare themselves against it (cover() in OCCL_LIB): what belongs to the frontmost surface or lies in front of it
        is drawn, everything clearly behind it is not (X-Ray: softly) - independent of order and zoom level."""
        self._occl_on = False
        meshes = [(m, t) for m, t in meshes if m.geo.triangle_count]
        surfaces = [s for s in surfaces if s["count"] and s.get("occlude", True)]  # helper surfaces hide nothing
        if not meshes and not surfaces:
            return
        ctx = self.ctx
        if self._occl is None or self._occl[2] != size:
            if self._occl is not None:
                for res in self._occl[:2] + self._occl[3:]:
                    res.release()
            tex = ctx.texture(size, 4, dtype="f4")
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            depth = ctx.depth_renderbuffer(size)
            acc = ctx.texture(size, 1, dtype="f4")
            acc.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._occl = (ctx.framebuffer(color_attachments=[tex], depth_attachment=depth), tex, size, depth, acc,
                          ctx.framebuffer(color_attachments=[acc]))
        fbo = self._occl[0]
        fbo.use()
        ctx.viewport = (0, 0, size[0], size[1])
        fbo.clear(1e30, 1.0, 0.0, -1.0, depth=1.0)
        ctx.enable_only(moderngl.DEPTH_TEST)
        self._depth_mask(True)
        ctx.depth_func = "<"
        eye, back, ortho = self._view
        view = (tuple(float(x) for x in eye), tuple(float(x) for x in back), 1.0 if ortho else 0.0)
        if meshes:
            prog = self.occl_mesh_prog
            prog["eye"].value, prog["view_back"].value, prog["ortho"].value = view
            for mesh, _ in meshes:
                if isinstance(mesh, _MeshBlock):
                    self._draw_block(mesh, "occl", vp)
                    continue
                prog["mvp"].write(_mat_bytes(vp @ mesh.xform))
                prog["model"].write(_mat_bytes(mesh.xform))
                # tolerance only in pixels (OCCL_MIN_PX in the shader), not by mesh size: otherwise with a
                # large mesh (a whole airplane) a small sketch behind it would shine through far
                prog["radius"].value = MESH_OCCL_RADIUS
                prog["mesh_id"].value = float(mesh.id)
                self._mesh_buffers(mesh.geo).occl.render(moderngl.TRIANGLES)
        if surfaces:  # surfaces cover like meshes (lines on them belong to them: tolerance via their id)
            prog = self.occl_surf_prog
            prog["vp"].write(_mat_bytes(vp))
            prog["bias"].value = 0.0
            prog["eye"].value, prog["view_back"].value, prog["ortho"].value = view
            vao = self._surface_vao(prog)
            prog["occluders_only"].value = 1  # (helper surfaces drop out in the shader; radius and id from the table)
            vao.render(moderngl.TRIANGLES, vertices=sum(s["count"] for s in self._surfaces))
            counters.C["zeichenaufrufe"] += 1
        self._occl_on = True
        # opacity of the frontmost layer per pixel: many thin surfels on top of each other cover together
        acc_fbo = self._occl[5]
        acc_fbo.use()
        acc_fbo.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable_only(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._occl_dense_on = False
        if meshes:
            prog = self.occl_acc_mesh_prog
            self._occl_uniforms(prog)  # once per pass, per mesh only its id (hundreds with USD hierarchies)
            prog["acc_value"].value = 20.0
            for mesh, _ in meshes:
                if isinstance(mesh, _MeshBlock):
                    self._draw_block(mesh, "occl_acc", vp)
                    continue
                self._occl_owner(prog, owner=mesh.id)
                prog["mvp"].write(_mat_bytes(vp @ mesh.xform))
                prog["model"].write(_mat_bytes(mesh.xform))
                self._mesh_buffers(mesh.geo).occl_acc.render(moderngl.TRIANGLES)
        if surfaces:
            prog = self.occl_acc_surf_prog
            prog["vp"].write(_mat_bytes(vp))
            prog["bias"].value = 0.0
            vao = self._surface_vao(prog)
            self._occl_uniforms(prog)
            prog["occluders_only"].value = 1
            vao.render(moderngl.TRIANGLES, vertices=sum(s["count"] for s in self._surfaces))
            counters.C["zeichenaufrufe"] += 1
        self._occl_dense_on = True

    def _update_shadow(self, strokes, meshes):
        """Re-render the sun's shadow map if sun, strokes, surfaces or meshes have changed:
        depth from the sun's view (orthographic over the whole scene)."""
        light = self.light
        if not light or not light.get("shadow"):
            return
        if self.shadow_hold and self._shadow is not None:
            return  # in a gizmo drag: the shadows from the start of the drag stay, exact again after release
        mesh_key = tuple((m.geo, m.xform.tobytes(), getattr(m, "vis_rev", 0)) for m, _ in meshes)  # identity (update_pick)
        key = (tuple(np.round(light["dir"], 5)), self._stroke_geo_rev, mesh_key, self._surface_key)
        if key == self._shadow_key and self._shadow is not None:
            return
        self._shadow_key = key
        ctx = self.ctx
        # extent of the scene
        parts = [np.asarray(s[0], float) for s in strokes if len(s[0])]
        for m, _ in meshes:
            lo, hi = m.world_bounds()
            parts.append(np.array([lo, hi]))
        for s in self._surfaces:
            parts.append(np.array([s["lo"], s["hi"]]))
        if not parts:
            return
        allp = np.vstack(parts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        center, radius = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1e-3) * 1.05
        d = np.asarray(light["dir"], float)
        d /= np.linalg.norm(d)
        eye = center + d * radius * 2
        up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.95 else np.array([0.0, 0.0, 1.0])
        f = -d
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = r, u, -f
        view[:3, 3] = -view[:3, :3] @ eye
        near, far = radius * 0.5, radius * 3.5
        proj = np.eye(4)
        proj[0, 0] = proj[1, 1] = 1.0 / radius
        proj[2, 2] = -2.0 / (far - near)
        proj[2, 3] = -(far + near) / (far - near)
        self._shadow_vp = proj @ view
        self._shadow_bias = SHADOW_BIAS
        self._shadow_lift = radius * 0.004  # lines: check this far toward the sun
        self._shadow_texel = 2.0 * radius / SHADOW_SIZE  # world size of a texel of the shadow map
        if self._shadow is None:
            tex = ctx.depth_texture((SHADOW_SIZE, SHADOW_SIZE))
            tex.compare_func = ""
            tex.repeat_x = tex.repeat_y = False
            self._shadow = (ctx.framebuffer(depth_attachment=tex), tex, SHADOW_SIZE)
        fbo = self._shadow[0]
        fbo.use()
        ctx.viewport = (0, 0, SHADOW_SIZE, SHADOW_SIZE)
        fbo.clear(depth=1.0)
        ctx.enable_only(moderngl.DEPTH_TEST)
        svp = _mat_bytes(self._shadow_vp)
        # lines as ribbons at their stroke width (across the sun), at least SHADOW_LINE_PX texels wide
        lines = [(np.asarray(s[0], float), s[1]) for s in strokes
                 if len(s[0]) >= 2 and not (len(s) > 5 and s[5] is not None)]
        if lines:
            data = shadow_ribbons(lines, d, self._shadow_texel * SHADOW_LINE_PX, r)
            if self._shadow_lines is None or data.nbytes > self._shadow_lines[0].size:
                if self._shadow_lines is not None:
                    self._shadow_lines[1].release()
                    self._shadow_lines[0].release()
                vbo = ctx.buffer(reserve=max(data.nbytes * 2, 1 << 14), dynamic=True)
                self._shadow_lines = (vbo, ctx.vertex_array(self.shadow_line_prog, [(vbo, "3f", "in_pos")]))
            self._shadow_lines[0].write(data.tobytes())
            self.shadow_line_prog["svp"].write(svp)
            self._shadow_lines[1].render(moderngl.TRIANGLES, vertices=len(data))
        # surfaces
        if self._surfaces:
            prog = self.shadow_surf_prog
            prog["vp"].write(svp)
            prog["bias"].value = 0.0
            vao = self._surface_vao(prog)
            vao.render(moderngl.TRIANGLES, vertices=sum(s["count"] for s in self._surfaces))
            counters.C["zeichenaufrufe"] += 1
        # Meshes
        prog = self.pick_prog
        for i, (mesh, _) in enumerate(meshes):
            if isinstance(mesh, _MeshBlock):
                self._draw_block(mesh, "pick", self._shadow_vp)
                continue
            if mesh.geo.triangle_count == 0:
                continue
            prog["mvp"].write(_mat_bytes(self._shadow_vp @ mesh.xform))
            prog["model"].write(_mat_bytes(mesh.xform))
            prog["mesh_index"].value = float(i + 1)
            self._mesh_buffers(mesh.geo).pick.render(moderngl.TRIANGLES)

    def _msaa_target(self, size):
        if self._msaa_size != size:
            if self._msaa_fbo is not None:
                for att in self._msaa_fbo.color_attachments:
                    att.release()
                self._msaa_fbo.depth_attachment.release()
                self._msaa_fbo.release()
            samples = min(self.MSAA_SAMPLES, self.ctx.max_samples)
            self._msaa_fbo = self.ctx.framebuffer(
                [self.ctx.renderbuffer(size, 4, samples=samples)],
                self.ctx.depth_renderbuffer(size, samples=samples))
            self._msaa_size = size
        return self._msaa_fbo

    def _draw_meshes(self, vp, light_eye, meshes, alpha):
        prog = self.mesh_prog
        self._light_uniforms(prog, surface=True)  # light mode: sun and shadow like with the surfaces
        prog["eye"].value = tuple(float(x) for x in light_eye)
        prog["alpha"].value = alpha
        if "lift" in prog:
            # check point a few texels of the shadow map away from the mesh: otherwise it shadows itself
            prog["lift"].value = max(getattr(self, "_shadow_lift", 0.02), 3.0 * getattr(self, "_shadow_texel", 0.0))
        for mesh, tint in meshes:
            if isinstance(mesh, _MeshBlock):
                self._draw_block(mesh, "shade", vp, light_eye=light_eye, alpha=alpha)
                continue
            if mesh.geo.triangle_count == 0:
                continue
            prog["mvp"].write(_mat_bytes(vp @ mesh.xform))
            prog["model"].write(_mat_bytes(mesh.xform))
            prog["tint"].value = tint
            self._mesh_buffers(mesh.geo).shade.render(moderngl.TRIANGLES)

    @staticmethod
    def _unblock(meshes):
        """Mesh list with the drawn parts of the block individually (wireframe)."""
        out = []
        for m, t in meshes:
            if isinstance(m, _MeshBlock):
                rows = m.rows()
                out += [(mm, tuple(rows[k, 0])) for k, mm in enumerate(m.meshes) if rows[k, 1, 1] > 0.5]
            else:
                out.append((m, t))
        return out

    def _draw_block(self, block, name, vp, vis_col=1, light_eye=None, alpha=1.0):
        """Draw the block in one pass (program like the one per mesh, per-part values from the table)."""
        prog = self.block_progs[name]
        prog["mvp"].write(_mat_bytes(vp))
        if name == "shade":
            self._light_uniforms(prog, surface=True)
            prog["eye"].value = tuple(float(x) for x in light_eye)
            prog["alpha"].value = alpha
            if "lift" in prog:
                prog["lift"].value = max(getattr(self, "_shadow_lift", 0.02), 3.0 * getattr(self, "_shadow_texel", 0.0))
        elif name == "occl":
            eye, back, ortho = self._view
            prog["eye"].value = tuple(float(x) for x in eye)
            prog["view_back"].value = tuple(float(x) for x in back)
            prog["ortho"].value = 1.0 if ortho else 0.0
            prog["radius"].value = MESH_OCCL_RADIUS
        elif name == "occl_acc":
            self._occl_uniforms(prog)
            prog["acc_value"].value = 20.0
        block.bind(prog, vis_col)
        block.vaos[name].render(moderngl.TRIANGLES)
        counters.C["zeichenaufrufe"] += 1

    def _draw_wires(self, vp, meshes, alpha, bias=0.0):
        """Wireframe: edges of the meshes in the stroke color (selected ones tinted like when shading); bias: shifted
        toward the camera (depth) so edges on their own surface do not flicker."""
        prog = self.wire_prog
        prog["bias"].value = float(bias)
        ink = theme.rgb(theme.STROKE_COLOR)
        for mesh, tint in self._unblock(meshes):
            if mesh.geo.triangle_count == 0:
                continue
            prog["mvp"].write(_mat_bytes(vp @ mesh.xform))
            if "model" in prog:
                prog["model"].write(_mat_bytes(mesh.xform))
            t = float(tint[3])
            prog["color"].value = (*(a + (b - a) * t for a, b in zip(ink, tint[:3])), float(alpha))
            self._mesh_buffers(mesh.geo).wire(prog, mesh.geo).render(moderngl.LINES)

    def _infinite_quad(self):
        """Per infinite plane a large quad around the point under the camera."""
        if not self._infinite:
            return None
        eye, _, ortho = self._view
        quads = []
        for layer in self._infinite:
            o, u, v, reach = layer["origin"], layer["u"], layer["v"], layer["reach"]
            n = np.cross(u, v)
            n /= max(np.linalg.norm(n), 1e-12)
            center = o if ortho else eye - n * float((eye - o) @ n)
            du, dv = u * reach, v * reach
            c = [center - du - dv, center + du - dv, center + du + dv, center - du + dv]
            quads.append(np.array([c[0], c[1], c[2], c[0], c[2], c[3]]))
        return np.vstack(quads)

    def _draw_infinite(self, vp, alpha_scale=1.0):
        if not self._infinite:
            return
        prog = self.inf_prog
        prog["vp"].write(_mat_bytes(vp))
        prog["bias"].value = 0.0
        eye, back, ortho = self._view
        prog["eye"].value = tuple(float(x) for x in eye)
        prog["view_back"].value = tuple(float(x) for x in back)
        prog["ortho"].value = 1.0 if ortho else 0.0
        self._occl_uniforms(prog)
        for k, layer in enumerate(self._infinite):
            self._occl_owner(prog, owner=layer.get("id", -1))
            a = layer["strength"] * alpha_scale
            core, tile = layer["core"], layer["tile"]
            prog["origin"].value = tuple(float(x) for x in layer["origin"])
            prog["axis_u"].value = tuple(float(x) for x in layer["u"])
            prog["axis_v"].value = tuple(float(x) for x in layer["v"])
            prog["has_core"].value = 1.0 if core is not None else 0.0
            prog["half_size"].value = core if core is not None else (0.0, 0.0)
            prog["tile_size"].value = tile
            prog["tile_offset"].value = core if core is not None else (0.0, 0.0)
            prog["grid_step"].value = (tile[0] / 10, tile[1] / 10)
            prog["fill"].value = (*layer["fill"][:3], theme.INFINITE_FILL_ALPHA * a)
            prog["line"].value = tuple(float(x) for x in layer["line"][:3])
            prog["grid_alpha"].value = theme.INFINITE_GRID_ALPHA * a
            prog["tile_alpha"].value = theme.INFINITE_TILE_ALPHA * a
            prog["reach"].value = float(layer["reach"])
            self._inf_vao.render(moderngl.TRIANGLES, vertices=6, first=self._inf_first + 6 * k)

    def _picture_texture(self, asset, data):
        tex = self._picture_textures.get(asset)
        if tex is None:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QImage
            from .pictures import MAX_TEXTURE
            image = QImage.fromData(data)
            if image.isNull():
                return None
            if max(image.width(), image.height()) > MAX_TEXTURE:
                image = image.scaled(MAX_TEXTURE, MAX_TEXTURE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image = image.convertToFormat(QImage.Format_RGBA8888).flipped(Qt.Vertical)
            tex = self.ctx.texture((image.width(), image.height()), 4,
                                   bytes(image.constBits())[:image.width() * image.height() * 4])
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self._picture_textures[asset] = tex
        return tex

    def _draw_pictures(self, vp):
        """Projected images on helper planes and helper surfaces (pictures.py)."""
        if not self.pictures:
            return
        prog = self.picture_prog
        prog["vp"].write(_mat_bytes(vp))
        prog["bias"].value = 0.0
        self._occl_uniforms(prog)
        for asset, data, matrix, verts, owner in self.pictures:
            tex = self._picture_texture(asset, data)
            if tex is None:
                continue
            tex.use(location=PICTURE_UNIT)
            prog["picture"].value = PICTURE_UNIT
            prog["picture_m"].write(_mat_bytes(matrix))
            self._occl_owner(prog, owner=owner, under=0.0)
            buf = self.ctx.buffer(np.asarray(verts, "f4").tobytes())
            vao = self.ctx.simple_vertex_array(prog, buf, "in_pos")
            vao.render(moderngl.TRIANGLES)
            vao.release()
            buf.release()

    def _draw_planes(self, vp, planes, alpha_scale=1.0):
        if not planes and not self._infinite:
            return
        chunks = []
        for plane, *_ in planes:
            o, u, v = plane.origin, plane.axis_u, plane.axis_v
            hu, hv = u * plane.size_u / 2, v * plane.size_v / 2
            c = [o - hu - hv, o + hu - hv, o + hu + hv, o - hu + hv]
            fill = [c[0], c[1], c[2], c[0], c[2], c[3]]
            grid = []
            for i in range(1, 10):
                s = -1 + 2 * i / 10
                grid += [o + hu * s - hv, o + hu * s + hv, o + hv * s - hu, o + hv * s + hu]
            border = [c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]]
            chunks.append(np.array(fill + grid + border))
        inf = self._infinite_quad()
        self._inf_first = sum(len(c) for c in chunks)
        self._flat_vertices(np.vstack(chunks + ([inf] if inf is not None else [])))

        prog = self.flat_prog
        prog["vp"].write(_mat_bytes(vp))
        prog["bias"].value = 0.0
        eye, back, ortho = self._view
        prog["eye"].value = tuple(float(x) for x in eye)
        prog["view_back"].value = tuple(float(x) for x in back)
        prog["ortho"].value = 1.0 if ortho else 0.0
        prog["milk"].value = theme.TRACING_VIEW_MILK
        self._occl_uniforms(prog)
        n_fill, n_grid, n_border = 6, 36, 8
        stride = n_fill + n_grid + n_border
        for k, (plane, fill, line, highlighted) in enumerate(planes):
            first = k * stride
            self._occl_owner(prog, owner=plane.id, under=self._plane_under(highlighted))
            prog["soft"].value = 0.0 if int(highlighted) else 1.0
            prog["color"].value = (*fill[:3], fill[3] * alpha_scale)
            prog["grain"].value = 1.0  # fibers only in the paper surface
            self._flat_vao.render(moderngl.TRIANGLES, vertices=n_fill, first=first)
            prog["grain"].value = 0.0
            grid_a, border_a, width = theme.PLANE_EMPHASIS[int(highlighted)]
            prog["under"].value = 0.0  # grid not over a surface
            prog["color"].value = (*line[:3], grid_a * alpha_scale)  # grid: hinted, active clearly
            self._flat_vao.render(moderngl.LINES, vertices=n_grid, first=first + n_fill)
            prog["under"].value = -1.0  # border stays
            prog["color"].value = (*line[:3], border_a * alpha_scale)
            self.ctx.line_width = width
            self._flat_vao.render(moderngl.LINES, vertices=n_border, first=first + n_fill + n_grid)
        self.ctx.line_width = 1.0
        self._draw_infinite(vp, alpha_scale)

    # --- Picking ---
    def update_pick(self, size, cam, proj, meshes):
        """Re-render the position buffer if camera, size or meshes have changed."""
        # the geometries themselves (comparison by identity), not their id: a freed id can get a new shape
        key = (size, cam.eye.tobytes(), cam.rot.tobytes(), np.asarray(proj).tobytes(),
               tuple((m.geo, m.xform.tobytes()) for m in meshes))
        if key == self._pick_key:
            return
        ctx = self.ctx
        if self._pick_size != size:
            if self._pick_fbo is not None:
                for att in self._pick_fbo.color_attachments:
                    att.release()
                self._pick_fbo.depth_attachment.release()
                self._pick_fbo.release()
            color = ctx.texture(size, 4, dtype="f4")
            prim = ctx.texture(size, 1, dtype="f4")  # triangle number per pixel
            self._pick_fbo = ctx.framebuffer([color, prim], ctx.depth_renderbuffer(size))
            self._pick_size = size
        self._pick_fbo.use()
        ctx.viewport = (0, 0, *size)
        ctx.scissor = None
        self._pick_fbo.clear(0, 0, 0, 0, depth=1.0)
        ctx.enable_only(moderngl.DEPTH_TEST)
        vp = proj @ view_matrix(cam)
        prog = self.pick_prog
        block, in_block = self._block, set()
        if block is not None and len(meshes) >= BLOCK_MIN:  # meshes in the block: one pass, number from the table
            from .scene import MeshObject
            rows = block.rows()
            rows[:, 2, 0] = 0.0
            for i, mesh in enumerate(meshes):
                k = block.current(mesh) if type(mesh) is MeshObject else None
                if k is not None:
                    rows[k, 2, 0] = float(i + 1)
                    in_block.add(i)
            if in_block:
                self._draw_block(block, "pick", vp, vis_col=2)
        for i, mesh in enumerate(meshes):
            if i in in_block or mesh.geo.triangle_count == 0:
                continue
            prog["mvp"].write(_mat_bytes(vp @ mesh.xform))
            prog["model"].write(_mat_bytes(mesh.xform))
            prog["mesh_index"].value = float(i + 1)
            self._mesh_buffers(mesh.geo).pick.render(moderngl.TRIANGLES)
        ctx.enable_only(moderngl.NOTHING)
        self._pick_meshes = list(meshes)
        self._pick_key = key

    def read_pick(self, x, y, w=1, h=1):
        """Rectangle from (x, y) bottom left in physical pixels -> (array (h, w, 4), x0, y0).
        Per pixel xyz of the surface and mesh number (0 = nothing hit). Clipped at the border."""
        if self._pick_fbo is None:
            return np.zeros((1, 1, 4), dtype="f4"), x, y
        pw, ph = self._pick_size
        x0, y0 = max(0, min(pw - 1, x)), max(0, min(ph - 1, y))
        w, h = max(1, min(w, pw - x0)), max(1, min(h, ph - y0))
        data = self._pick_fbo.read(viewport=(x0, y0, w, h), components=4, dtype="f4")
        return np.frombuffer(data, dtype="f4").reshape(h, w, 4), x0, y0

    def read_pick_points(self, coords):
        """Individual pixels (physical, bottom left) of the position buffer -> (array (N, 4), triangle numbers (N,)).
        The GPU collects them into one row, only that is read - instead of the whole rectangle around all points."""
        coords = np.ascontiguousarray(coords, dtype="f4").reshape(-1, 2)
        n = len(coords)
        if self._pick_fbo is None or not n:
            return np.zeros((n, 4), dtype="f4"), np.zeros(n, dtype=np.int64)
        ctx = self.ctx
        if getattr(self, "_gather", None) is None:
            prog = program(ctx, 'pick_gather')
            self._gather = [prog, None, None, None, 0]  # prog, vbo, vao, fbo, capacity
        g = self._gather
        if n > g[4]:
            for res in g[1:4]:
                if res is not None:
                    res.release()
            cap = max(n, 512) * 2
            g[1] = ctx.buffer(reserve=cap * 8, dynamic=True)
            g[2] = ctx.vertex_array(g[0], [(g[1], "2f", "in_xy")])
            g[3] = ctx.framebuffer([ctx.texture((cap, 1), 4, dtype="f4"), ctx.texture((cap, 1), 1, dtype="f4")])
            g[4] = cap
        g[1].write(coords.tobytes())
        g[3].use()
        ctx.viewport = (0, 0, n, 1)
        ctx.enable_only(moderngl.PROGRAM_POINT_SIZE)
        self._pick_fbo.color_attachments[0].use(location=0)
        self._pick_fbo.color_attachments[1].use(location=1)
        g[0]["pick_tex"].value = 0
        g[0]["prim_tex"].value = 1
        g[0]["count"].value = float(n)
        g[2].render(moderngl.POINTS, vertices=n)
        ctx.enable_only(moderngl.NOTHING)
        data = g[3].read(viewport=(0, 0, n, 1), components=4, dtype="f4")
        prim = g[3].read(viewport=(0, 0, n, 1), components=1, dtype="f4", attachment=1)
        return (np.frombuffer(data, dtype="f4").reshape(n, 4),
                np.round(np.frombuffer(prim, dtype="f4")).astype(np.int64))

    def points_visible(self, points):
        """(N,) bool: which points in the last frame are not hidden by a surface (coverage map, like with the
        strokes). Call right after render() (same context); binds its own target - afterwards bind the image again.
        Without a coverage map: all visible."""
        pts = np.ascontiguousarray(points, dtype="f4").reshape(-1, 3)
        n = len(pts)
        if not n or not getattr(self, "_occl_on", False):
            return np.ones(n, bool)
        ctx = self.ctx
        if getattr(self, "_vis", None) is None:
            self._vis = [program(ctx, 'point_vis'), None, None, None, 0]
        g = self._vis
        if n > g[4]:
            for res in g[1:4]:
                if res is not None:
                    res.release()
            cap = max(n, 512) * 2
            g[1] = ctx.buffer(reserve=cap * 12, dynamic=True)
            g[2] = ctx.vertex_array(g[0], [(g[1], "3f", "in_pos")])
            g[3] = ctx.framebuffer([ctx.texture((cap, 1), 1, dtype="f4")])
            g[4] = cap
        g[1].write(pts)
        g[3].use()
        ctx.viewport = (0, 0, n, 1)
        ctx.enable_only(moderngl.PROGRAM_POINT_SIZE)
        prog = g[0]
        prog["count"].value = float(n)
        prog["vp"].write(_mat_bytes(self._frame_vp))
        prog["size"].value = tuple(float(x) for x in self._frame_size)
        self._occl_uniforms(prog)
        g[2].render(moderngl.POINTS, vertices=n)
        ctx.enable_only(moderngl.NOTHING)
        vals = np.frombuffer(g[3].read(viewport=(0, 0, n, 1), components=1, dtype="f4"), dtype="f4")
        return vals > 0.5

    def id_map(self):
        """(h, w) float: per pixel of the last frame the id of the frontmost mesh or surface (copies under
        generators: their display id), -1 = nothing - color ID of the brief. All meshes cover, also semi-transparent ones;
        helper surfaces do not. Row 0 = bottom. Call right after render() (same context); rebuilds the coverage map
        - afterwards bind the image again."""
        w, h = self._frame_size
        self._update_occlusion(self._frame_vp, (w, h), getattr(self, "_frame_meshes", ()), 1.0, self._surfaces)
        if not self._occl_on:
            return np.full((h, w), -1.0, dtype="f4")
        return np.frombuffer(self._occl[1].read(), dtype="f4").reshape(h, w, 4)[..., 3].copy()

    def pick_mesh(self, index):
        return self._pick_meshes[index - 1] if 0 < index <= len(self._pick_meshes) else None
