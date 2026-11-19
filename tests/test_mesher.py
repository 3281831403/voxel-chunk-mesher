"""voxel-chunk-mesher 验收测试。

15 个测试全部针对 voxel/mesher.py 的公开契约；本文件不可修改。
区块统一用稀疏字典 {(x, y, z): material_id} 表示，0 为空气。
面为 (x, y, z, axis, sign, material)，axis 0/1/2 对应 X/Y/Z。
"""

import pytest

from voxel.mesher import (
    MeshError,
    assign_materials,
    build_vertices,
    compute_ao,
    cull_hidden,
    estimate_triangles,
    greedy_merge,
    mesh_batch,
    stitch_borders,
)

PALETTE = {1: "stone", 2: "dirt"}


def merge_all(faces):
    """按三个轴分别贪心合并，返回全部矩形面。"""
    merged = []
    for axis in (0, 1, 2):
        merged.extend(greedy_merge(faces, axis))
    return merged


# --------------------------------------------------------------- 10 个缺陷

def test_cull_hidden_uses_cross_chunk_neighbors():
    """缺陷 1：只看块内邻居，区块边界处重复生成面。"""
    faces = cull_hidden({(0, 0, 0): 1}, {(1, 0, 0): 1})
    assert len(faces) == 5
    assert (0, 0, 0, 0, 1, 1) not in faces
    assert (0, 0, 0, 0, -1, 1) in faces


def test_greedy_merge_merges_along_y_and_z():
    """缺陷 2：贪心合并只沿 X 轴，Y/Z 方向不合并。"""
    merged_y = greedy_merge([(0, y, 0, 0, 1, 1) for y in range(3)], 0)
    assert len(merged_y) == 1
    assert merged_y[0]["area"] == 3
    assert merged_y[0]["extent"] == (3, 1)

    merged_z = greedy_merge([(0, 0, z, 1, 1, 1) for z in range(3)], 1)
    assert len(merged_z) == 1
    assert merged_z[0]["extent"] == (3, 1)

    merged_x = greedy_merge([(x, 0, 0, 2, 1, 1) for x in range(3)], 2)
    assert len(merged_x) == 1
    assert merged_x[0]["extent"] == (3, 1)


def test_build_vertices_winds_ccw_outward():
    """缺陷 3：顶点顺序顺时针，背面剔除后模型内外翻。"""
    faces = merge_all(cull_hidden({(0, 0, 0): 1, (1, 0, 0): 1}, {}))
    mesh = build_vertices(faces)
    vertices = mesh["vertices"]
    normals = mesh["normals"]
    indices = mesh["indices"]
    assert mesh["triangle_count"] == len(indices) // 3
    assert mesh["triangle_count"] > 0
    assert set(normals) == {
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    }
    for i in range(0, len(indices), 3):
        a = vertices[indices[i]]
        b = vertices[indices[i + 1]]
        c = vertices[indices[i + 2]]
        n = normals[indices[i]]
        e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        )
        dot = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2]
        assert dot > 0


def test_compute_ao_counts_corner_occluder():
    """缺陷 4：AO 只算一个邻接，角落出现亮斑。"""
    blocks = {(0, 0, 0): 1, (1, 1, 1): 1}
    ao = compute_ao(blocks, [(0, 0, 0, 2, 1, 1)])
    assert len(ao) == 4
    assert all(0.0 <= value <= 1.0 for value in ao)
    assert ao[0] == pytest.approx(1.0)
    assert ao[2] == pytest.approx(2.0 / 3.0)


def test_stitch_borders_dedupes_seam_faces():
    """缺陷 5：跨区块同类面不去重，接缝处 Z-fighting。"""
    stitched = stitch_borders({"a": {(0, 0, 0): 1}, "b": {(1, 0, 0): 1}})
    assert stitched["blocks"] == {(0, 0, 0): 1, (1, 0, 0): 1}
    faces = stitched["faces"]
    assert len(faces) == 10
    assert len(set(faces)) == 10
    assert (0, 0, 0, 0, 1, 1) not in faces
    assert (1, 0, 0, 0, -1, 1) not in faces


def test_assign_materials_rejects_unknown_block_id():
    """缺陷 6：材质映射越界不报错，输出里出现 -1 材质。"""
    assert assign_materials({(0, 0, 0): 1, (1, 0, 0): 1}, PALETTE) == {
        (0, 0, 0): 0,
        (1, 0, 0): 0,
    }
    with pytest.raises(MeshError):
        assign_materials({(0, 0, 0): 1, (2, 0, 0): -1}, PALETTE)


def test_estimate_triangles_counts_merged_quads():
    """缺陷 7：三角形数按面数 ×2 估算，合并后统计偏大。"""
    faces = cull_hidden({(x, 0, 0): 1 for x in range(3)}, {})
    merged = merge_all(faces)
    assert estimate_triangles(merged) == 2 * len(merged)
    assert estimate_triangles(merged) < 2 * len(faces)


def test_mesh_batch_is_deterministic_across_dict_order():
    """缺陷 8：稀疏字典遍历顺序不稳定，两次网格顺序不同。"""
    blocks_a = {(0, 0, 0): 1, (1, 0, 0): 2, (0, 0, 1): 1}
    blocks_b = {(0, 0, 1): 1, (1, 0, 0): 2, (0, 0, 0): 1}
    assert mesh_batch({"a": blocks_a}, PALETTE) == mesh_batch({"b": blocks_b}, PALETTE)


def test_mesh_batch_rolls_back_when_a_chunk_is_invalid():
    """缺陷 9：单区块非法只跳过，批量结果缺块。"""
    chunks = {"good": {(0, 0, 0): 1}, "bad": {(4, 0, 0): 99}}
    with pytest.raises(MeshError):
        mesh_batch(chunks, PALETTE)


def test_cull_hidden_treats_air_as_empty():
    """缺陷 10：空气体素也参与占位判断，内部面被生成。"""
    assert cull_hidden({(0, 0, 0): 0}, {}) == []
    faces = cull_hidden({(0, 0, 0): 1, (1, 0, 0): 0, (0, 1, 0): 0}, {})
    assert len(faces) == 6
    assert (0, 0, 0, 0, 1, 1) in faces
    assert (0, 0, 0, 1, 1, 1) in faces


# --------------------------------------------------------------- 5 个组合

def test_single_voxel_yields_six_faces():
    faces = cull_hidden({(0, 0, 0): 1}, {})
    assert len(faces) == 6
    mesh = build_vertices(merge_all(faces))
    assert len(mesh["vertices"]) == 24
    assert len(mesh["indices"]) == 36
    assert mesh["triangle_count"] == 12


def test_empty_chunk_yields_empty_mesh():
    assert cull_hidden({}, {}) == []
    assert build_vertices([]) == {
        "vertices": [],
        "normals": [],
        "indices": [],
        "materials": [],
        "triangle_count": 0,
    }


def test_palette_missing_material_raises_mesh_error():
    with pytest.raises(MeshError):
        assign_materials({(0, 0, 0): 3}, {1: "stone", 2: "dirt"})


def test_same_input_twice_gives_identical_mesh():
    chunks = {"a": {(0, 0, 0): 1, (1, 0, 0): 1, (0, 0, 1): 2}}
    assert mesh_batch(chunks, PALETTE) == mesh_batch(chunks, PALETTE)


def test_merged_triangles_at_most_half_of_unmerged():
    blocks = {(x, y, z): 1 for x in range(3) for y in range(3) for z in range(3)}
    faces = cull_hidden(blocks, {})
    merged = merge_all(faces)
    unmerged_triangles = 2 * len(faces)
    assert estimate_triangles(merged) * 2 <= unmerged_triangles
