"""稀疏体素区块网格化。

坐标约定
--------
* 体素坐标 ``(x, y, z)`` 是整数格点，方块占据单位立方体
  ``[x, x+1] × [y, y+1] × [z, z+1]``。
* 区块是稀疏字典 ``{(x, y, z): material_id}``：``material_id == 0`` 表示空气，
  正整数表示实心方块。
* 面是六元组 ``(x, y, z, axis, sign, material)``：``axis`` 取 0/1/2 表示法线沿
  X/Y/Z 轴，``sign`` 取 ``+1``/``-1`` 表示朝向该轴的正/负方向。

顶点顺序
--------
所有输出的三角形顶点均按逆时针（CCW）排列，观察方向为从面的外侧看向模型；
即对三角形 ``(v0, v1, v2)`` 有 ``(v1 - v0) × (v2 - v0)`` 与面法线同向。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = [
    "MeshError",
    "assign_materials",
    "build_vertices",
    "compute_ao",
    "cull_hidden",
    "estimate_triangles",
    "greedy_merge",
    "mesh_batch",
    "stitch_borders",
]


class MeshError(Exception):
    """网格化过程中的输入或契约错误（未知材质、区块冲突等）。"""


def cull_hidden(
    blocks: Mapping[tuple[int, int, int], int],
    neighbors: Mapping[tuple[int, int, int], int],
) -> list[tuple[int, int, int, int, int, int]]:
    """剔除隐藏面，返回区块内所有与空气相邻的单位面。

    参数
    ----
    blocks
        本区块的稀疏字典，键为世界坐标 ``(x, y, z)``，值为材质 id（0 为空气）。
    neighbors
        相邻区块的稀疏字典，坐标同为世界坐标；只用于判断本区块边界外一圈的
        邻接体素，本身不生成面。

    契约
    ----
    * 只为 ``blocks`` 中的实心体素生成面：对每个实心体素的 6 个方向各检查一次，
      若该方向上的邻接体素（在同区块的 ``blocks`` 或跨区块的 ``neighbors`` 中
      查找）不是实心（缺失或为 0），就生成面
      ``(x, y, z, axis, sign, material)``。
    * 区块边界外的邻居参与判断，接缝两侧相对的隐藏面不得重复生成。
    * 空气体素（值为 0）不生成任何面，也不得因为它出现在字典里就当作占位；
      相邻的空气体素同样不能遮挡实心体素的面。
    * 返回列表按元组字典序排序，保证结果与字典插入顺序无关。
    """

    def solid(pos: tuple[int, int, int]) -> bool:
        if pos in blocks:
            return blocks[pos] != 0
        return neighbors.get(pos, 0) != 0

    faces: list[tuple[int, int, int, int, int, int]] = []
    for pos, material in blocks.items():
        if material == 0:
            continue
        x, y, z = pos
        for axis in range(3):
            for sign in (1, -1):
                npos = [x, y, z]
                npos[axis] += sign
                if not solid((npos[0], npos[1], npos[2])):
                    faces.append((x, y, z, axis, sign, material))
    faces.sort()
    return faces


def greedy_merge(
    faces: Iterable[tuple[int, int, int, int, int, int]],
    axis: int,
) -> list[dict]:
    """把同一平面、同材质、同朝向的单位面贪心合并成大矩形。

    参数
    ----
    faces
        ``cull_hidden`` 返回的单位面序列。
    axis
        只处理法线沿该轴的面（``face[3] == axis``），其它轴的面直接忽略。

    平面内坐标
    ----------
    ``t1 = (axis + 1) % 3``，``t2 = (axis + 2) % 3``，满足
    ``e_t1 × e_t2 = e_axis``。在同一 ``(sign, material, 沿 axis 的坐标)`` 分组内，
    把 ``(t1, t2)`` 网格上边相邻的单位面合并为矩形。

    返回
    ----
    ``list[dict]``，每个矩形包含：

    * ``axis`` / ``sign`` / ``material``：与输入面一致；
    * ``origin``：``(x, y, z)``，矩形最小角对应的体素坐标（沿 ``axis`` 的分量
      是该平面的体素坐标，即 ``sign == +1`` 时平面在其 ``+1`` 处）；
    * ``extent``：``(n1, n2)``，沿 ``t1``、``t2`` 方向的边长（单位面个数）；
    * ``area``：``n1 * n2``。

    合并结果必须无重叠地覆盖全部输入面，所有 ``area`` 之和等于输入面数；
    返回列表按 ``(axis, sign, material, origin)`` 字典序排序。
    """
    t1 = (axis + 1) % 3
    t2 = (axis + 2) % 3

    groups: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for face in faces:
        if face[3] != axis:
            continue
        coord = (face[0], face[1], face[2])
        key = (face[4], face[5], coord[axis])
        groups.setdefault(key, set()).add((coord[t1], coord[t2]))

    rects: list[dict] = []
    for (sign, material, acoord), cells in groups.items():
        remaining = set(cells)
        while remaining:
            a0, b0 = min(remaining)
            n1 = 1
            while (a0 + n1, b0) in remaining:
                n1 += 1
            n2 = 1
            while all((a0 + i, b0 + n2) in remaining for i in range(n1)):
                n2 += 1
            for i in range(n1):
                for j in range(n2):
                    remaining.discard((a0 + i, b0 + j))
            origin = [0, 0, 0]
            origin[axis] = acoord
            origin[t1] = a0
            origin[t2] = b0
            rects.append(
                {
                    "axis": axis,
                    "sign": sign,
                    "material": material,
                    "origin": (origin[0], origin[1], origin[2]),
                    "extent": (n1, n2),
                    "area": n1 * n2,
                }
            )
    rects.sort(key=lambda r: (r["axis"], r["sign"], r["material"], r["origin"]))
    return rects


def build_vertices(faces: Iterable[Mapping[str, object]]) -> dict:
    """把合并后的矩形面转换成三角网格。

    参数
    ----
    faces
        ``greedy_merge`` 返回的矩形 ``dict`` 序列。

    返回
    ----
    ``dict``，恰好包含以下键：

    * ``vertices``：每个矩形 4 个顶点，坐标 ``(x, y, z)``，按矩形顺序追加；
    * ``normals``：与 ``vertices`` 等长，每个顶点一个单位法线 ``(nx, ny, nz)``；
    * ``indices``：每个矩形 6 个索引，两个三角形分别为 ``0-1-2``、``0-2-3``；
    * ``materials``：每个矩形 1 个材质 id；
    * ``triangle_count``：``len(indices) // 3``。

    顶点顺序
    --------
    每个矩形取最小角 ``origin``，把沿 ``axis`` 的分量平移到该面所在的平面
    （``sign == +1`` 时加 1），得到基点 ``base``；顶点依次为 ``base``、
    ``base + n1*e_t1``、``base + n1*e_t1 + n2*e_t2``、``base + n2*e_t2``。
    该顺序从面外侧观察为逆时针（CCW），即每个三角形
    ``(v0, v1, v2)`` 的 ``(v1 - v0) × (v2 - v0)`` 与顶点法线点积为正；
    ``sign == -1`` 时上述四个顶点反向排列。

    ``faces`` 为空时返回五个空列表/0 组成的空网格。
    """
    vertices: list[tuple[int, int, int]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []
    materials: list[int] = []

    for rect in faces:
        axis = rect["axis"]
        sign = rect["sign"]
        n1, n2 = rect["extent"]
        t1 = (axis + 1) % 3
        t2 = (axis + 2) % 3

        base = list(rect["origin"])
        if sign == 1:
            base[axis] += 1

        def corner(a: int, b: int) -> tuple[int, int, int]:
            p = list(base)
            p[t1] += a * n1
            p[t2] += b * n2
            return (p[0], p[1], p[2])

        quad = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)]
        if sign == -1:
            quad.reverse()

        normal = [0.0, 0.0, 0.0]
        normal[axis] = float(sign)

        start = len(vertices)
        vertices.extend(quad)
        normals.extend([tuple(normal)] * 4)
        indices.extend([start, start + 1, start + 2, start, start + 2, start + 3])
        materials.append(rect["material"])

    return {
        "vertices": vertices,
        "normals": normals,
        "indices": indices,
        "materials": materials,
        "triangle_count": len(indices) // 3,
    }


def compute_ao(
    blocks: Mapping[tuple[int, int, int], int],
    faces: Iterable[tuple[int, int, int, int, int, int]],
) -> list[float]:
    """按三邻接体素计算环境光遮蔽（AO）。

    参数
    ----
    blocks
        稀疏字典，含空气（0）与实心体素，坐标为世界坐标；可同时包含面所在
        区块及其邻居。
    faces
        单位面序列（格式与 ``cull_hidden`` 返回值一致）。

    返回
    ----
    长度为 ``4 * len(faces)`` 的浮点列表：每个面的 4 个顶点按
    ``build_vertices`` 的顶点顺序各给一个 ``0.0..1.0`` 的 AO 值
    （1 表示完全开阔，0 表示完全遮挡）。

    计算方式
    --------
    对每个顶点，取该顶点外侧（沿法线方向平移一层）的三个体素：

    * ``side1``、``side2``：沿两条切向轴、朝该顶点外侧相邻的两个体素；
    * ``corner``：两者对角的那个外侧体素。

    记三者是否实心为 ``s1``、``s2``、``c``（实心为 1、空气为 0），遮挡计数
    ``count = s1 + s2 + (0 if (s1 and s2) else c)``，返回
    ``1.0 - count / 3.0``。只算一两个邻接会漏掉角落遮挡，是不允许的。
    """

    def solid(pos: tuple[int, int, int]) -> int:
        return 1 if blocks.get(pos, 0) != 0 else 0

    ao: list[float] = []
    for face in faces:
        x, y, z, axis, sign, _material = face
        t1 = (axis + 1) % 3
        t2 = (axis + 2) % 3

        # 与 build_vertices 相同的顶点顺序（单位面 n1 == n2 == 1）。
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        if sign == -1:
            corners.reverse()

        for a, b in corners:
            d1 = 1 if a else -1
            d2 = 1 if b else -1
            p1 = [x, y, z]
            p1[axis] += sign
            p1[t1] += d1
            p2 = [x, y, z]
            p2[axis] += sign
            p2[t2] += d2
            pc = [x, y, z]
            pc[axis] += sign
            pc[t1] += d1
            pc[t2] += d2
            s1 = solid((p1[0], p1[1], p1[2]))
            s2 = solid((p2[0], p2[1], p2[2]))
            c = solid((pc[0], pc[1], pc[2]))
            count = s1 + s2 + (0 if (s1 and s2) else c)
            ao.append(1.0 - count / 3.0)
    return ao


def stitch_borders(
    chunks: Mapping[object, Mapping[tuple[int, int, int], int]],
) -> dict:
    """拼接多个区块并去掉接缝处重复生成的面。

    参数
    ----
    chunks
        ``{区块键: 稀疏字典}``；各区块的体素坐标都是世界坐标。

    返回
    ----
    ``dict``，包含：

    * ``blocks``：合并后的稀疏字典（同一坐标只保留一份）；
    * ``faces``：合并后统一做跨区块剔除得到的单位面列表，按字典序排序，
      同一位置的面最多出现一次（接缝两侧相对的隐藏面必须剔除）。

    同一坐标在多个区块中出现时：材质相同视为重复并去重；材质不同抛
    ``MeshError``。
    """
    blocks: dict[tuple[int, int, int], int] = {}
    for key in sorted(chunks, key=repr):
        for pos, material in chunks[key].items():
            existing = blocks.get(pos)
            if existing is not None and existing != material:
                raise MeshError(
                    f"坐标 {pos} 在区块间冲突：{existing} != {material}"
                )
            blocks[pos] = material
    return {"blocks": blocks, "faces": cull_hidden(blocks, {})}


def assign_materials(
    blocks: Mapping[tuple[int, int, int], int],
    palette: Mapping[int, str],
) -> dict[tuple[int, int, int], int]:
    """把方块 id 映射为紧凑的材质索引。

    参数
    ----
    blocks
        ``{(x, y, z): 方块 id}``；id ``0`` 为空气，不进入输出。
    palette
        ``{方块 id: 材质名}``；材质索引为按 id 升序排序后的位置（从 0 开始）。

    返回
    ----
    ``{(x, y, z): 材质索引}``，只包含非空气体素，键按坐标排序。

    非空气方块 id 不在 ``palette`` 中时抛 ``MeshError``；不允许用 ``-1``
    之类的哨兵值兜底。
    """
    index_of = {block_id: i for i, block_id in enumerate(sorted(palette))}
    result: dict[tuple[int, int, int], int] = {}
    for pos in sorted(blocks):
        block_id = blocks[pos]
        if block_id == 0:
            continue
        if block_id not in index_of:
            raise MeshError(f"未知方块 id：{block_id}（坐标 {pos}）")
        result[pos] = index_of[block_id]
    return result


def estimate_triangles(faces: Iterable[Mapping[str, object]]) -> int:
    """统计合并后网格的三角形数。

    参数
    ----
    faces
        ``greedy_merge`` 返回的矩形面列表。

    返回
    ----
    ``2 * len(faces)``：每个矩形两个三角形。

    不得按矩形的 ``area`` 或未合并的单位面数统计，否则合并后的数值会退回
    未合并规模，明显偏大。
    """
    return 2 * len(list(faces))


def mesh_batch(
    chunks: Mapping[object, Mapping[tuple[int, int, int], int]],
    palette: Mapping[int, str],
) -> dict:
    """批量拼接区块并生成一份完整网格。

    参数
    ----
    chunks
        ``{区块键: 稀疏字典}``，各区块坐标均为世界坐标。
    palette
        同 ``assign_materials`` 的 ``{方块 id: 材质名}``。

    返回
    ----
    ``dict``，包含：

    * ``blocks``：合并去重后的稀疏字典；
    * ``faces``：跨区块剔除后、按字典序排序的单位面列表；
    * ``materials``：``assign_materials(blocks, palette)`` 的结果；
    * ``mesh``：``build_vertices`` 处理所有矩形面（按三个轴分别合并）后的
      网格字典。

    原子性与确定性
    --------------
    先校验全部区块（材质、重复坐标等），任一区块非法就抛 ``MeshError``，
    不得只跳过非法区块或返回其余区块的部分结果。输出与区块键、字典插入
    顺序无关：同一场景两次调用必须完全相等。
    """
    stitched = stitch_borders(chunks)
    blocks = stitched["blocks"]
    faces = stitched["faces"]
    materials = assign_materials(blocks, palette)
    rects: list[dict] = []
    for axis in (0, 1, 2):
        rects.extend(greedy_merge(faces, axis))
    return {
        "blocks": blocks,
        "faces": faces,
        "materials": materials,
        "mesh": build_vertices(rects),
    }
