# voxel-chunk-mesher

## 题面

开放世界换成体素地形后，面数暴涨还漏面：初版网格器只按块内邻居剔除隐藏面，区块边界处重复生成面；贪心合并只沿 X 轴，Y/Z 方向不合并、面数翻倍；顶点顺序写成顺时针，背面剔除后模型内外翻；AO 只算一个邻接体素，角落出现亮斑；跨区块同类面不去重，接缝处 Z-fighting；材质映射越界不报错，输出里出现 -1 材质；三角形数按面数 ×2 估算，合并后统计偏大；稀疏字典遍历顺序不稳定，两次网格顺序不同；单区块非法只跳过，批量结果缺块；空气体素也参与占位判断，内部面被生成。voxel/mesher.py 是核心，tests/test_mesher.py 15个红。要修的是：按 docstring 契约实现 cull_hidden、greedy_merge、build_vertices、compute_ao、stitch_borders、assign_materials、estimate_triangles、mesh_batch 八个函数，逐条对照 15 个测试消除上述缺陷。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，且没有 collection error、import error 或 syntax error（输出尾部形如 `15 failed in 0.0Xs`）。修复目标是 15 个测试全部通过（`15 passed`）。

运行环境：Python 3 + pytest，离线运行，无随机数，无外部依赖。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约定

`voxel/mesher.py` 里的 8 个函数当前全部 `raise NotImplementedError`，docstring 就是契约：体素坐标为整数格点，区块用 `{(x, y, z): 0|1}` 稀疏字典表示（0 为空气），面为 `(x, y, z, axis, sign, material)`，顶点顺序为逆时针（CCW，从面外侧观察），非法输入抛 `MeshError`。

## 限制

- `tests/` 不可修改。15 个测试就是验收标准，只能改 `voxel/mesher.py`。
- 只用 Python 标准库（`collections`、`math`）和 pytest。
