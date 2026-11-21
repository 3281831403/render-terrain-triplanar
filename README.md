# render-terrain-triplanar

## 题面

地形换成三平面映射后，陡坡发亮、接缝处还有一道竖线：权重用绝对法线值不归一，陡坡过亮；Y 轴投影 UV 对调，纹理镜像；混合不按权重归一，总和 1.5；法线混合后不归一，光照变暗；过渡带用固定宽度，陡坡出现硬边；缝处 UV 不对齐，接缝明显；法线为零不报错，权重 NaN；负权重不校验，颜色取反；批量共用权重缓冲，样本之间串；sharpness<=0 不校验，除零。triplanar/blend.py 是核心，tests/test_blend.py 15个红。要修的是：按 8 个函数的 docstring 契约实现三轴权重归一、投影 UV、混合归一、法线归一、过渡带宽、缝 UV 对齐、输入校验、批量缓冲隔离与整体回滚，让 15 个测试全部通过。

## 运行

```powershell
cd D:\gsb-20261119\render-terrain-triplanar
python -m pytest tests -q
```

初始仓库必须恰好输出 `15 failed`，不能出现 collection、import 或 syntax error；实现完成后目标是 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 文件结构

```text
render-terrain-triplanar/
  README.md  conftest.py  .gitignore
  triplanar/__init__.py  triplanar/blend.py
  tests/test_blend.py
```

`triplanar/blend.py` 中的 8 个公开函数当前全部是 `raise NotImplementedError` 的桩；世界坐标、混合锐度、轴投影、UV 缝对齐与 1e-6 容差写在各函数的 docstring 中，异常统一使用 `TriplanarError`。

## 数据形状

采样是内联字典，基础字段为 `{"pos": (x, y, z), "normal": (nx, ny, nz)}`，坐标均为世界空间。纹理值是三元组 `(r, g, b)`，三个分量均在 0..1。三轴权重用三元组表示，顺序固定为 `(x, y, z)`。

`triplanar_batch` 的 `config` 可选字段为 `sharpness`（默认 1.0）、`scale`（默认 1.0）、`threshold`（默认 1.0）。成功时返回 `{"count": int, "samples": [...]}`，每个样本结果包含 `pos`、`weights`、`uvs`（`x`/`y`/`z` 三组 UV）与 `band`；空样本返回 `{"count": 0, "samples": []}`。

## 约束

- `tests/test_blend.py` 不可修改；只允许实现 `triplanar/blend.py` 中的桩函数。
- 仅使用标准库（`math`）与 `pytest`；离线、无随机、不联网。
- 15 个测试在初始仓库中必须全部失败，且失败类型来自桩抛出的 `NotImplementedError`，不能是 collection、import 或 syntax error。
- 公共接口的数值比较统一使用绝对容差 `1e-6`。
