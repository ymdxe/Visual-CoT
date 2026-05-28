# Benchmark 样本图片

## 原始输入图片

来自 `data/benchmarks/small_benchmark_*_100.json` 的代表性输入样本，按数据集前缀命名。

| 数据集 | 文件 | 问题（节选） |
|---|---|---|
| GQA (spatial) | `gqa_2389707.jpg` | What kind of bag do you think is the sign in front of? |
| GQA (spatial) | `gqa_2349892.jpg` | Which kind of toy is to the right of the bear? |
| Visual7W | `visual7w_2385707.jpg` | Where is this shot? |
| Visual7W | `visual7w_2338170.jpg` | What is the man riding on? |
| DocVQA | `docvqa_gnnp0227_6.png` | Did John have had any graduate level course in statistics? |
| DocVQA | `docvqa_tyhk0228_2.png` | What is the estimated cost range of 'HCFCs and HCFC Equipment'? |
| InfographicsVQA | `infographicsvqa_44972.jpeg` | who is more likely to be involved in representing data visually? |
| InfographicsVQA | `infographicsvqa_31595.jpeg` | What percentage of internet users in South Africa are men? |

## 三联图示例（原图 + bbox 标注 + 裁剪区域）

每张图展示一条完整的 VisCoT 推理链路：左侧原图、中间红框标注 GPT 输出的归一化 bbox、右侧按 bbox 裁出的局部图（供第二轮答题）。

| 数据集 | 文件 | 问题 → 答案 |
|---|---|---|
| GQA | `example_gqa_2389707.png` | What kind of bag is the sign in front of? → shopping bag |
| Visual7W | `example_visual7w_2385707.png` | Where is this shot? → outdoors |
| DocVQA | `example_docvqa_gnnp0227_6.png` | Did John have had any graduate level course in statistics? → yes |
| InfographicsVQA | `example_infographicsvqa_44972.png` | who is more likely to be involved in representing data visually? → data scientist |
| CUB | `example_cub_herring_gull.png` | Does the bird have broad wings and purple legs? → No |

三联图由 `tools/thesis/render_sample_triptych.py` 生成，可复用于其它样本：

```bash
python tools/thesis/render_sample_triptych.py \
  --benchmark data/benchmarks/small_benchmark_docvqa100.json \
  --question-id 791 \
  --image-folder playground/data \
  --output images/benchmarks/example_docvqa_gnnp0227_6.png
```

> 对应 benchmark JSON 中 `image` 字段的原始路径在 `playground/data/cot/<dataset>/...`，原始数据集图像目录不入库（已在 `.gitignore`）。
