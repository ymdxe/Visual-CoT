# Benchmark 样本图片

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

> 对应 benchmark JSON 中 `image` 字段的原始路径在 `playground/data/cot/<dataset>/...`，原始数据集图像目录不入库（已在 `.gitignore`）。
