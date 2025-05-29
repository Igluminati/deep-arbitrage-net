import nbformat
import sys
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Path to the Obsidian markdown file
file_name = sys.argv[1]
input_path = f"/home/steak/Desktop/My Vault/1 - Knowledge Trees/Arbitrage Deep Learning Project/Neural Networks from Scratch/{file_name}"

output_file = file_name.replace(".md", ".ipynb").replace(" ", "_").lower()
output_path = f"/home/steak/Desktop/deep-arbitrage-net/neural_nets_scratch/{output_file}"

with open(input_path, "r", encoding="utf-8") as f:
    markdown_content = f.read()

cells = []
inside_code_block = False
code_lines = []
markdown_lines = []

for line in markdown_content.splitlines():
    if line.strip().startswith("```python"):
        inside_code_block = True
        if markdown_lines:
            cells.append(new_markdown_cell('\n'.join(markdown_lines)))
            markdown_lines = []
        continue
    elif line.strip() == "```":
        inside_code_block = False
        cells.append(new_code_cell('\n'.join(code_lines)))
        code_lines = []
        continue

    if inside_code_block:
        code_lines.append(line)
    else:
        markdown_lines.append(line)

if markdown_lines:
    cells.append(new_markdown_cell('\n'.join(markdown_lines)))

nb = new_notebook(cells=cells)

with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
