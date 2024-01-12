from py_markdown_table.markdown_table import markdown_table

ONNX_ZOO_URL_START = 'https://github.com/gyulaz-htec/models/tree/5faef4c33eba0395177850e1e31c4a6a9e634c82/'

def save_to_markdown(statistics, fp16=False):
    data = []
    for path, stats in statistics.items():
        tar_file_name = path.split('/')[-1]
        model = tar_file_name.replace('.tar.gz', '')
        url = f"{ONNX_ZOO_URL_START}{path}"
        link = f"[{model}.onnx]({url})"
        opset = model.replace('-int8', '').replace('-qdq', '').split('-')[-1]
        compiles = ':green_heart:' if stats[0] else ':broken_heart:'
        valid = ':green_heart:' if stats[1] else ':broken_heart:'
        error = str(stats[2])
        record = {'Model': link, 'Opset': opset, 'Compilation': compiles, 'Validation': valid, 'Error': error}
        data.append(record)
    if len(data) > 0:
        table = markdown_table(data).set_params(row_sep = 'markdown', quote = False).get_markdown()
        file_name_ending = "_FP16" if fp16 else '';
        with open(f"MIGRAPHX{file_name_ending}.md", "w") as f:
            f.write(table)