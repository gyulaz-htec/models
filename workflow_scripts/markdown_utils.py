from py_markdown_table.markdown_table import markdown_table

ONNX_ZOO_URL_START = 'https://github.com/gyulaz-htec/models/tree/migraphx_testing/'

CATEGORY_DICT = {
    "machine_comprehension": "Machine Comprehension",
    "classification": "Image classification",
    "object_detection": "Object Detection & Image Segmentation",
    "body_analysis": "Body, Face & Gesture Analysis",
    "style_transfer": "Image Manipulation",
    "super_resolution": "Image Manipulation"
}

def get_category_from_path(path):
    for k, v in CATEGORY_DICT.items():
        if k in path:
            return v
    raise ValueError(f"No category found for {path}")

def group_statistics(statistics):
    grouped_data = {}
    for path, stats in statistics.items():
        category = get_category_from_path(path)
        tar_file_name = path.split('/')[-1]
        model = tar_file_name.replace('.tar.gz', '')
        url = f"{ONNX_ZOO_URL_START}{path}"
        link = f"[{model}.onnx]({url})"
        opset = model.replace('-int8', '').replace('-qdq', '').split('-')[-1]
        compiles = ':green_heart:' if stats[0] else ':broken_heart:'
        valid = ':green_heart:' if stats[1] else ':broken_heart:'
        error = str(stats[2])
        record = {'Model': link, 'Opset': opset, 'Compilation': compiles, 'Validation': valid, 'Error': error}
        if category in grouped_data:
            grouped_data[category].append(record)
        else:
            grouped_data[category] = [record]

    return grouped_data

def save_to_markdown(statistics, fp16=False):
    grouped_data = group_statistics(statistics)
    if len(grouped_data) == 0:
        return
    file_name_ending = "_FP16" if fp16 else '';
    with open(f"MIGRAPHX{file_name_ending}.md", "w") as f:
        for group, records in grouped_data.items():
            f.write(f"## {group}\n\n")
            table = markdown_table(records).set_params(row_sep = 'markdown', quote = False).get_markdown()
            f.write(f"{table}\n")