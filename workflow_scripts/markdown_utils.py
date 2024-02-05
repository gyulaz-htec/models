from py_markdown_table.markdown_table import markdown_table

ONNX_ZOO_URL_START = "https://github.com/gyulaz-htec/models/tree/migraphx_testing/"

CATEGORY_DICT = {
    "machine_comprehension": "Machine Comprehension",
    "classification": "Image classification",
    "object_detection": "Object Detection & Image Segmentation",
    "body_analysis": "Body, Face & Gesture Analysis",
    "style_transfer": "Image Manipulation",
    "super_resolution": "Image Manipulation",
}


def get_category_from_path(path):
    for k, v in CATEGORY_DICT.items():
        if k in path:
            return v
    raise ValueError(f"No category found for {path}")


def process_statistics(statistics, fp16):
    grouped_data = {}
    passing = 0
    accuracy_issue = 0
    compile_issue = 0
    runtime_issue = 0
    for path, stats in statistics.items():
        if stats.valid:
            passing += 1
        elif "Max diff(s)" in stats.error:
            accuracy_issue += 1
        elif stats.compiles:
            runtime_issue += 1
        else:
            compile_issue += 1
        category = get_category_from_path(path)
        tar_file_name = path.split("/")[-1]
        model = tar_file_name.replace(".tar.gz", "")
        url = f"{ONNX_ZOO_URL_START}{path}"
        link = f"[{model}.onnx]({url})"
        compiles = ":green_heart:" if stats.compiles else ":broken_heart:"
        valid = ":green_heart:" if stats.valid else ":broken_heart:"
        repro_command = ""
        if not stats.valid:
            fp16_option = "--fp16" if fp16 else ""
            repro_command = f"python3 workflow_scripts/test_models.py --target migraphx --model {path} {fp16_option}"
        record = {
            "Model": link,
            "Opset": stats.opset,
            "Compilation": compiles,
            "Validation": valid,
            "Error": stats.error,
            "Reproduce step": repro_command,
        }
        if category in grouped_data:
            grouped_data[category].append(record)
        else:
            grouped_data[category] = [record]

    return grouped_data, passing, accuracy_issue, compile_issue, runtime_issue


def save_to_markdown(statistics, fp16=False):
    (
        grouped_data,
        passing,
        accuracy_issue,
        compile_issue,
        runtime_issue,
    ) = process_statistics(statistics, fp16)
    if len(grouped_data) == 0:
        return
    file_name_ending = "_FP16" if fp16 else ""
    with open(f"MIGRAPHX_REF{file_name_ending}.md", "w") as f:
        for group, records in sorted(grouped_data.items()):
            records.sort(key=lambda r: r["Model"])
            f.write(f"## {group}\n\n")
            table = (
                markdown_table(records)
                .set_params(row_sep="markdown", quote=False)
                .get_markdown()
            )
            f.write(f"{table}\n")
        f.write(f"## Satistics:\n")
        f.write(f"Passing models: {passing}\n")
        f.write(f"Accuracy issue(s): {accuracy_issue}\n")
        f.write(f"Compile issue(s): {compile_issue}\n")
        f.write(f"Runtime issue(s): {runtime_issue}\n")
