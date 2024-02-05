from py_markdown_table.markdown_table import markdown_table
import json

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


def process_statistics(statistics):
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
        compiles = ":white_check_mark:" if stats.compiles else ":x:"
        valid = ":white_check_mark:" if stats.valid else ":x:"
        record = {
            "Model": link,
            "Opset": stats.opset,
            "Compilation": compiles,
            "Validation": valid,
            "Error": stats.error,
        }
        if category in grouped_data:
            grouped_data[category].append(record)
        else:
            grouped_data[category] = [record]

    return grouped_data, passing, accuracy_issue, compile_issue, runtime_issue


def merge_fp32_fp16_data(fp32_data, fp16_data):
    fp32_fp16_report = {}
    for category, fp32_records in fp32_data.items():
        fp32_fp16_report[category] = []
        fp16_records = fp16_data[category]
        for fp32_record in fp32_records:
            matching_fp16s = [
                r for r in fp16_records if r["Model"] == fp32_record["Model"]
            ]
            has_fp16 = len(matching_fp16s) > 0
            record = {
                "Model": fp32_record["Model"],
                "Opset": fp32_record["Opset"],
                "FP32 Comp": fp32_record["Compilation"],
                "FP16 Comp": (
                    matching_fp16s[0]["Compilation"]
                    if has_fp16
                    else ":heavy_minus_sign:"
                ),
                "FP32 Acc": fp32_record["Validation"],
                "FP16 Acc": (
                    matching_fp16s[0]["Validation"]
                    if has_fp16
                    else ":heavy_minus_sign:"
                ),
                "FP32 Err": fp32_record["Error"],
                "FP16 Err": matching_fp16s[0]["Error"] if has_fp16 else "",
            }
            fp32_fp16_report[category].append(record)
    return fp32_fp16_report


def write_markdown_file(file_name, data, passing=0, accuracy=0, compile=0, runtime=0):
    with open(file_name, "w") as f:
        for category, records in sorted(data.items()):
            records.sort(key=lambda r: r["Model"])
            f.write(f"## {category}\n\n")
            table = (
                markdown_table(records)
                .set_params(row_sep="markdown", quote=False)
                .get_markdown()
            )
            f.write(f"{table}\n")

        if (passing + accuracy + compile + runtime) > 0:
            f.write(f"## Satistics:\n")
            f.write(f"Passing models: {passing}\n")
            f.write(f"Accuracy issue(s): {accuracy}\n")
            f.write(f"Compile issue(s): {compile}\n")
            f.write(f"Runtime issue(s): {runtime}\n")


def save_to_markdown(statistics):
    # with open("bak.json", "w") as bak:
    #     bak.write(json.dumps(statistics))
    fp32_fp16_stats = {}
    for quant, stats in statistics.items():
        (
            grouped_data,
            passing,
            accuracy_issue,
            compile_issue,
            runtime_issue,
        ) = process_statistics(stats)
        if len(grouped_data) == 0:
            continue
        if quant in ["fp32", "fp16"]:
            fp32_fp16_stats[quant] = grouped_data
        write_markdown_file(
            f"MIGRAPHX_ref_{quant}.md",
            grouped_data,
            passing,
            accuracy_issue,
            compile_issue,
            runtime_issue,
        )

    if len(fp32_fp16_stats) == 2:
        fp32_fp16_report = merge_fp32_fp16_data(
            fp32_fp16_stats["fp32"], fp32_fp16_stats["fp16"]
        )
        write_markdown_file(f"MIGRAPHX_ref_FP32_FP16.md", fp32_fp16_report)
