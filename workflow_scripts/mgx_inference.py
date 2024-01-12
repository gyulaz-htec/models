import migraphx as mgx
import numpy as np
import os

def load_mgx_model(path, fp16, shapes={}):
    file = path.replace(".onnx", "")
    print(f"Loading model from {file}")
    if os.path.isfile(f"{file}.mxr"):
        print("Found mxr, loading it...")
        prog = mgx.load(f"{file}.mxr", format="msgpack")
    elif os.path.isfile(f"{file}.onnx"):
        print("Parsing from onnx file...")
        prog = mgx.parse_onnx(f"{file}.onnx", map_input_dims=shapes)
        if fp16:
            mgx.quantize_fp16(prog)
        prog.compile(mgx.get_target("gpu"))
        print(f"Saving model to mxr file...")
        mgx.save(prog, f"{file}.mxr", format="msgpack")
    else:
        raise ValueError(f"No .onnx or .mxr file found. Please download it and re-try.")
    return prog

class MGXSession():
    def __init__(self, path, fp16):
        self.program = load_mgx_model(path, fp16)

    def get_inputs(self):
        inputs = [v for k, v in self.program.get_parameter_shapes().items()]
        return inputs


    def get_outputs(self):
        outputs = self.program.get_output_shapes()
        return outputs

    def run(self, output_names, inputs):
        return self.program.run(inputs)

def session(path, fp16):
    return MGXSession(path, fp16)