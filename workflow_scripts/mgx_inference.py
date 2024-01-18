import migraphx as mgx
import numpy as np
import os

def load_mgx_model(path, fp16, shapes={}):
    file = path.replace(".onnx", "")
    print(f"Loading model from {file}")
    if os.path.isfile(f"{file}.onnx"):
        print("Parsing from onnx file...")
        prog = mgx.parse_onnx(f"{file}.onnx", map_input_dims=shapes)
        if fp16:
            mgx.quantize_fp16(prog)
        prog.compile(mgx.get_target("gpu"))
    else:
        raise ValueError(f"No .onnx file found on path {path}.")
    return prog

class MGXSession():
    def __init__(self, path, fp16):
        self.path = path
        self.fp16 = fp16
        self.program = load_mgx_model(path, fp16)

    def get_inputs(self):
        inputs = [v for k, v in self.program.get_parameter_shapes().items()]
        return inputs


    def get_outputs(self):
        outputs = self.program.get_output_shapes()
        return outputs

    def run(self, inputs):
        expected = [(k, v.lens()) for k, v  in self.program.get_parameter_shapes().items()]
        expected.sort(key=lambda a: a[0])
        actual = [(k, list(v.shape)) for k, v in inputs.items()]
        actual.sort(key=lambda a: a[0])
        if (actual != expected):
            print(f"Actual input shape differs from expected, reloading model with correct shapes")
            self.program = load_mgx_model(self.path, self.fp16, dict(actual))
        print("Running migraphx inference")
        return self.program.run(inputs)

def session(path, fp16):
    return MGXSession(path, fp16)