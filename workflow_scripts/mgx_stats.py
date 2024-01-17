import onnx

def get_opset(model_path):
    model = onnx.load(model_path)
    return model.opset_import[0].version

class MGXRunStats():
    def __init__(self, model_path, compiles=True, valid=True, error=""):
        self.opset = get_opset(model_path)
        self.compiles = compiles
        self.valid = valid
        self.error = error

    def set_invalid(self, error=""):
        self.valid = False
        self.error = error

    def set_not_compiles(self, error=""):
        self.compiles = False
        self.set_invalid(error)
