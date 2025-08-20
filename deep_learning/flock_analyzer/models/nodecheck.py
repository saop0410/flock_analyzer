import onnx
model = onnx.load("analyzer_ViT.onnx")


for output in model.graph.output:
    print("Output name:", output.name)

for input_tensor in model.graph.input:
    print("Input name:", input_tensor.name)
    dims = input_tensor.type.tensor_type.shape.dim
    print("Shape:", [d.dim_value for d in dims])