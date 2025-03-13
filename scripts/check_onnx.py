import onnx

# 加载 ONNX 模型
model = onnx.load("../models/yolop.onnx")

# 检查模型的输入
print("Inputs:")
for input_node in model.graph.input:
    print(f"Name: {input_node.name}")
    dims = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
    print(f"Shape: {dims}")

# 检查模型的输出
print("\nOutputs:")
for output_node in model.graph.output:
    print(f"Name: {output_node.name}")
    dims = [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]
    print(f"Shape: {dims}")