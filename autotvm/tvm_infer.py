'''
Descripttion: 
version: 
Author: ***
Date: 2024-06-13 16:49:57
LastEditors: gengyou.lu
LastEditTime: 2024-06-13 17:27:38
'''

import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import timeit
from scipy.special import softmax

np.random.seed(0)


# 加载模型
model_path = "data/resnet50-v2-7.onnx"
onnx_model = onnx.load(model_path)

# 加载图像
image_path = "data/kitten.jpg"
resized_image = Image.open(image_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")
print(img_data.shape)


''' 图像预处理 '''
img_data = np.transpose(img_data, (2, 0, 1))

imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

img_data = np.expand_dims(norm_img_data, axis=0)


''' 使用 Relay 编译模型 '''
target = "llvm"

input_name = "data"
shape_dict = {input_name: img_data.shape}

# 编译模型
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# 将模型构建到 tvm 库中
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


''' 在 TVM Runtime 执行 '''
dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()


''' 收集基本性能数据 '''
timing_number = 10
timing_repeat = 10

unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}
print(unoptimized)


''' 后处理 '''
label_path = "data/synset.txt"
with open(label_path, 'r') as f:
    labels = [l.rstrip() for l in f]

scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]

for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))


