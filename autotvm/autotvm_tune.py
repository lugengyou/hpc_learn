'''
Descripttion: 
version: 
Author: ***
Date: 2024-06-13 19:02:29
LastEditors: gengyou.lu
LastEditTime: 2024-06-13 19:21:26
'''
import onnx
from PIL import Image
import numpy as np
import tvm
from tvm.contrib import graph_executor
import tvm.relay as relay
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
from scipy.special import softmax
import timeit

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


''' 创建 tvm 运行器 '''
number = 10
repeat = 1
min_repeat_ms = 0
timeout = 10

runner = autotvm.LocalRunner(number=number,
                             repeat=repeat,
                             timeout=timeout,
                             min_repeat_ms=min_repeat_ms,
                             enable_cpu_cache_flush=True
                             )

tuning_option = {
    "tunner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "autotvm/resnet-50-v2-autotuning.json",
}

# # 首先从 onnx 模型中提取任务
# tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# # 按顺序调优提取的任务
# for i, task in enumerate(tasks):
#     prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    
#     # choose tuner
#     tuner = "xgb"

#     # create tuner
#     if tuner == "xgb":
#         tuner_obj = XGBTuner(task, loss_type="reg")
#     elif tuner == "xgb_knob":
#         tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
#     elif tuner == "xgb_itervar":
#         tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
#     elif tuner == "xgb_curve":
#         tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
#     elif tuner == "xgb_rank":
#         tuner_obj = XGBTuner(task, loss_type="rank")
#     elif tuner == "xgb_rank_knob":
#         tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
#     elif tuner == "xgb_rank_itervar":
#         tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
#     elif tuner == "xgb_rank_curve":
#         tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
#     elif tuner == "xgb_rank_binary":
#         tuner_obj = XGBTuner(task, loss_type="rank-binary")
#     elif tuner == "xgb_rank_binary_knob":
#         tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
#     elif tuner == "xgb_rank_binary_itervar":
#         tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
#     elif tuner == "xgb_rank_binary_curve":
#         tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
#     elif tuner == "ga":
#         tuner_obj = GATuner(task, pop_size=50)
#     elif tuner == "random":
#         tuner_obj = RandomTuner(task)
#     elif tuner == "gridsearch":
#         tuner_obj = GridSearchTuner(task)
#     else:
#         raise ValueError("Invalid tuner: " + tuner)

#     tuner_obj.tune(
#         n_trial=min(tuning_option["trials"], len(task.config_space)),
#         early_stopping=tuning_option["early_stopping"],
#         measure_option=tuning_option["measure_option"],
#         callbacks=[
#             autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
#             autotvm.callback.log_to_file(tuning_option["tuning_records"]),
#         ],
#     )


''' 使用调优数据编译优化模型 '''
with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


''' 后处理 '''
label_path = "data/synset.txt"
with open(label_path, 'r') as f:
    labels = [l.rstrip() for l in f]

dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))



''' 比较优化和未优化的模型 '''
timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))



