'''
Descripttion: 
version: 
Author: ***
Date: 2024-06-13 15:52:27
LastEditors: gengyou.lu
LastEditTime: 2024-06-13 16:46:10
'''

from tvm.driver import tvmc


model = tvmc.load('my_model.onnx')

# model.summary()

# tune
# tvmc.tune(model, target="llvm")

package = tvmc.compile(model, target="llvm")

result = tvmc.run(package, device="cpu")

print(result)



