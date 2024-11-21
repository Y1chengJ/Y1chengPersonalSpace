import numpy as np
import tvm 
from tvm.ir.module import IRModule
from tvm.script import tir as T


dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
# a @ b is equivalent to np.matmul(a, b)
c_mm_relu = np.maximum(a_np @ b_np, 0)


'''
This is a low-level realization of matrix multiplication with ReLU activation function.
'''
def numpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    Y = np.empty((128, 128), dtype="float32")

    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]

    for i in range(128):
        for j in range(128):
            Y[i, j] = max(Y[i, j], 0)


'''
Use TVM to realize the same function.
'''
@tvm.script.ir_module # indicate this is an IRModule
class MyModule:
    @T.prim_func # indicate this is a primitive tensor function
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"]) -> None:
        T.func_attr({"global_symbol": "mm_relu", "tir.noalis": True})
        '''
        global_symbol: The name of the function
        tir.noalias: All the buffer memories do not overlap
        '''
        Y = T.alloc_buffer((128, 128), "float32")

        for i, j, k in T.grid(128, 128, 128):
            with T.block('Y'):
                '''
                vi, vj represent the spatial axis, and vk represents the reduction axis.
                In calculation, we specify vi, vj, which is the spatial location of Y[vi, vj], 
                while vk is iterated to calculate the value of Y[vi, vj], doing reduction.
                Spatial axis will show in the output tensor, while reduction axis will not, but it will be used in calculation.
                '''
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k) 
            with T.init():
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

        for i, j in  T.grid(128, 128):
            with T.block('C'):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float(0))


'''
Sugars for block axis binding
'''
@tvm.script.ir_module
class MyModuleWithAxisRemapSugar:
    @T.prim_func
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"]) -> None:
        T.func_attr({"global_symbol": "mm_relu", "tir.noalis": True})
        Y = T.alloc_buffer((128, 128), "float32")

        for i, j, k in T.grid(128, 128, 128):
            with T.block('Y'):
                '''
                vi, vj represent the spatial axis, and vk represents the reduction axis.
                In calculation, we specify vi, vj, which is the spatial location of Y[vi, vj], 
                while vk is iterated to calculate the value of Y[vi, vj], doing reduction.
                Spatial axis will show in the output tensor, while reduction axis will not, but it will be used in calculation.
                '''
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

        for i, j in  T.grid(128, 128):
            with T.block('C'):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float(0))


'''
Multiple functions in one module
'''
@tvm.script.ir_module
class MyModuleWithTwoFunctions:
    @T.prim_func
    def mm(A: T.Buffer[(128, 128), "float32"],
           B: T.Buffer[(128, 128), "float32"],
           Y: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "mm", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

    @T.prim_func
    def relu(A: T.Buffer[(128, 128), "float32"],
             B: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))


'''
Transformation
'''
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            for k in range(128):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v2(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)

sch = tvm.tir.schedule(MyModule)

block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)

j0, j1 = sch.split(j, factors=[None, 4])
sch.reorder(i, j0, k, j1) # make it the same as the numpy version above

block_C = sch.get_block("C", func_name="mm_relu")
sch.reverse_compute_at(block_C, j0)


rt_lib = tvm.build(MyModule, target="llvm")
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.array(np.empty((128, 128), dtype=dtype))

func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)


rt_lib_after = tvm.build(sch.mod, target="llvm")
rt_lib_after(a_nd, b_nd, c_nd)

f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)


from tvm import te
A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")

te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})
MyModuleFromTE.show()
