import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
load_engine_path = "python_API/yolov8m_trt.engine"

def load_engine(load_engine_path, runtime):
    with open(load_engine_path, "rb") as f:
        serialized_engine = f.read()

    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

def do_inference(engine, input_data):
    
    context = engine.create_execution_context()
    input_name= engine.get_tensor_name(0)
    output_name= engine.get_tensor_name(1)

    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    print("Input shape :",input_shape)
    print("Output shape :",output_shape)


    print("Input_itemsize:" , input_data.itemsize)
    input_size = np.prod(input_shape) * input_data.itemsize  # Calculate total bytes for input
    output_size = np.prod(output_shape) * input_data.itemsize  # Calculate total bytes for output

    print("Input_size:", input_size)

    # Allocate memory for inputs and outputs
    d_input = cuda.mem_alloc(int(input_size))
    d_output = cuda.mem_alloc(int(output_size))

    input_data = np.ascontiguousarray(input_data)
    cuda.memcpy_htod(d_input, input_data)

    context.execute_v2([int(d_input), int(d_output)])
    output_data = np.empty(output_shape, dtype= np.float32)
    cuda.memcpy_dtoh(output_data,d_output)

    return output_data

img_path = "sample_images/1200x810.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))  # Resize to model input shape
input_data = np.asarray(img, dtype=np.float32) / 255.0  # Normalize the image
input_data = input_data.transpose((2, 0, 1))  # Change to CHW format
input_data = np.expand_dims(input_data, axis=0) 

runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(load_engine_path, runtime)

output = do_inference(engine, input_data)

print("Output: ", output)






