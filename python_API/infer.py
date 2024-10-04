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

# Allocate buffers for input and output
def allocate_buffers(engine):
    # Get input and output tensor names
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    # Get input/output shapes
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # Host memory for inputs and outputs
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=trt.nptype(trt.float32))
    
    print("Volume of output shape: ",trt.volume(output_shape))
    # Allocate memory on the GPU
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    return input_name, output_name, h_input, d_input, h_output, d_output, stream

def do_inference(engine, input_name, output_name, h_input, d_input, h_output, d_output, stream):
    with engine.create_execution_context() as context:
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))

        # Transfer the input data to the GPU
        cuda.memcpy_htod_async(d_input, h_input, stream)

        # Run inference asynchronously
        context.execute_async_v3(stream_handle=stream.handle)
        
        # Transfer the output data back to the CPU
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        return h_output
    
# After running inference and before your program exits
def cleanup(d_input, d_output, stream):
    # Free GPU memory
    d_input.free()
    d_output.free()
    # Synchronize and clean up the stream
    stream.synchronize()  # Ensure all operations on the stream are finished
    del stream  # Delete the stream if no longer needed

# Preprocess image (resize, normalize, format)
img_path = "sample_images/_101823025_abbeyroadfans.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))  # Resize to model input shape
input_data = np.asarray(img, dtype=np.float32) / 255.0  # Normalize the image
input_data = input_data.transpose((2, 0, 1))  # Change to CHW format
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(load_engine_path, runtime)
input_name, output_name, h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
# Copy input data to GPU memory
np.copyto(h_input, input_data.ravel())  # Flattened array

# Run inference
output = do_inference(engine, input_name, output_name, h_input, d_input, h_output, d_output, stream)
print("Output: ", output)

# Call cleanup after inference
cleanup(d_input, d_output, stream)

with open("output.txt", "w") as f:
    f.write(str(output.tolist()))

print("Output shape",output.shape)

