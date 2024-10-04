import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(ONNX_FILE_PATH):

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_FILE_PATH, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model file")
    
    config = builder.create_builder_config()
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_serialized_network(network, config)

    return engine

def save_engine(serialized_engine, engine_file_path):
    with open(engine_file_path,"wb") as f:
        f.write(serialized_engine)

onnx_file_path = "python_API/yolov8m.onnx"
engine_file_path = "python_API/yolov8m_trt.engine"

engine = build_engine(onnx_file_path)
save_engine(engine,engine_file_path)