
from optimum.exporters.onnx import main_export

MODEL_PATH = "models/final_model"
ONNX_PATH = "models/ONNX"

if __name__ == "__main__":
    main_export(MODEL_PATH, ONNX_PATH, task="text-classification")
    print(f"Model exported to: {ONNX_PATH}")
