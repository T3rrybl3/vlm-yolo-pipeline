import sys  # lets python read cmd line args
from pipeline.pipeline import PerceptionPipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    pipeline = PerceptionPipeline()
    result = pipeline.run(sys.argv[1])

    print(
        f'Detections by YOLO: {result["detections"]} \n Results by VLM: {result["people"]}')
