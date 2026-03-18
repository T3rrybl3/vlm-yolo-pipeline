import sys  # lets python read cmd line args
from pipeline.pipeline import PerceptionPipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_or_video_path|webcam|camera_index>")
        sys.exit(1)

    pipeline = PerceptionPipeline()
    result = pipeline.run(sys.argv[1])

    if result.get("source", "").startswith("webcam:"):
        print(f'Webcam session finished \n Tracked people: {result["people"]}')
    elif "frames" in result:
        print(
            f'Processed {result["total_frames"]} frames at {result["fps"]} FPS \n Tracked people: {result["people"]}')
    else:
        print(
            f'Detections by YOLO: {result["detections"]} \n Results by VLM: {result["people"]}')
