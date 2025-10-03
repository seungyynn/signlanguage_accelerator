import argparse
import os
from collections import deque
from typing import Deque, List

import cv2
import numpy as np
import tensorflow as tf


class TFLiteClassifier:
    def __init__(self, model_path: str, labels_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

        # Determine input size
        _, self.input_h, self.input_w, self.input_c = self.input_details[0]["shape"]
        self.is_int8 = self.input_details[0]["dtype"] == np.int8
        self.is_fp16 = self.input_details[0]["dtype"] == np.float16

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels: List[str] = [line.strip() for line in f if line.strip()]

        # Get quantization params if needed
        self.input_scale = self.input_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
        self.input_zero_point = self.input_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        # Model's first layer rescales to [-1,1], so keep 0..255 here
        if self.is_int8:
            # Quantize to int8 using input scale/zero_point
            img_q = np.round(img / self.input_scale + self.input_zero_point).astype(np.int8)
            return img_q[np.newaxis, ...]
        elif self.is_fp16:
            return img.astype(np.float16)[np.newaxis, ...]
        else:
            return img[np.newaxis, ...]

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)
        if output.dtype == np.int8:
            # Dequantize logits if necessary; however, many TF Lite models output probs float
            output_details = self.output_details[0]
            scale = output_details.get("quantization_parameters", {}).get("scales", [1.0])[0]
            zero_point = output_details.get("quantization_parameters", {}).get("zero_points", [0])[0]
            output = (output.astype(np.float32) - zero_point) * scale
        return output[0]


def draw_overlay(frame: np.ndarray, label: str, prob: float) -> None:
    h, w = frame.shape[:2]
    text = f"{label}: {prob*100:.1f}%"
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Real-time TFLite ASL classifier")
    parser.add_argument("--model", type=str, required=True, help="Path to TFLite model (fp16 or int8)")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.txt")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--smoothing", type=int, default=8, help="Temporal smoothing window size")
    parser.add_argument("--min-prob", type=float, default=0.5, help="Min probability to show label")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    classifier = TFLiteClassifier(args.model, args.labels)
    probs_window: Deque[np.ndarray] = deque(maxlen=args.smoothing)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inp = classifier.preprocess(frame)
        probs = classifier.infer(inp)

        probs_window.append(probs)
        avg_probs = np.mean(probs_window, axis=0)
        top_idx = int(np.argmax(avg_probs))
        top_prob = float(avg_probs[top_idx])
        top_label = classifier.labels[top_idx]

        if top_prob >= args.min_prob:
            draw_overlay(frame, top_label, top_prob)
        else:
            draw_overlay(frame, "…", top_prob)

        cv2.imshow("ASL Alphabet - TFLite", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
