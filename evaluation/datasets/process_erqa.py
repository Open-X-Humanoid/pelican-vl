import os
import json
import tensorflow as tf
from PIL import Image
import io
import numpy as np


def parse_example(example_proto):
    """Parse a TFRecord example containing question, image, answer, and metadata."""
    feature_description = {
        'answer': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'question_type': tf.io.VarLenFeature(tf.string),
        'visual_indices': tf.io.VarLenFeature(tf.int64),
        'question': tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Convert sparse tensors to dense tensors
    parsed_features['visual_indices'] = tf.sparse.to_dense(parsed_features['visual_indices'])
    parsed_features['image/encoded'] = tf.sparse.to_dense(parsed_features['image/encoded'])
    parsed_features['question_type'] = tf.sparse.to_dense(parsed_features['question_type'])

    return parsed_features

def tensor_to_pil(image_tensor):
    """Convert a TensorFlow image tensor to a PIL Image."""
    if isinstance(image_tensor, bytes):
        return Image.open(io.BytesIO(image_tensor))
    else:
        # If it's a numpy array
        return Image.fromarray(image_tensor.astype('uint8'))

def save_erqa_dataset(dataset, output_dir="ERQA"):
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "erqa.jsonl")
    count = 0

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for i, example in enumerate(dataset):
            answer = example['answer'].numpy().decode('utf-8')
            question = example['question'].numpy().decode('utf-8')
            question_type = example['question_type'][0].numpy().decode('utf-8') if len(example['question_type']) > 0 else "Unknown"
            visual_indices = example['visual_indices'].numpy().tolist()
            images_encoded = example['image/encoded'].numpy()

            image_paths = []
            for j, img_encoded in enumerate(images_encoded):
                img_pil = tensor_to_pil(tf.io.decode_image(img_encoded).numpy())
                img_filename = f"ex{i+1}_img{j+1}.jpg"
                img_path = os.path.join(image_dir, img_filename)
                img_pil = img_pil.convert("RGB")
                img_pil.save(img_path)
                image_paths.append(os.path.relpath(img_path, output_dir))

            record = {
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "visual_indices": visual_indices,
                "images": image_paths
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} examples...")

    print(f"Saved {count} examples to {jsonl_path}")


def main():
    tfrecord_path = 'erqa.tfrecord'

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    save_erqa_dataset(dataset)

if __name__ == "__main__":
    main()