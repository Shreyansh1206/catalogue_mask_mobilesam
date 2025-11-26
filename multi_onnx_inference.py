import onnxruntime as ort
import numpy as np
from PIL import Image

# The encoder was exported with a 1024 long-side preprocessing (dummy image 1024x1024).
# Resize to 1024x1024 so the encoder's internal positional embeddings / reshape nodes
# match the exported graph expectations.


from typing import Union, List


def segmentation(
    image_path: str, bounding_box: Union[List[int], List[List[int]]]
) -> np.ndarray:
    """
    Perform image segmentation using MobileSAM ONNX models (encoder + decoder).
    Args:
        image_path (str): Path to the input image file
        bounding_box (list): Single box [x1,y1,x2,y2] or list of boxes [[x1,y1,x2,y2], ...]
    Returns:
        np.ndarray: Boolean array of shape (N, H, W) where N is number of boxes.
                    Returns None if no boxes provided.
    """

    # Load the ONNX models
    encoder_model_path = "./mobile_sam_image_encoder.onnx"
    decoder_model_path = "./mobile_sam.onnx"

    encoder_session = ort.InferenceSession(
        encoder_model_path, providers=["CPUExecutionProvider"]
    )
    decoder_session = ort.InferenceSession(
        decoder_model_path, providers=["CPUExecutionProvider"]
    )

    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(img_pil)
    original_height, original_width = image_np.shape[:2]

    # Resize to 1024x1024 (the export used a 1024x1024 dummy image).
    img_resized = img_pil.resize((1024, 1024), Image.BILINEAR)
    image_resized = np.array(img_resized).astype(np.float32)

    # SAM normalization
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image_resized = (image_resized - mean) / std

    # [1, 3, 1024, 1024]
    image_input = np.transpose(image_resized, (2, 0, 1))[None, :, :, :].astype(
        np.float32
    )

    # Run encoder (ONCE)
    encoder_outs = encoder_session.run(None, {"input_image": image_input})
    image_embeddings = encoder_outs[0]
    # print(
    #     f"Encoder output dtype={image_embeddings.dtype} shape={image_embeddings.shape}"
    # )

    # Ensure image_embeddings is in CHW format expected by the decoder.
    dec_input_meta = decoder_session.get_inputs()[0]
    dec_shape = dec_input_meta.shape

    image_embeddings = image_embeddings.astype(np.float32)

    # Reshape logic if needed (simplified from previous version for brevity, keeping core logic)
    if image_embeddings.ndim == 3 and len(dec_shape) == 4:
        batch, N, C = image_embeddings.shape
        _, C_exp, H_exp, W_exp = dec_shape
        if int(C) == int(C_exp) and int(N) == int(int(H_exp) * int(W_exp)):
            image_embeddings = image_embeddings.reshape(
                batch, int(H_exp), int(W_exp), int(C_exp)
            ).transpose(0, 3, 1, 2)

    # Handle single box vs multiple boxes
    if isinstance(bounding_box[0], int):
        boxes = [bounding_box]
    else:
        boxes = bounding_box

    if not boxes:
        return np.array([], dtype=bool)

    # print(f"Processing {len(boxes)} bounding boxes...")

    collected_masks = []

    for i, box in enumerate(boxes):
        # Normalize bounding box coordinates
        x_min, y_min, x_max, y_max = box
        x_min_norm = (x_min / original_width) * 1024
        y_min_norm = (y_min / original_height) * 1024
        x_max_norm = (x_max / original_width) * 1024
        y_max_norm = (y_max / original_height) * 1024

        # Convert bounding box → corner points in 1024x1024 coords
        point_coords = np.array(
            [[[x_min_norm, y_min_norm], [x_max_norm, y_max_norm]]], dtype=np.float32
        )

        # Pad to [1,5,2]
        padded_point_coords = np.zeros((1, 5, 2), dtype=np.float32)
        padded_point_coords[0, :2, :] = point_coords[0]

        # Point labels: 2 = top-left, 3 = bottom-right, -1 = padding
        point_labels = np.array([[2, 3, -1, -1, -1]], dtype=np.float32)

        # Decoder inputs
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        orig_im_size = np.array([original_height, original_width], dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "point_coords": padded_point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size,
        }

        masks, _, _ = decoder_session.run(None, decoder_inputs)

        # masks shape: [1, 1, H, W]
        mask = masks[0, 0, :, :]
        mask_bool = mask > 0.5
        collected_masks.append(mask_bool)

    # Stack masks into (N, H, W)
    if collected_masks:
        final_masks = np.stack(collected_masks, axis=0)
    else:
        final_masks = np.array([], dtype=bool)

    # print(f"Inference_time: {end_time - start_time}")

    return final_masks


if __name__ == "__main__":
    image_path = "assets/doggy_test.jpg"
    bounding_box = [100, 100, 400, 400]
    segmentation(image_path, bounding_box)
