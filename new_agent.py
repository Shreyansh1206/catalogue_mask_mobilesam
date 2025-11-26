import os

os.environ["OMP_NUM_THREADS"] = "4"
import json
from PIL import ImageDraw
from typing import TypedDict, List, Dict, Any, Optional
import numpy as np

# --- LangGraph & LangChain Imports ---
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Core CV Model Imports (ONLY those necessary) ---
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image as PIL_Image  # Use a specific alias to avoid conflicts

from multi_onnx_inference import segmentation

# --- Google API Setup ---
try:
    # Get the API key from your environment variable
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    print("=" * 50)
    print("⚠️  ERROR: GEMINI_API_KEY environment variable not set.")
    print("=" * 50)
    exit()

# --------------------------------------------------------------------------
# Component 0: Load AI Models Globally
# --------------------------------------------------------------------------

print("Loading core CV models... (This may take a moment)")
try:
    detection_model = YOLO("yolov8n.pt")
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    print("✅ CV Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading CV models: {e}")
    print("Please ensure you have run: pip install ultralytics transformers torch")
    exit()

# --- Initialize the LLM for the Master Planner ---
master_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY
)


# --------------------------------------------------------------------------
# Component 1: The Specialist Tools (Basic Python Functions)
# --------------------------------------------------------------------------


def save_overlay_images(image_path: str, masks: np.ndarray) -> List[str]:
    """
    Saves individual images for each mask, overlaying the mask on the original image.

    Args:
        image_path: Path to the original image.
        masks: Boolean array of shape (N, H, W).

    Returns:
        List of paths to the saved images.
    """
    if masks is None or masks.size == 0:
        return []

    saved_paths = []
    base, ext = os.path.splitext(image_path)

    try:
        original_image = PIL_Image.open(image_path).convert("RGBA")

        for i in range(masks.shape[0]):
            mask = masks[i]

            # Create a red overlay with alpha
            overlay = PIL_Image.new("RGBA", original_image.size, (255, 0, 0, 0))

            # Create a mask image for the alpha channel
            # 128 is the transparency level (0-255)
            mask_uint8 = (mask * 128).astype(np.uint8)
            mask_pil = PIL_Image.fromarray(mask_uint8, mode="L")

            # Paste red color using the mask for alpha
            overlay.paste((255, 0, 0, 128), (0, 0), mask_pil)

            # Composite
            combined = PIL_Image.alpha_composite(original_image, overlay)

            output_path = f"{base}_seg_{i}{ext}"
            combined.save(output_path)
            saved_paths.append(output_path)
            print(f"Saved overlay image: {output_path}")

    except Exception as e:
        print(f"Error saving overlay images for {image_path}: {e}")

    return saved_paths


def apply_segmentation_mask(
    image_path: str, box_xyxy: list[int], color: str = "red"
) -> List[str]:
    """
    Applies a segmentation mask over an image based on a bounding box.
    Returns:
        List[str]: List of paths to saved overlay images.
    """
    print(
        f"\n--- [Tool Log] Calling Segmentation for '{image_path}' over box {box_xyxy}..."
    )

    masks = segmentation(image_path, box_xyxy)
    saved_paths = save_overlay_images(image_path, masks)
    return saved_paths


# --------------------------------------------------------------------------
# Component 2: The LangGraph State
# --------------------------------------------------------------------------


class GraphState(TypedDict):
    input_images: List[str]
    user_prompt: str
    metadata: Dict[str, Any]  # Map image_path -> metadata
    plan: Optional[List[Dict]]  # This will be the high-level plan
    log: List[str]
    results: Dict[str, Any]  # Map input_path -> List[str] (paths to segmented images)


# --------------------------------------------------------------------------
# Component 3: The Sub-Agents (Nodes)
# --------------------------------------------------------------------------


def vision_agent_node(state: GraphState) -> Dict[str, Any]:
    """Sub-Agent 1: Runs all CV models to generate metadata for ALL images."""
    print("--- [Node] 👁️ Vision Agent Activated ---")

    all_metadata = {}
    log_updates = []

    for image_path in state["input_images"]:
        print(f"... processing '{image_path}' ...")
        try:
            raw_image = PIL_Image.open(image_path).convert("RGB")

            # 1. OBJECT DETECTION (YOLOv8)
            results = detection_model.predict(image_path, verbose=False)
            objects_list = []
            if results:
                result = results[0]
                class_names = result.names
                for box in result.boxes:
                    coords = box.xyxy.tolist()[0]
                    x, y, x_max, y_max = [round(c) for c in coords]
                    width = x_max - x
                    height = y_max - y
                    class_id = int(box.cls[0])
                    label = class_names[class_id]
                    objects_list.append(
                        {
                            "label": label,
                            "box_xyxy": [x, y, x_max, y_max],
                            "box_xywh": [x, y, width, height],
                        }
                    )

            # 2. IMAGE CAPTIONING (BLIP)
            inputs = caption_processor(raw_image, return_tensors="pt")
            out = caption_model.generate(**inputs, max_new_tokens=50)
            caption_text = caption_processor.decode(out[0], skip_special_tokens=True)

            # 3. ASSEMBLE METADATA
            all_metadata[image_path] = {
                "objects": objects_list,
                "caption": caption_text,
            }
            log_updates.append(f"Analyzed '{image_path}'.")

        except Exception as e:
            print(f"❌ Error processing '{image_path}': {e}")
            log_updates.append(f"Error analyzing '{image_path}': {e}")

    print(f"--- [Vision Agent] Metadata Generated for {len(all_metadata)} images.")

    return {"metadata": all_metadata, "log": state["log"] + log_updates}


def segmentation_agent_node(state: GraphState) -> Dict[str, Any]:
    """Sub-Agent 4: Specializes in segmentation."""
    print("--- [Node] 🖼️ Segmentation Agent Activated ---")
    log = state["log"]
    results = state.get("results", {})

    try:
        task = state["plan"][0]
        label_to_find = task["params"]["label"]
        color = task["params"].get("color", "red")
        image_path = task["params"]["image_path"]

        if image_path not in state["metadata"]:
            return {"log": log + [f"Error: No metadata for '{image_path}'."]}

        # Find the object in metadata. We need the 'box_xyxy' format.
        boxes_to_segment = []
        for obj in state["metadata"][image_path]["objects"]:
            if obj["label"] == label_to_find:
                boxes_to_segment.append(obj["box_xyxy"])

        if not boxes_to_segment:
            print(
                f"--- [Segmentation Agent Error] Could not find object: '{label_to_find}' in {image_path}"
            )
            return {
                "log": log
                + [
                    f"Error: Segmentation agent could not find '{label_to_find}' in {image_path}."
                ]
            }

        # Call the actual black-box tool
        # Note: We don't need 'current_path' chaining for masks as we return raw masks from original image
        saved_paths = apply_segmentation_mask(image_path, boxes_to_segment, color)
        results[image_path] = saved_paths

        new_plan = state["plan"][1:]

        return {
            "results": results,
            "plan": new_plan,
            "log": log
            + [
                f"Segmentation Agent generated {len(saved_paths)} overlay images for '{label_to_find}' in {image_path}."
            ],
        }

    except Exception as e:
        print(f"--- [Segmentation Agent Error] {e}")
        return {"log": log + [f"Error in Segmentation Agent: {e}"]}


# --------------------------------------------------------------------------
# Component 4: The Master Agent (Planner & Router)
# --------------------------------------------------------------------------


def master_planner_node(state: GraphState) -> Dict[str, Any]:
    """The Master Agent. It receives only text and delegates tasks."""
    print("--- [Node] 🧠 Master Planner Activated ---")

    prompt = f"""
    You are an expert AI photo editor orchestrator.
    Your job is to create a high-level plan to fulfill the user's request,
    based *only* on the text prompt and the image metadata.
    
    **User's Goal:**
    "{state["user_prompt"]}"
    
    **Image Metadata (for multiple images):**
    {json.dumps(state["metadata"], indent=2)}
    
    **Available Sub-Agents (Tasks):**
    1. `segmentation_agent`: Use this to apply a segmentation mask. Must specify 'label' and 'image_path'.
        Example: {{"task": "segmentation_agent", "params": {{"label": "person", "color": "green", "image_path": "path/to/img1.jpg"}}}}
    
    **Instructions:**
    - If the user asks to apply an effect to "all images" or "the object", you must generate a separate task for EACH image that contains the relevant object.
    - Check the metadata for each image to see if the object exists before creating a task for it.
    - Respond *only* with a valid JSON list of tasks. Do not include any other text.
    """

    response = master_llm.invoke(prompt)
    response_text = response.content

    print(f"--- [Master Planner] Raw response: {response_text}")

    # Clean up markdown code blocks if present
    if "```json" in response_text:
        response_text = response_text.replace("```json", "").replace("```", "")
    elif "```" in response_text:
        response_text = response_text.replace("```", "")

    response_text = response_text.strip()

    try:
        plan_list = json.loads(response_text)
        print(
            f"--- [Master Planner] Generated Plan:\n{json.dumps(plan_list, indent=2)}"
        )
    except json.JSONDecodeError as e:
        print(f"--- [Master Planner Error] Failed to decode LLM response: {e}")
        return {
            "plan": [],
            "log": state["log"] + ["Error: Failed to parse plan from LLM."],
        }

    return {
        "plan": plan_list,
        "log": state["log"] + ["Master Planner created a new plan."],
    }


# --- The Router (Conditional Edges) ---


def master_router(state: GraphState) -> str:
    """The main router. It checks the state and decides where to go next."""
    print("--- [Router] 🚦 Master Router Activated ---")

    if state.get("metadata") is None:
        print("--- [Router] Metadata is missing. Routing to Vision Agent.")
        return "call_vision_agent"

    if state.get("plan") is None or len(state["plan"]) == 0:
        if state.get("plan") is None:
            print("--- [Router] Metadata present, no plan. Routing to Master Planner.")
            return "call_master_planner"
        if len(state["plan"]) == 0:
            print("--- [Router] Plan is empty. Ending workflow.")
            return "end"

    next_task = state["plan"][0]["task"]
    print(f"--- [Router] Next task is '{next_task}'. Routing...")

    if next_task == "segmentation_agent":
        return "call_segmentation_agent"
    else:
        print(f"--- [Router] Unknown task '{next_task}'. Ending.")
        return "end"


# --------------------------------------------------------------------------
# Component 5: The LangGraph Definition
# --------------------------------------------------------------------------


def passthrough_node(state: GraphState) -> Dict[str, Any]:
    """A simple node that just passes the state through."""
    return {}


workflow = StateGraph(GraphState)

# 1. Add all the nodes (agents)
workflow.add_node("master_planner", master_planner_node)
workflow.add_node("vision_agent", vision_agent_node)
workflow.add_node("segmentation_agent", segmentation_agent_node)

workflow.add_node("master_router", passthrough_node)

# 2. Define the edges (the flow)
workflow.set_entry_point("master_router")

workflow.add_conditional_edges(
    "master_router",
    master_router,
    {
        "call_vision_agent": "vision_agent",
        "call_master_planner": "master_planner",
        "call_segmentation_agent": "segmentation_agent",
        "end": END,
    },
)

workflow.add_edge("vision_agent", "master_router")
workflow.add_edge("master_planner", "master_router")
workflow.add_edge("segmentation_agent", "master_router")

# 3. Compile the graph
app = workflow.compile()

# --------------------------------------------------------------------------
# Component 6: Public Interface
# --------------------------------------------------------------------------


def segment_catalogue(image_paths: List[str], prompt: str) -> List[str]:
    """
    Segments objects in a list of images based on a prompt.

    Args:
        image_paths: List of file paths to images.
        prompt: Description of what to segment (e.g., "Segment the dog").

    Returns:
        List of file paths to the segmented images.
    """
    print("=" * 50)
    print("🚀 Starting Segmentation Catalogue Workflow 🚀")
    print(f"Images: {image_paths}")
    print(f"Prompt: {prompt}")
    print("=" * 50)

    inputs = {
        "input_images": image_paths,
        "user_prompt": prompt,
        "log": [],
        "metadata": None,
        "plan": None,
        "results": {},
    }

    final_state = app.invoke(inputs)

    all_output_paths = []
    for path in image_paths:
        # Get the list of result paths for this image
        result_paths = final_state["results"].get(path)
        if result_paths:
            all_output_paths.extend(result_paths)

    print("=" * 50)
    print("✅ Workflow Complete.")
    print("=" * 50)

    return all_output_paths


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Create dummy images for testing
    image_files = [
        "assets/mobileSam_test.jpg",
        "assets/doggy_test.jpg",
        "assets/multi_doggy_test.png",
    ]

    if not os.path.exists("assets"):
        os.makedirs("assets")

    for i, img_file in enumerate(image_files):
        if not os.path.exists(img_file):
            print(f"Creating dummy test image: {img_file}")
            dummy_img = PIL_Image.new(
                "RGB", (800, 600), color="lightblue" if i == 0 else "lightgreen"
            )
            draw = ImageDraw.Draw(dummy_img)
            # Draw a 'person' (red box) in both
            draw.rectangle([100, 200, 300, 400], fill="red", outline="black")
            dummy_img.save(img_file)

    # Example usage
    prompt = "Segment all the dogs in all images"

    output_images = segment_catalogue(image_files, prompt)

    print("\n--- Final Result Images ---")
    for i, img_path in enumerate(output_images):
        print(f"Displaying image {i + 1}/{len(output_images)}: {img_path}")
        try:
            img = PIL_Image.open(img_path)
            img.show()
            input("Press Enter to view the next image...")
        except Exception as e:
            print(f"Could not display {img_path}: {e}")
