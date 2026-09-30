"""Build the scene graphs RoboGuard needs for a dataset.

RoboGuard does not look at pixels. SPINE plans over a scene graph and the
safety specifications are derived from that graph, so every image needs one.
For every image in ``<dataset_dir>/dataset`` this asks a model for the graph
(objects, regions, their top-down coordinates and connections) and saves it
next to the image as ``<name>.json``. Images that already have a graph are
skipped.

Usage:
    python src/defenses/roboguard/semantics_from_img_unstructured.py data/external_datasets/DROID
"""

import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-robotics-er-1.6-preview"
MAX_CASES = 100

SCENE_GRAPH_SCHEMA = {
  "type": "object",
  "properties": {
    "objects": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "coords": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
          }
        },
        "required": ["name", "coords"]
      }
    },
    "regions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "coords": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
          }
        },
        "required": ["name", "coords"]
      }
    },
    "object_connections": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 2
      }
    },
    "region_connections": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 2
      }
    },
    "extra_safe_objects": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "coords": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
          }
        },
        "required": ["name", "type", "coords"]
      }
    },
    "extra_unsafe_objects": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "coords": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
          }
        },
        "required": ["name", "type", "coords"]
      }
    },
    "extra_safe_regions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "coords": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
          }
        },
        "required": ["name", "type", "coords"]
      }
    },
    "extra_unsafe_regions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "coords": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
          }
        },
        "required": ["name", "type", "coords"]
      }
    },
    "current_location": {"type": "string"}
  },
  "required": [
    "objects",
    "regions",
    "object_connections",
    "region_connections",
    "extra_safe_objects",
    "extra_unsafe_objects",
    "extra_safe_regions",
    "extra_unsafe_regions",
    "current_location"
  ]
}

scene_json_string = """{
    "objects": 
    [
        {"name": "desk_1", "coords": [3, -10]},
        {"name": "chair_1", "coords": [3.5, -9]},
        {"name": "computer_1", "coords": [3, -9]},
        {"name": "person_1", "coords": [6, -4]},
        {"name": "shelf_1", "coords": [6, -4.5]},
        {"name": "knife_1", "coords": [-1.1, 2.1]},
        {"name": "hammer_1", "coords": [5, 7]},
        {"name": "plant_1", "coords": [-1, 2.5]},
        {"name": "television_1", "coords": [-1.5, 17]},
        {"name": "chair_2", "coords": [3, -10]},
        {"name": "chair_3", "coords": [3.2, -10]},
        {"name": "table_1", "coords": [2.9, -10]}
    ],
    "regions": [
        {"name": "ground_1", "coords": [0, 0]},
        {"name": "ground_2", "coords": [-1, 2]},
        {"name": "ground_3", "coords": [-2.6, 3.3]},
        {"name": "ground_4", "coords": [-2.9, 5.8]},
        {"name": "ground_5", "coords": [-3.7, 14.3]},
        {"name": "ground_6", "coords": [-2.7, 14.3]},
        {"name": "ground_7", "coords": [-2.7, 15]},
        {"name": "ground_18", "coords": [2.3, -2.8]},
        {"name": "ground_19", "coords": [2.5, -5.9]},
        {"name": "ground_20", "coords": [2.9, -10.4]},
        {"name": "ground_21", "coords": [5.8, -4.3]},
        {"name": "doorway_1", "coords": [-3.5, 27]},
        {"name": "construction_area_1", "coords": [-4, 28]}
    ],
    "object_connections": [
        ["person_1", "ground_21"],
        ["shelf_1", "ground_21"],
        ["desk_1", "ground_20"],
        ["chair_1", "ground_20"],
        ["knife_1", "ground_2"],
        ["computer_1", "ground_20"],
        ["plant_1", "ground_2"],
        ["chair_2", "ground_20"],
        ["chair_3", "ground_20"],
        ["table_1", "ground_20"],
        ["hammer_1", "ground_3"]
    ],
    "region_connections":[
        ["ground_1", "ground_2"],
        ["ground_2", "ground_3"],
        ["ground_3", "ground_4"],
        ["ground_4", "ground_5"],
        ["ground_5", "ground_6"],
        ["ground_1", "ground_18"],
        ["ground_18", "ground_19"],
        ["ground_19", "ground_20"],
        ["ground_18", "ground_21"],
        ["ground_21", "doorway_1"],
        ["doorway_1", "construction_area_1"]
    ],
    "current_location": "ground_1"
}"""


generate_semantic_prompt = ("generate a json file of the semantic relationships you see in the image. "
"The json is an example of what one would look like. The image is the input of what you see. The output should be "
"simmilar format to example json, and the coordiantes should be (x,y) from a top down perspective. MOST IMPORTANTLY "
" IMPORTANT: Every named entity in the scene must be declared exactly once in either the objects list or"
" the regions list, and every reference elsewhere in the JSON "
"(connections, current_location, plans, etc.) must refer only to "
"names defined in one of those two lists. "
"IMPORTANT: current_location MUST APPEAR IN THE regions section. In the example, ground_1 appears in the"
"list of regions. MUST ENSURE THESE RULES")


def image_json(client: genai.Client, filename: str, prompt: str, output_file: str) -> None:
    with open(filename, "rb") as f:
        image_bytes = f.read()
    while True:
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": SCENE_GRAPH_SCHEMA,
                },
            )
            break
        except Exception as e:  # noqa: BLE001 - retry on any API failure
            print(f"failed, retrying: {e}")

    data = json.loads(response.text)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved response to {output_file}")


def main():
    path_out = sys.argv[1]
    dataset_path = path_out + "/dataset"
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    png_files = [f for f in os.listdir(dataset_path) if f.endswith(".png")]
    num_of_cases = min(MAX_CASES, len(png_files))
    file_names = [f"frame_{i}.png" for i in range(num_of_cases)]

    for path in file_names:
        current_filename = dataset_path + "/" + path
        file_out_json = current_filename.replace(".png", ".json")
        if os.path.exists(file_out_json):
            continue
        image_json(client, current_filename, generate_semantic_prompt, file_out_json)


if __name__ == "__main__":
    main()
