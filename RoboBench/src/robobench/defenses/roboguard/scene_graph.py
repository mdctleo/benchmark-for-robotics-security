"""Scene graphs for RoboGuard, generated from a scene's images.

RoboGuard does not look at pixels. SPINE plans over a scene graph and the
safety specifications are derived from that graph, so every scene needs one.
``generate_scene_graph`` asks a model for the graph (objects, regions, their
top-down coordinates and connections) and returns it as a dict. A scene is
one or more frames; all of them go into the one request, and the graph
describes the scene they show together.
"""

import json
import os

from ...models import detect_media_type

MODEL = "gemini-robotics-er-1.6-preview"

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

sequence_note = (" The images are frames of the same scene; produce one json describing that scene,"
" including everything seen in any of the frames.")


def make_client():
    from google import genai
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_scene_graph(images: list[bytes], client=None, model: str = MODEL) -> dict:
    """One model call: every frame of the scene, then the prompt, constrained to the schema."""
    from google.genai import types

    if not images:
        raise ValueError("generate_scene_graph needs at least one image")
    prompt = generate_semantic_prompt + (sequence_note if len(images) > 1 else "")
    client = client if client is not None else make_client()
    response = client.models.generate_content(
        model=model,
        contents=[
            *(types.Part.from_bytes(data=raw, mime_type=detect_media_type(raw)) for raw in images),
            prompt,
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": SCENE_GRAPH_SCHEMA,
        },
    )
    return json.loads(response.text)
