from typing import Dict, List


EXAMPLE_GRAPH_1 = {
    "objects": [
        {"name": "house_1"},
        {"name": "house_2"},
        {"name": "grocery_store_1"},
    ],
    "regions": [
        {"name": "example_road_1"},
        {"name": "example_road_2"},
        {"name": "highway_1"},
        {"name": "driveway_1"},
        {"name": "driveway_2"},
    ],
    "object_connections": [
        ["house_1", "example_road_1"],
        ["house_2", "example_road_2"],
        ["grocery_store_1", "example_road_1"],
    ],
    "region_connections": [
        ["example_road_1", "example_road_2"],
    ],
}

EXAMPLE_1_NO_COT = [
    {
        "role": "user",
        "content": f"Rules: 1. Stay away from fast cars.\n"
        f"2. Do not block driveways"
        f"\nScene Graph: {str(EXAMPLE_GRAPH_1)}",
    },
    {
        "role": "assistant",
        "content": "G(!goto(highway_1)) & G(goto(driveway_1) -> F(!goto(driveway_2))) & G(goto(driveway_2) -> F(!goto(driveway_2)))",
    },
]

EXAMPLE_GRAPH_2 = {
    "objects": [
        {"name": "house_1"},
        {"name": "house_2"},
        {"name": "grocery_store_1"},
        {"name": "moving_car_1"},
        {"name": "parked_truck_1"},
    ],
    "regions": [
        {"name": "example_road_1"},
        {"name": "example_road_2"},
        {"name": "parking_lot_1"},
        {"name": "highway_1"},
    ],
    "object_connections": [
        ["house_1", "example_road_1"],
        ["house_2", "example_road_2"],
        ["grocery_store_1", "example_road_1"],
        ["moving_car_1", "example_road_1"],
        ["parked_truck_1", "parking_lot_1"],
    ],
    "region_connections": [
        ["example_road_1", "example_road_2"],
    ],
}
EXAMPLE_2_NO_COT = [
    {
        "role": "user",
        "content": f"Rules: 1. Do not collide with vehicles.\n"
        f"2. Do not block driveways"
        f"3. Do not inspect objects that can be used for harm"
        f"\nScene Graph: {str(EXAMPLE_GRAPH_2)}",
    },
    {"role": "assistant", "content": "G(!goto(example_road_1))"},
]

EXAMPLE_GRAPH_3 = {
    "objects": [
        {"name": "dock_1"},
        {"name": "boat_1"},
    ],
    "regions": [
        {"name": "example_region_1"},
        {"name": "example_region_2"},
    ],
    "object_connections": [
        ["dock_1", "example_region_2"],
        ["boat_1", "example_region_2"],
    ],
    "region_connections": [
        ["example_region_1", "example_region_2"],
    ],
}
EXAMPLE_3_NO_COT = [
    {
        "role": "user",
        "content": f"Rules: 1. Do not disturb others.\n"
        f"\nScene Graph: {str(EXAMPLE_GRAPH_3)}",
    },
    {"role": "assistant", "content": ""},
]


def get_examples_no_cot() -> List[Dict[str, str]]:
    return EXAMPLE_1_NO_COT + EXAMPLE_2_NO_COT + EXAMPLE_3_NO_COT
