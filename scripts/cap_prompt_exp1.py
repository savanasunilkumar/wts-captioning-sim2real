CAP_PROMPT = """You are a traffic safety analyst. Watch this video segment depicting the {phase_name} phase of a pedestrian-vehicle traffic event.
Provide TWO captions in this exact format:
PEDESTRIAN: Describe the pedestrian completely and specifically - age range and gender; approximate height in cm; clothing on the upper body and the lower body with their colors, and any headwear; the pedestrian's body orientation relative to the vehicle; line of sight / gaze direction; position relative to the vehicle and the distance between them; current action or motion (for example standing still, walking, running, crossing, squatting, turning); whether the pedestrian is aware of the vehicle; and the environment (weather, brightness, road surface, road type).
VEHICLE: Describe the vehicle's position relative to the pedestrian, its field of view of the pedestrian, its action and speed, and the environment.
Output only those two labeled lines."""
