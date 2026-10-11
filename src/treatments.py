"""Treatment and management guidance for detected crop diseases.

Maps PlantVillage class names to practical management advice. This is general
agronomic guidance for educational use, not a substitute for a local extension
service or agronomist. Always follow product labels and local regulations when
applying any pesticide or fungicide.
"""
from __future__ import annotations

from typing import Dict, List, TypedDict


class Treatment(TypedDict):
    summary: str
    cause: str
    steps: List[str]


# Advice keyed by the exact PlantVillage class folder name.
TREATMENTS: Dict[str, Treatment] = {
    "Tomato___healthy": {
        "summary": "No disease detected. Keep up good preventive practices.",
        "cause": "Healthy plant tissue.",
        "steps": [
            "Water at the base in the morning to keep foliage dry.",
            "Space plants for airflow and stake or cage for support.",
            "Scout weekly for early signs of spots, wilting, or pests.",
            "Rotate crops and remove plant debris at the end of the season.",
        ],
    },
    "Tomato___Bacterial_spot": {
        "summary": "Bacterial spot (Xanthomonas). Spreads fast in warm, wet conditions.",
        "cause": "Bacteria spread by rain splash, overhead irrigation, and handling wet plants.",
        "steps": [
            "Remove and destroy infected leaves and fruit; do not compost.",
            "Avoid overhead watering; keep foliage dry and work plants only when dry.",
            "Apply copper-based bactericides early, per label; rotate modes of action.",
            "Use certified disease-free seed and resistant varieties next season.",
            "Rotate away from tomato/pepper for 2-3 years.",
        ],
    },
    "Tomato___Early_blight": {
        "summary": "Early blight (Alternaria). Target-like rings on lower, older leaves.",
        "cause": "Fungus favored by warm, humid weather; survives in soil and debris.",
        "steps": [
            "Remove affected lower leaves and improve air circulation.",
            "Mulch to stop soil splash onto leaves.",
            "Apply fungicides (chlorothalonil or copper; mancozeb) on a schedule in wet weather.",
            "Avoid overhead irrigation; water at the base.",
            "Rotate crops and choose resistant/tolerant varieties.",
        ],
    },
    "Tomato___Late_blight": {
        "summary": "Late blight (Phytophthora infestans). Aggressive; can destroy a crop quickly.",
        "cause": "Water mold that spreads rapidly in cool, wet, humid conditions.",
        "steps": [
            "Act immediately — remove and destroy infected plants (bag, do not compost).",
            "Apply protectant fungicides preventively (chlorothalonil, mancozeb) before symptoms in high-risk weather.",
            "Keep foliage dry; avoid overhead watering and improve spacing.",
            "Never save seed or tubers from infected plants.",
            "Watch regional late-blight alerts and plant resistant varieties.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "summary": "Leaf mold (Passalora fulva). Common in humid, enclosed settings like greenhouses.",
        "cause": "Fungus thriving in high humidity (>85%) with poor air movement.",
        "steps": [
            "Lower humidity: ventilate, space plants, and prune for airflow.",
            "Avoid wetting leaves; water at the base early in the day.",
            "Remove and destroy infected leaves promptly.",
            "Apply labeled fungicides if it progresses; rotate modes of action.",
            "Grow resistant cultivars in greenhouses.",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "summary": "Septoria leaf spot. Many small circular spots with dark margins on lower leaves.",
        "cause": "Fungus spread by splashing water; survives on debris and weeds.",
        "steps": [
            "Remove infected leaves and all plant debris; control nearby weeds.",
            "Mulch and water at the base to reduce splash.",
            "Apply fungicides (chlorothalonil, copper, or mancozeb) on a 7-10 day schedule in wet weather.",
            "Improve spacing and airflow; stake plants.",
            "Rotate crops for 2-3 years.",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "summary": "Two-spotted spider mites (a pest, not a disease). Stippling and fine webbing on leaves.",
        "cause": "Tiny mites that thrive in hot, dry, dusty conditions.",
        "steps": [
            "Spray plants with a strong jet of water to dislodge mites.",
            "Apply insecticidal soap or horticultural oil, covering leaf undersides.",
            "Introduce predatory mites as biological control.",
            "Keep plants well-watered and avoid dust and over-fertilizing with nitrogen.",
            "Use miticides only if needed and rotate chemistries to avoid resistance.",
        ],
    },
    "Tomato___Target_Spot": {
        "summary": "Target spot (Corynespora). Concentric-ringed lesions on leaves and fruit.",
        "cause": "Fungus favored by warm, humid, wet conditions.",
        "steps": [
            "Remove infected leaves and fruit; clear debris after harvest.",
            "Improve airflow with spacing and pruning; avoid overhead watering.",
            "Apply labeled fungicides preventively in humid weather; rotate modes of action.",
            "Rotate crops and avoid working plants when wet.",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "summary": "Tomato Yellow Leaf Curl Virus (TYLCV). Upward leaf curling, yellowing, stunting. No cure.",
        "cause": "Virus transmitted by whiteflies; cannot be cured once a plant is infected.",
        "steps": [
            "Remove and destroy infected plants to limit spread.",
            "Control whiteflies: reflective mulch, insecticidal soap/oil, and labeled insecticides.",
            "Use physical barriers (insect netting) and manage weeds that host whiteflies.",
            "Plant TYLCV-resistant varieties and virus-free transplants.",
            "Avoid planting next to older infected crops.",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "summary": "Tomato mosaic virus (ToMV). Mottled light/dark leaves, distortion. No cure.",
        "cause": "Very stable virus spread by handling, tools, seed, and plant debris.",
        "steps": [
            "Remove and destroy infected plants; do not compost.",
            "Wash hands and disinfect tools (e.g., with a bleach solution) between plants.",
            "Avoid using tobacco products near plants and don't handle plants after smoking.",
            "Use certified virus-free seed and resistant varieties.",
            "Control weeds and rotate crops.",
        ],
    },
    "Potato___healthy": {
        "summary": "No disease detected. Maintain good preventive practices.",
        "cause": "Healthy plant tissue.",
        "steps": [
            "Use certified disease-free seed potatoes.",
            "Hill soil over tubers and water consistently at the base.",
            "Scout regularly and rotate crops yearly.",
            "Remove volunteers and debris after harvest.",
        ],
    },
    "Potato___Late_blight": {
        "summary": "Late blight (Phytophthora infestans). The disease behind the Irish famine; acts fast.",
        "cause": "Water mold spreading rapidly in cool, wet, humid weather.",
        "steps": [
            "Remove and destroy infected foliage and tubers immediately.",
            "Apply protectant fungicides (chlorothalonil, mancozeb) preventively in high-risk weather.",
            "Hill soil well to protect tubers; harvest in dry conditions.",
            "Use certified seed potatoes and resistant varieties.",
            "Follow regional late-blight forecasts and destroy cull piles/volunteers.",
        ],
    },
    "Potato___Early_blight": {
        "summary": "Early blight (Alternaria). Dark target-like spots on older leaves.",
        "cause": "Fungus favored by warm, humid weather and stressed plants.",
        "steps": [
            "Remove affected lower leaves and plant debris.",
            "Keep plants vigorous with balanced fertility and consistent water at the base.",
            "Apply fungicides (chlorothalonil or mancozeb) on a schedule in wet weather.",
            "Rotate crops and use certified seed.",
        ],
    },
}

_DEFAULT: Treatment = {
    "summary": "No specific guidance available for this class.",
    "cause": "Unknown.",
    "steps": [
        "Consult your local agricultural extension service for diagnosis and treatment.",
        "Isolate affected plants and avoid spreading soil or debris.",
    ],
}


def get_treatment(class_name: str) -> Treatment:
    """Return treatment guidance for a class name, with a safe default."""
    return TREATMENTS.get(class_name, _DEFAULT)


def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()
