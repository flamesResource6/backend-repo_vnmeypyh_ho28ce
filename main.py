import os
import random
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document
from bson import ObjectId  # provided by pymongo

app = FastAPI(title="StoryForge API", description="Dynamic interactive story generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------
# Models
# ----------------------
class StartStoryRequest(BaseModel):
    protagonist: str = Field(..., min_length=1, max_length=60)
    setting: str = Field("Neon Harbor", max_length=80)
    weights: Dict[str, int] = Field(
        default_factory=lambda: {"adventure": 60, "romance": 20, "nightlife": 20}
    )


class Choice(BaseModel):
    id: int
    text: str


class Scene(BaseModel):
    index: int
    text: str
    choices: List[Choice]


class StartStoryResponse(BaseModel):
    story_id: str
    title: str
    scene: Scene


class ChooseRequest(BaseModel):
    story_id: str
    choice_id: int


class ChooseResponse(BaseModel):
    scene: Scene
    is_complete: bool


# ----------------------
# Helpers – lightweight procedural generator (no external AI)
# ----------------------
GEN_MAX_STEPS = 7


def _seed(story_id: str, step: int) -> int:
    # Stable pseudo-random seeded by story and step
    return abs(hash(f"{story_id}:{step}")) % (2**32)


def _weighted_pick(weights: Dict[str, int], rnd: random.Random) -> str:
    total = sum(max(0, v) for v in weights.values()) or 1
    r = rnd.randint(1, total)
    cum = 0
    for k, v in weights.items():
        cum += max(0, v)
        if r <= cum:
            return k
    return list(weights.keys())[0]


def _scene_blueprint(theme: str) -> Dict[str, Any]:
    if theme == "adventure":
        return {
            "openings": [
                "Sodium lights streak past as the hover-cab skims over {setting}. Rumors speak of a hidden vault beneath the old arcade.",
                "A courier bag thumps your side—inside, a map etched on synth-leather. The first mark glows near the waterfront.",
                "The city hums—mag-rails, neon, and whispers. Tonight, a door opens that only the bold will enter.",
            ],
            "beats": [
                "You tail a masked figure into an alley where the walls pulse with holo-graffiti.",
                "A lockbox asks three questions in a language no one remembers.",
                "A drone swarm scatters, revealing a service hatch marked with a silver spade.",
            ],
            "choices": [
                ("Cut through the arcade backrooms", "You slip past stacked cabinets; pixels flicker like fireflies."),
                ("Scale the rain-slick fire escape", "Metal sings underfoot as the city widens below."),
                ("Hack the service hatch", "The panel blinks; you hum the password like a lullaby to machines."),
            ],
        }
    if theme == "romance":
        return {
            "openings": [
                "Tonight the rain tastes like citrus, and you swear the skyline leans closer when {crush} laughs.",
                "The club's velvet bass slows when your eyes meet. Time, suddenly generous, gifts you a second look.",
                "A text arrives: 'Meet me where the city keeps its secrets.' Your pulse agrees before you do.",
            ],
            "beats": [
                "Two glasses fog in the glow of a neon heart.",
                "A rooftop garden soaks up moonlight as if it were applause.",
                "A song you both pretend not to love finds you anyway.",
            ],
            "choices": [
                ("Dance like the future is watching", "You step into rhythm, and the room steps aside."),
                ("Share the umbrella and a secret", "Raindrops turn into punctuation around your laughter."),
                ("Leave a note under the vinyl sleeve", "Ink dries slow; meaning doesn't."),
            ],
        }
    # nightlife
    return {
        "openings": [
            "The queue snakes around the block; the bouncer's smile is a rumor with muscles.",
            "A neon sign buzzes in cursive: 'After'. Inside, time is a circle and you're at the center.",
            "Subway air and perfume mix as the midnight market stretches like a second sky.",
        ],
        "beats": [
            "A DJ scrubs galaxies between two records.",
            "Street food sizzles like letters forming your name.",
            "The bartender deals stories better than cards.",
        ],
        "choices": [
            ("Slip into the secret room behind the mirror", "A corridor of LEDs counts your footsteps."),
            ("Challenge the bartender for a signature mix", "They nod, alchemy in their wrists."),
            ("Follow the saxophone into the blue-lit alley", "Notes curl into constellations above the bricks."),
        ],
    }


def _make_scene(protagonist: str, setting: str, weights: Dict[str, int], step: int, story_id: str) -> Scene:
    rnd = random.Random(_seed(story_id, step))
    theme = _weighted_pick(weights, rnd)
    bp = _scene_blueprint(theme)

    opening = rnd.choice(bp["openings"]).format(setting=setting, crush=protagonist)
    beat = rnd.choice(bp["beats"]) if step > 0 else opening

    # Build text differently for first scene
    if step == 0:
        text = f"{opening}\n\n{protagonist} feels the night lean in, curious."
    else:
        text = f"{beat}"

    # Prepare 2-3 choices
    raw_choices = rnd.sample(bp["choices"], k=2 + rnd.randint(0, 1))
    choices = [Choice(id=i + 1, text=rc[0]) for i, rc in enumerate(raw_choices)]

    return Scene(index=step, text=text, choices=choices)


# ----------------------
# Routes
# ----------------------
@app.get("/")
def index():
    return {"message": "StoryForge backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, "name", None)
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


@app.get("/schema")
def schema_view():
    # Minimal schema export for reference in admin tools
    return {
        "story": {
            "title": "string",
            "protagonist": "string",
            "setting": "string",
            "weights": {"adventure": "int", "romance": "int", "nightlife": "int"},
            "steps": "int",
            "scenes": "array",
            "created_at": "datetime",
        }
    }


@app.post("/api/story/start", response_model=StartStoryResponse)
def start_story(req: StartStoryRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    title = f"{req.protagonist} in {req.setting}"
    doc = {
        "title": title,
        "protagonist": req.protagonist,
        "setting": req.setting,
        "weights": req.weights,
        "steps": 0,
        "scenes": [],
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    story_id = create_document("story", doc)

    scene0 = _make_scene(req.protagonist, req.setting, req.weights, 0, story_id)

    # persist first scene
    db["story"].update_one(
        {"_id": ObjectId(story_id)},
        {"$set": {"steps": 1, "updated_at": datetime.now(timezone.utc)}, "$push": {"scenes": scene0.model_dump()}},
    )

    return StartStoryResponse(story_id=story_id, title=title, scene=scene0)


@app.get("/api/story/list")
def list_stories(limit: int = 10):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    docs = db["story"].find({}, {"scenes": {"$slice": 1}}).sort("created_at", -1).limit(limit)
    items = []
    for d in docs:
        items.append(
            {
                "id": str(d.get("_id")),
                "title": d.get("title"),
                "created_at": d.get("created_at"),
                "steps": d.get("steps", 0),
            }
        )
    return {"items": items}


@app.get("/api/story/{story_id}")
def get_story(story_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    doc = db["story"].find_one({"_id": ObjectId(story_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Story not found")
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.post("/api/story/choose", response_model=ChooseResponse)
def choose(req: ChooseRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    story = db["story"].find_one({"_id": ObjectId(req.story_id)})
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")

    steps = int(story.get("steps", 0))
    weights = story.get("weights", {"adventure": 40, "romance": 30, "nightlife": 30})
    protagonist = story.get("protagonist", "You")
    setting = story.get("setting", "the city")

    is_complete = steps >= GEN_MAX_STEPS - 1

    next_scene = _make_scene(protagonist, setting, weights, steps, req.story_id)

    # If final, modify text to feel conclusive
    if is_complete:
        next_scene = Scene(
            index=next_scene.index,
            text=next_scene.text
            + "\n\nDawn softens the neon. Some doors close; better ones wait. The night keeps your secret, and you keep its map.",
            choices=[],
        )

    db["story"].update_one(
        {"_id": ObjectId(req.story_id)},
        {
            "$set": {"steps": steps + 1, "updated_at": datetime.now(timezone.utc), "status": "complete" if is_complete else "active"},
            "$push": {"scenes": next_scene.model_dump()},
        },
    )

    return ChooseResponse(scene=next_scene, is_complete=is_complete)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
