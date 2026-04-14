import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from claudette import Chat
from elevenlabs.client import ElevenLabs

def validate_keys():
    if ("ELEVENLABS_API_KEY" in os.environ):
        print("ElevenLabs API Key is set")
    else:
        print("No Key. Set ELEVENLABS_API_KEY in environment and restart")
        raise ValueError("No ELEVENLABS_API_KEY")

    if ("HF_TOKEN" in os.environ):
        print("HuggingFace API Key is set")
    else:
        print("No Key. Set HF_TOKEN in environment and restart")
        raise ValueError("No HF_TOKEN")

    if ("ANTHROPIC_API_KEY" in os.environ):
        print("Anthropic API Key is set")
    else:
        print("No Key. Set ANTHROPIC_API_KEY in environment and restart")
        raise ValueError("No ANTHROPIC_API_KEY")


RANT_TEXT = """
This is not a small issue! Anyone pretending otherwise is lying to your face! Every single day, decisions are made behind closed doors with zero accountability. !e are told to simply trust the process even though it has failed us over and over again. This is not incompetence anymore. It is a deliberate pattern! They know exactly what they are doing!

Why are ordinary citizens expected to follow every rule to the letter while those in power play by completely different standards? You ask questions and you get dismissed, labeled as extreme. These are not conspiracy theories. They are observable facts that anyone paying attention can see clearly.

Who actually benefits from this?! Not you. Not your family. Not your community. The gap between what they promise and what they deliver keeps growing wider, and at some point we all need to stop pretending that things are fine when they clearly are not.
"""

NEWS_TEXT = """
Good evening. Tonight, we begin with developments in Washington. Officials announced a new policy initiative aimed at addressing ongoing economic concerns. The proposal, introduced earlier today, is expected to face debate in the coming weeks.

According to sources familiar with the matter, the plan includes a series of measures designed to stabilize markets and provide support to affected industries. Analysts say the impact of these changes will depend largely on how they are implemented and received by both lawmakers and the public.

In other news, international leaders continue discussions on climate agreements. Several key meetings are scheduled for later this month. While progress has been reported, significant differences remain on critical issues.

We will continue to follow these stories and bring you updates as more information becomes available.
"""

CHAOS_TEXT = """
ok this is wild like actually insane?? just saw 3 different things happen at once and none of it makes sense

why is everything breaking lol nothing loads then suddenly it does and then it crashes again

also did anyone else notice that thing earlier?? no one is talking about it but it was everywhere for like 5 minutes

anyway im tired of this, going offline for a bit, this whole thing is too much
"""
TEST_SOURCES = {
    "rant": RANT_TEXT,
    "chaos": CHAOS_TEXT,
    "news": NEWS_TEXT}

SOURCES = {
    "mlk_dream": "https://kr.usembassy.gov/martin-luther-king-jr-dream-speech-1963/", #https://www.americanrhetoric.com/speeches/mlkihaveadream.htm",
    "gettysburg": "https://www.abrahamlincolnonline.org/lincoln/speeches/gettysburg.htm",
    "churchill_beaches": "https://winstonchurchill.org/resources/speeches/1940-the-finest-hour/we-shall-fight-on-the-beaches/",
}
NEWS_SOURCES = {
    "reuters": "http://feeds.reuters.com/reuters/topNews",
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "nyt": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "rollingStone": "https://www.rollingstone.com/feed/",
    "alJazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "fox": "https://moxie.foxnews.com/google-publisher/latest.xml",
    "tmz": "https://www.tmz.com/rss.xml",
    "dailymail": "https://dailymail.co.uk/news/index.rss",
}
DRY_RUN_LIMIT = 3      # number of hours to process
VARIANTS = 1           # clips per hour
CLIP_SECONDS = 30       # desired length (passed into prompt)
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}



import feedparser

def fetch_feed(url):
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        items.append({
            "title": entry.title,
            "published": entry.get("published", entry.get("updated", ""))
        })
    return items
def fetch_speech(url): #this would be used for speeches or text in general.
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = []
    main = soup.find("main")
    candidates = main.find_all("p") if main else soup.find_all("p")

    for p in candidates:
        text = p.get_text(" ", strip=True)
        if len(text) >= 80:
            paragraphs.append(text)

    return paragraphs
def text_to_rows(text, source):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    return [
        {"source": source, "section_id": i, "text": p}
        for i, p in enumerate(paragraphs)
    ]



from functools import lru_cache

@lru_cache()
def get_model():
    return SentenceTransformer("all-mpnet-base-v2")

@lru_cache()
def get_sentiment_pipe():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@lru_cache()
def get_emotion_pipe():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
from sentence_transformers import SentenceTransformer

model = get_model() 

from transformers import pipeline
import re

# sentiment: positive/negative
sentiment_pipe = get_sentiment_pipe()

# emotion: joy / anger / fear / sadness / etc.
emotion_pipe = get_emotion_pipe()

def chunk_text(text, max_words=250):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def split_sentences(text):
    parts = re.split(r'[.!?]+', text)
    return [p.strip() for p in parts if p.strip()]

def tokenize(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def tone_score(text):
    chunks = chunk_text(text, max_words=250)
    preds = emotion_pipe(chunks)

    vals = []
    for chunk_preds in preds:
        scores = {d["label"].lower(): d["score"] for d in chunk_preds}

        positive = (
            scores.get("joy", 0.0) +
            scores.get("love", 0.0) +
            scores.get("surprise", 0.0) * 0.25
        )

        negative = (
            scores.get("anger", 0.0) +
            scores.get("fear", 0.0) +
            scores.get("sadness", 0.0) +
            scores.get("disgust", 0.0)
        )

        raw = positive - negative
        val = 0.5 + 0.5 * raw
        vals.append(float(np.clip(val, 0, 1)))

    return float(np.mean(vals))

def pace_score(text):
    chunks = chunk_text(text, max_words=250)
    preds = emotion_pipe(chunks)

    urgency_vals = []
    for chunk_preds in preds:
        scores = {d["label"].lower(): d["score"] for d in chunk_preds}

        urgency = (
            scores.get("anger", 0.0) +
            scores.get("fear", 0.0) +
            scores.get("surprise", 0.0) +
            scores.get("disgust", 0.0) * 0.5 +
            scores.get("neutral", 0.0) * 0.35 +
            scores.get("joy", 0.0) * 0.5 +
            scores.get("sadness", 0.0) * 0.2
        )
        urgency_vals.append(min(urgency, 1.0))

    model_score = float(np.mean(urgency_vals)) if urgency_vals else 0.0

    # structural signals
    sentences = split_sentences(text)
    words = tokenize(text)
    
    exclam = text.count("!") + text.count("?")
    short_sentences = sum(1 for s in sentences if len(tokenize(s)) < 8) if sentences else 0
    avg_len = np.mean([len(tokenize(s)) for s in sentences if tokenize(s)]) if sentences else 20
    
    structural = (
        (exclam / 10.0) * 0.3 +
        (short_sentences / max(len(sentences), 1)) * 0.4 +
        (1.0 / (avg_len + 1)) * 5.0 * 0.3  # shorter avg = higher pace
    )
    structural = float(np.clip(structural, 0, 1))

    return float(np.clip(0.4 * model_score + 0.6 * structural, 0, 1))

def coherence_score(embs):
    if len(embs) < 2:
        return 0.0
    sims = cosine_similarity(embs)
    return float(np.mean(sims))

def repetition_score(text):
    words = tokenize(text)
    if not words:
        return 0.0
    return 1.0 - (len(set(words)) / len(words))
    
def intensity_score(text):
    chunks = chunk_text(text, max_words=250)
    preds = emotion_pipe(chunks)

    model_vals = []
    for chunk_preds in preds:
        scores = {d["label"].lower(): d["score"] for d in chunk_preds}
        agitation = (
            scores.get("anger", 0.0) +
            scores.get("fear", 0.0) +
            scores.get("disgust", 0.0)
        )
        model_vals.append(min(agitation, 1.0))

    model_score = float(np.mean(model_vals)) if model_vals else 0.0

    # structural boost so chaotic / bursty text is not mislabeled as calm
    exclam = text.count("!")
    question = text.count("?")
    short_sentences = sum(
        1 for s in split_sentences(text)
        if len(tokenize(s)) < 6
    )

    structural_score = (exclam + question) / 10.0 + short_sentences / 20.0
    structural_score = float(np.clip(structural_score, 0, 1))

    return float(np.clip(0.5 * model_score + 0.5 * structural_score, 0, 1))
def getJoinedText(group):
    if (sourceType == "news"):
        returnText = " ".join(group["title"].tolist())
    else:
        returnText = " ".join(group["text"].tolist())
    return returnText

def source_metrics(group):
    full_text = getJoinedText(group)
    embs = np.vstack(group["embedding"].values)

    return {
        "repetition": repetition_score(full_text),
        "pace": pace_score(full_text),
        "intensity": intensity_score(full_text),
        "tone": tone_score(full_text),
        "coherence": coherence_score(embs),
    }
def avg_similarity(embs):
    embs = np.array(embs)
    norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = norm @ norm.T
    return sim.mean()
from sklearn.cluster import KMeans # import HDBSCAN

def cluster_count(embs, k=3):
    hdb = kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto") #HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = kmeans.fit_predict(embs) #hdb.fit(embs)
    return len(set(labels))

    #return len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
def cluster_spread(embs, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embs)
    
    counts = np.bincount(labels)
    probs = counts / counts.sum()
    
    # entropy → how evenly spread across clusters
    return -np.sum(probs * np.log(probs + 1e-9))
def continuity_score(embs):
    sims = []
    for i in range(len(embs) - 1):
        sim = cosine_similarity([embs[i]], [embs[i + 1]])[0][0]
        sims.append(sim)
    return float(np.mean(sims))
def build_prompt(params):
    intensity = params["intensity"]
    coherence = params["coherence"]
    pace = params["pace"]
    repetition = params["repetition"]
    tone = params["tone"]

    # Intensity — 5 tiers
    if intensity > 0.8:
        intensity_desc = "explosive, overwhelming, chaotic energy"
    elif intensity >= 0.5:
        intensity_desc = "urgent, relentless, focused drive"
    elif intensity > 0.4:
        intensity_desc = "building momentum, pushing forward"
    elif intensity > 0.2:
        intensity_desc = "simmering excitement, restrained but present"
    else:
        intensity_desc = "still, barely moving, subdued"

    # Coherence — 5 tiers
    if coherence > 0.8:
        coherence_desc = "locked-in, unwavering, hypnotic repetition"
    elif coherence > 0.6:
        coherence_desc = "steady and focused with minor detours"
    elif coherence > 0.4:
        coherence_desc = "loosely connected, drifting between ideas"
    elif coherence > 0.2:
        coherence_desc = "fractured, jumping between unrelated fragments"
    else:
        coherence_desc = "total chaos, no thread holding it together"

    # Pace — 5 tiers
    if pace > 0.8:
        pace_desc = "frantic, breathless, machine-gun rhythm"
    elif pace > 0.6:
        pace_desc = "fast and urgent, rapid-fire delivery"
    elif pace > 0.4:
        pace_desc = "moderate but purposeful movement"
    elif pace > 0.2:
        pace_desc = "slow and deliberate, heavy footsteps"
    else:
        pace_desc = "glacial, drawn out, almost frozen"

    # Repetition — 5 tiers
    if repetition > 0.8:
        repetition_desc = "obsessive loops on the same idea relentlessly, perfect repetition"
    elif repetition > 0.6:
        repetition_desc = "strong recurring motifs, circling back insistently"
    elif repetition > 0.4:
        repetition_desc = "some repeating patterns with variation"
    elif repetition > 0.2:
        repetition_desc = "mostly fresh ideas, little callback"
    else:
        repetition_desc = "constant change, never repeating"

    # Tone — 5 tiers
    if tone > 0.8:
        tone_desc = "radiant, joyful, triumphant"
    elif tone > 0.6:
        tone_desc = "warm, hopeful, gently uplifting"
    elif tone > 0.4:
        tone_desc = "neutral, observational, emotionally flat"
    elif tone > 0.2:
        tone_desc = "uneasy, critical, desiring change"
    else:
        tone_desc = "seething anger, dark and confrontational"

    return f"""
Create a made for radio song clip that does not have words.

Energy: {intensity_desc}
Structure: {coherence_desc}
Pace: {pace_desc}
Motifs: {repetition_desc}
Emotional tone: {tone_desc}

No words. Keep it concise and clearly reflect these rules.
"""
from functools import lru_cache

@lru_cache()
def get_el_client():
    return ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

def generate_music(prompt, hcFlag):
    client = get_el_client()

    if (hcFlag):
        prompt="a new funk soul song with heavy bass and a rap break. do not use ANY words"

    return client.music.compose(
        prompt=prompt,
        music_length_ms=CLIP_SECONDS * 500,
    )
import json, re

def parse_styles(resp):
    text = resp.content[0].text
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)


def get_full_plan(text, metrics, duration_ms=CLIP_SECONDS * 1000):
   

    chat = Chat("claude-sonnet-4-20250514")

    prompt = f"""Here is a text excerpt:
"{text}"

Computed metrics:
- intensity: {metrics['intensity']:.2f}
- pace: {metrics['pace']:.2f}
- tone: {metrics['tone']:.2f}
- coherence: {metrics['coherence']:.2f}
- repetition: {metrics['repetition']:.2f}

Generate an ElevenLabs music composition plan as JSON. Total duration: {duration_ms}ms. Instrumental only.

Use the text content to choose appropriate genres and instruments for global styles.
Use the metrics to shape the section structure — number of sections, durations, and local energy levels.

Return ONLY a JSON object with this exact structure:
{{
  "positive_global_styles": ["2-4 specific genre/instrument terms"],
  "negative_global_styles": ["2-4 terms to exclude"],
  "sections": [
    {{
      "section_name": "name",
      "positive_local_styles": ["2-4 specific instrument/structural terms"],
      "negative_local_styles": ["2-4 terms to exclude"],
      "duration_ms": integer,
      "lines": []
    }}
  ]
}}

Section durations must sum to {duration_ms}. Use 2-4 sections. No mood or emotion words — only genre names, instrument names, and structural music terms. Only use specific popular music genres (e.g. funk, hip-hop, grime, metal, thrash, rock, jazz, R&B, garage, drum and bass, punk, etc) over abstract genres like ambient or experimental. Use only band instruments (electric guitar, bass guitar, drum kit, synth). NEEVER use orchestral instruments (strings, brass, choir, woodwinds). ALWAYS use a full compliment of instruments
"""

    resp = chat(prompt)
    plan_dict = parse_styles(resp)
    
    from elevenlabs.types import MusicPrompt, SongSection
    return MusicPrompt(
        positive_global_styles=plan_dict["positive_global_styles"],
        negative_global_styles=plan_dict["negative_global_styles"],
        sections=[SongSection(**s) for s in plan_dict["sections"]]
    )
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np


def plot_umap(source):
   # group = df[(df["day"] == day) & (df["source"] == source)]
    group = df[(df["source"] == source)]

    if len(group) < 3:
        print("Not enough data")
        return

    embs = np.vstack(group["embedding"].values)

    reducer = umap.UMAP(n_components=2, random_state=42, init="random")
    coords = reducer.fit_transform(embs)

    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1])

    # label a few points
    if (sourceType == "news"):
        for i, title in enumerate(group["title"].tolist()[:10]):
            plt.text(coords[i,0], coords[i,1], str(i), fontsize=8)
    else:
        for i, title in enumerate(group["text"].tolist()[:10]):
            plt.text(coords[i,0], coords[i,1], str(i), fontsize=8)
            

    plt.title(f"Date — {source}")
    plt.show()

    # print mapping so you can inspect
   # if (sourceType == "news"):
   #     for i, t in enumerate(group["title"].tolist()):
   #         print(i, t)
   # else:
   #     for i, t in enumerate(group["text"].tolist()):
   #         print(i, t)
def get_simple_prompt(text, metrics):
    chat = Chat("claude-sonnet-4-20250514")
    prompt = f"""Here is a text excerpt:
"{text}"

Computed metrics:
- intensity: {metrics['intensity']:.2f}
- pace: {metrics['pace']:.2f}
- tone: {metrics['tone']:.2f}
- coherence: {metrics['coherence']:.2f}
- repetition: {metrics['repetition']:.2f}

Write a single short music prompt (1-2 sentences) describing a song that reflects this text's mood and energy. Use a specific genre (e.g. funk, soul, hip-hop, rock, jazz). Mention specific instruments. Say "no words". Return ONLY the prompt, nothing else."""

    resp = chat(prompt)
    return resp.content[0].text
def lookupSource(type, key):
    if (type == "news"):
        return NEWS_SOURCES[key]
    elif (type =="test"):
        return TEST_SOURCES[key]
    else:
        return SOURCES[key]

def process_source(source_name, source_url, source_type, output_dir="./outputs"):
    global sourceType
    sourceType = source_type

    source_data = source_url  # instead of lookupSource(...)
    # source_data = lookupSource(sourceType, source_name)

    if not source_data: 
        print(f"No source data for {source_type}")
        return
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Collect
    if source_type == "speech":
        paragraphs = fetch_speech(source_data)
        rows = [{"source": source_name, "section_id": i, "text": t} for i, t in enumerate(paragraphs)]
    elif source_type == "news":
        items = fetch_feed(source_data)
        for i in items:
            i["source"] = source_name
        rows = items
    elif source_type == "test":
        rows = text_to_rows(source_data, source_name)

    if not rows:
        print(f"Skipping {source_name} — no items returned")
        return None
    
    df = pd.DataFrame(rows)
    
    # 2. Embed
    text_col = "title" if source_type == "news" else "text"
    embeddings = get_model().encode(df[text_col].tolist(), show_progress_bar=True)
    df["embedding"] = list(embeddings)
    
    # 3. Score
    metrics = source_metrics(df)
    metrics["source"] = source_name
    print(f"Metrics for {source_name}:", metrics)
    
    # 4. Build plan
    full_text = " ".join(df[text_col].tolist())
    plan = get_full_plan(full_text, metrics)
    print(f"Plan for {source_name}:", plan.positive_global_styles)
    
    # 5. Generate audio
    client = get_el_client()
    result = client.music.compose_detailed(
        composition_plan=plan,
        respect_sections_durations=True,
    )
    
    filepath = f"{output_dir}/{source_name}_song.mp3"
    with open(filepath, "wb") as f:
        f.write(result.audio)
    
    print(f"Saved: {filepath}")
    return filepath
def validate_sources(sources, source_type):
    if not sources:
        raise ValueError("No sources provided")
    if source_type not in ("news", "speech"):
        raise ValueError(f"source_type must be 'news' or 'speech', got '{source_type}'")
    for name, url in sources.items():
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Invalid URL for source '{name}'")
    validate_keys()
def run_digest(sources, source_type, output_dir="./outputs"):
    validate_sources(sources, source_type)
    paths = []
    for name, url in sources.items():
        print(f"{name} and the url is: {url}")
        path = process_source(name, url, source_type, output_dir)
        if path: paths.append(path)
    return paths
