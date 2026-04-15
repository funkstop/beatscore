# beatscore

A small libary that converts structured text into short music clips by mapping properties of the input (intensity, repetition, pace, tone, coherence) into promptable audio generation using the ElevenLabs Music API. 

See wome examples here: [Example generated music](https://funkstop.github.io/beatscore)

The goal is to experiment how differences in source material can be perceptible in sound.

## Example

- Opinionated text → more forceful, higher energy
- Neutral news → more structured, calmer
- Fragmented text → faster, less predictable

## API

```python
run_digest(sources, source_type, output_dir)
```
where:
 * sources: a dictionary of articles, feeds, or any text.
 * source_type: one of 'news' or 'speech'. News expects sources to be a dictionary of rss feeds.
 * output_dir: the location for generated files




## News RSS Example:

```python
from beatscore import run_digest

sources = {
    "reuters": "http://feeds.reuters.com/reuters/topNews",
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "nyt": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "rollingStone": "https://www.rollingstone.com/feed/"
    }


run_digest(sources, "news", output_dir="./output")

```


## Speech example:

```python
from beatscore import run_digest

sources = {
    "mlk_dream": "https://kr.usembassy.gov/martin-luther-king-jr-dream-speech-1963/", #https://www.americanrhetoric.com/speeches/mlkihaveadream.htm",
    "gettysburg": "https://www.abrahamlincolnonline.org/lincoln/speeches/gettysburg.htm",
    "churchill_beaches": "https://winstonchurchill.org/resources/speeches/1940-the-finest-hour/we-shall-fight-on-the-beaches/",
    }


run_digest(sources, "speech", output_dir="./output")

```

## Setup

You'll need three API keys:
- `ELEVENLABS_API_KEY` — for music generation
- `HF_TOKEN` — for embeddings and emotion models  
- `ANTHROPIC_API_KEY` — for prompt engineering with Claude

Set them as environment variables or pass directly to functions.


## Installation

Install latest from the GitHub [repository][repo]:

```sh
$ pip install git+https://github.com/funkstop/beatscore.git
```

or from [pypi][pypi]


```sh
$ pip install beatscore
```


[repo]: https://github.com/funkstop/beatscore
[docs]: https://funkstop.github.io/beatscore/
[pypi]: https://pypi.org/project/beatscore/


### Documentation

Documentation can be found hosted on this GitHub [repository][repo]'s [pages][docs]. Additionally you can find package manager specific guidelines on [pypi][pypi].

[repo]: https://github.com/funkstop/beatscore
[docs]: https://funkstop.github.io/beatscore/
[pypi]: https://pypi.org/project/beatscore/


