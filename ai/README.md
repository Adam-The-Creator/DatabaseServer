# AI module

AI based decision support for the DataAnalysis web application (requirements `AI-01` to `AI-17` in
`DataAnalysis/requirements.md`). Like `analysis/` and `feedback/` it is **additive**: it only reads the existing
SQLite tables and MongoDB drawings and never changes either. What it stores are files in the analysis file store
(`analysis_store/ai/`).

**Everything here is decision support, never a diagnosis.** Every output is an *AI envelope*
(`envelope.py`): it is labelled `AI-generated`, carries the sentence that it has to be reviewed by a psychologist, and
says which model produced it, with which parameters, which prompt/rule/vocabulary version, how sure it is (and what
that number means) and what it is based on. The web app and the HTML reports show the same marking
(`AiBox.tsx`, `reporting.py`). A model's answer that uses clinical or diagnostic wording is discarded, not shown.

```text
ai/
  envelope.py    the envelope of every output, the reason texts, the wording guard
  settings.py    the switches and thresholds, read from config.py
  features.py    the metric vector and the robust statistics (shared by AI-10 and AI-14)
  cohort_ml.py   AI-10  atypical drawings (robust z-scores, isolation forest), similar drawings, PCA map
  clustering.py  AI-11  strokes into object-like groups (DBSCAN in space and time)
  fuzzy.py       AI-14  fuzzy logic indicators, reads rules/fuzzy_rules.toml
  rules/         the rule base psychologists can edit
  render.py      the canonical views (front, top, side, isometric) as PNG, numpy only
  semantic.py    AI-12  zero-shot labelling with OpenCLIP against data/vocabulary.toml
  vlm.py         AI-13  description by a vision-language model (via Ollama)
  llm.py         AI-16  conclusion by a language model (via Ollama)
  answers.py     checks of a model's JSON answer (format, wording)
  ollama.py      the client of the local Ollama server (loopback only, no proxy)
  insights.py    AI-15  what recurs across a session
  registry.py    AI-06  models and libraries with version, licence, status, and the explicit download
  jobs.py        AI-02  background jobs with progress and cancel
  store.py       AI-02  stored results (keyed, a result of an older key is shown as out of date)
  service.py     reads the databases, calls the functions, wraps and caches the results
  reporting.py   the AI sections of the HTML reports
routers/ai.py    the HTTP endpoints
tests/test_ai.py the tests
```

## What needs what

| Function | Needs | Without it |
| -------- | ----- | ---------- |
| AI-10 cohort analytics, AI-11 stroke groups | `numpy`, `scikit-learn` (in `requirements.txt`) | says *missing library* |
| AI-14 fuzzy indicators, AI-15 insights | nothing extra | |
| AI-12 semantic labels | `torch`, `open_clip_torch` (`requirements-ai.txt`) and the CLIP weights | says *missing library / missing model* |
| AI-13 description | [Ollama](https://ollama.com) running locally, and a vision model | says *service unreachable / missing model* |
| AI-16 conclusion | Ollama running locally, and a text model | same |

The server **starts and works without any of them** (AI-03): a function that cannot run says why, in the web app and on
the settings page. Nothing is downloaded silently (AI-06): the weights and the Ollama models come onto the machine only
when somebody starts the download on the *AI* page of the web app (or installs them by hand).

### Setting up the model based functions

**Step-by-step guide: [`INSTALL_MODELS.md`](INSTALL_MODELS.md)** (which Python environment, PyTorch and OpenCLIP from
`requirements-ai.txt`, the CLIP weights, Ollama and its two models, checks, troubleshooting).

The model names and the CLIP model are in `config.py` (`AI_TEXT_MODEL`, `AI_VISION_MODEL`, `AI_CLIP_MODEL`,
`AI_CLIP_PRETRAINED`). A GPU is optional: Ollama uses it by itself, CLIP uses it when PyTorch finds one.
**Licences:** they are recorded from the publishers' model cards and are *not yet verified* against the use in the study
(the settings page says so). That check is still to be done before the study.

## Switches and thresholds (`config.py`)

`AI_ENABLED` turns everything off, `AI_FEATURES` one function at a time. They are read when the server starts.
`AI_EVALUATED` stays `False` per function until the function was tried on a sample set together with a psychologist
(requirement AI-07, see `EVALUATION.md`): until then every result carries a "not evaluated" note. The other settings
are the minimum group sizes (`AI_MIN_COHORT`, `AI_MIN_FUZZY_COHORT`, `AI_MIN_ISOLATION_FOREST`), the job limits and the
Ollama address, which **must be a loopback address** (AI-01): the client refuses anything else and ignores proxy settings.

## Endpoints (all under `/ai`, Psychologist token only)

| Endpoint | What it returns |
| -------- | --------------- |
| `GET /status` | Which functions are on, evaluated and available (with the reason if not) |
| `GET /models`, `POST /models/{id}/download` | Libraries, rule base and models with version, licence, status; an explicit download (a job) |
| `GET /rules`, `GET /vocabulary` | The fuzzy rule base and the label vocabulary, as read from their files |
| `GET /drawings/{id}/cohort?scope=&ageMin=&ageMax=` | AI-10: is the drawing atypical, the most similar drawings, a 2D map of the group |
| `GET /drawings/{id}/groups?spaceEps=&timeEps=` | AI-11: stroke groups with bounding box, colours, size |
| `GET /drawings/{id}/indicators?scope=` | AI-14: the indicators with degrees, fired rules and inputs |
| `GET/POST/DELETE /drawings/{id}/{labels,description,summary}` | AI-12/13/16: the stored result (and whether it is up to date), start a job, discard |
| `GET /drawings/{id}/views/{front,top,side,iso}.png?groupId=` | The view a model is shown (evidence) |
| `GET /sessions/{id}/cohort`, `GET /sessions/{id}/insights` | AI-10 for a session, AI-15 |
| `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel` | The background jobs |

`scope` is the comparison group: `gameType` (the default), `session` or `all`. Browse them with the server's `/docs`.

## Things worth knowing

- **Atypical** means a robust z-score beyond ±3.5 on a measurement (median and MAD of the group, on a log scale for
  skewed quantities), and the group has to be spread on that measurement. A minimum spread per measurement
  (`features.MIN_SCALE`, for example 10 %) stops a tiny difference in a very tight group from counting. A measurement
  where more than half of the drawings have the same value can describe a drawing but never make it atypical.
  The isolation forest is a second opinion (it needs 12 drawings), the robust z-scores explain *which* measurement.
- **Fuzzy indicators** turn each measurement into low/medium/high by the 10th/50th/90th percentile of the comparison
  group, or by the fixed ranges of the rule file when the group has fewer than 8 drawings. The rule file
  (`rules/fuzzy_rules.toml`) is read again on every request; a file with an error is not used and the indicators say
  what is wrong. Its `status` says `proposal` until psychologists have reviewed it, and the web app shows that.
- **Stored results** are keyed by the drawing's content fingerprint, the analyzer, the model, the prompt and (for the
  conclusion) a hash of the facts it was written from. A result with an older key is still shown, marked out of date.
  Reports contain the stored results of the model based functions, an out of date one marked as such, and remember
  which AI results they were made with, so they become *out of date* when new ones appear.
- **Jobs** run on a small thread pool (`AI_MAX_CONCURRENT_JOBS`, default 1) so the game's requests are not starved. A job
  stops at its next step when cancelled; a request that is already waiting for a model ends at the model's timeout.
  Jobs are kept in memory, the finished results are files.
- **A closed local port** takes about 2 s to be refused on Windows. The question "is Ollama there?" therefore gives up
  after `AI_OLLAMA_PROBE_SECONDS` and the answer is remembered for a while (20 s when it is not there).
- **The language model never sees a name.** Its facts hold the game type, the age band, the measurements, the indicators
  and the other findings. The facts are stored with the result (evidence), so a psychologist can see what it was given.
- **Packaged server.** `Tools\Build-Release.ps1` bundles `ai\rules` and `ai\data`. numpy and scikit-learn are in
  `requirements.txt` and so in the exe; PyTorch is in `requirements-ai.txt` only, so in the exe the CLIP labelling says
  *missing library*.

## What the real models did

Run through a real server on the 19 demo drawings (flower, tree, house, sun, waves, spiral, scribble, drawn by the demo
generator, **not** by children), CLIP ViT-B-32 on the CPU, `qwen2.5vl:7b` and `qwen2.5:7b-instruct` in Ollama on the
RTX 5060 Ti. This is an impression from a few drawings, **not** the evaluation of requirement AI-07.

- **Speed.** CLIP: 5 to 15 seconds per drawing (4 views of the whole drawing and 2 views of each stroke group). The
  vision model: about 4 seconds, and about 55 seconds for the first call after a start (it loads the model). The
  conclusion: about 5 seconds.
- **Semantic labels.** Right for the house (98 % on its group), the sun, the sea (water on most groups), the scribbles
  (named "scribble" and marked unspecific) and the flowers. **Wrong, and not marked as unspecific:** the trees ("The old
  oak", "Branches") came out as *flower* with 52 to 73 %, and arcs and waves often as *rainbow*. The share is how CLIP
  spreads 100 % over the vocabulary, so a wrong answer can be confident. This is why the web app words it as a share, not
  a probability, and why the function needs the evaluation with psychologists before anyone relies on it.
- **Description.** With the first prompt the model answered `scene` with one word ("front", "isometric": it repeated the
  names of the views from the prompt). Prompt `vlm-2` and a minimum length in the schema fixed that: the scenes are now
  sentences that fit ("A colorful line drawing resembling a flower with a stem and leaves", "a sun with rays, three
  wavy lines below it"). The model's own `abstract` flag and its sentence sometimes disagree (a "house" marked abstract).
- **Conclusion.** With the first prompt the model guessed at feelings and preferences ("the participant's emotional
  state", "a preference for left-handed drawing") and proposed questions about emotions. Prompt `llm-2` forbids guessing
  and asks for neutral questions ("Can you describe the flower you created?"). It still sometimes adds a statement that is
  not in the facts ("typical for this age group"), which the web app warns about. The guard for clinical wording never
  fired on these answers.
- **Reports** with all six AI sections were generated and name the real models in their provenance.

## Tests

```text
python -m unittest discover -s tests -t .
```

`tests/test_ai.py` uses synthetic drawings with known answers (a planted outlier, planted stroke groups, hand-computed
membership degrees), a tiny fake Ollama server, and fake `open_clip` / `huggingface_hub` modules for the loading and
download of CLIP. The real models are not part of the test suite (they need gigabytes and a GPU or minutes of CPU): they
were tried by hand as described above.
