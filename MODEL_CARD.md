# Model card

## Intended use
Educational English sentiment/emotion exploration and research baseline comparison. No clinical or high-stakes inference. Original rules are MIT source code; external pretrained models retain their licenses.

## Implementations
- `custom-rule-baseline-v1`: English sentiment lexicon; parity-based local negation scope of three tokens; boundaries at punctuation and contrast; intensifiers and capped exclamation emphasis. Emotion keyword hits are suppressed under negation. Unknown emotion is possible. Softmax-normalized rule support is heuristic, not calibrated probability. No model training occurs.
- Cardiff NLP `twitter-roberta-base-sentiment-latest`, revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`, CC BY 4.0. Three sentiment labels; social-media pretraining/fine-tuning per [official card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- Bhadresh Savani `distilbert-base-uncased-emotion`, revision `ce6f4ffcde7642ca2cac02381a16da38e5498ff7`, Apache 2.0. Six labels aligned with the DAIR-AI Emotion research set. Pretrained classifier used without fine-tuning in this project. [Official card](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion).
- Browser sentiment: Xenova ONNX conversion pinned at `f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445`, quantized, Transformers.js 2.17.2. Upstream CC BY 4.0; conversion card does not independently declare a license. Browser results are not covered by the Python benchmark and need separate validation. No browser emotion transformer is claimed.

## Inference and explanation
Python runs CPU inference, mention/URL placeholders and a 256 subword-token cap. Lazy caches avoid repeated model loads. `trust_remote_code=False` prevents remote Python model code execution. Pinned revisions prevent silently changing weights. Optional token deletion reports the change in the original label score for up to sixteen word positions. This is sensitivity analysis, not causal proof or faithful attention attribution. It can form unnatural text and is expensive.

## Mood mapping
These six mood names are a suite-specific mapping inspired by MoodTrip; this is not an integration or a reproduction of MoodTrip’s broader mood taxonomy.

| Emotion | Suite mood | Lost information |
|---|---|---|
| sadness | sad | Severity and cause |
| joy | happy | Excitement versus contentment |
| love | affectionate | Relationship and target |
| anger | angry | Intensity and target |
| fear | anxious | Immediate fear versus anticipation |
| surprise | surprised | Pleasant versus unpleasant valence |
| unknown | unknown | No evidence; not a measured neutral state |

## Uncertainty and evaluation
Default threshold 0.60; uncertain flag accompanies the raw class rather than hiding it. Conflicting rule sentiment and absence of rule evidence are also uncertain. Evaluation scores all raw predictions and reports coverage. See generated JSON for current measured runs; no upstream paper numbers are copied as project metrics. ECE on rule support is descriptive only.

## Known limitations
Sarcasm, slang, misspellings, dialect, cultural context, code-mixing, domain shift and mixed emotions. Emotion classifier lacks neutral/optimism labels. Model confidence can be high on incorrect predictions. Financial PhraseBank uses investor-oriented polarity, which may differ from ordinary conversational sentiment. Model pretrained corpus overlap cannot be ruled out from split deduplication alone.

## Safety and privacy
No diagnosis, risk assessment or therapeutic recommendation. A small crisis-phrase list only displays a human-support message; it does not detect all crises. App code stores no text. Provider/request metadata can persist outside app control. No medical, demographic or psychological validity study has been conducted.
