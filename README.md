# FAB-G Qwen3-VL LoRA Inference

This directory contains the inference code for the FAB-G multi-agent framework
described in the paper. In this implementation, an agent is not a separate full
model. Instead, all agents share one Qwen3-VL-8B base model, and each agent is
implemented as a different LoRA adapter.

The system uses six LoRA adapters:

- `color`
- `composition`
- `line`
- `light`
- `brushstroke`
- `final`

The first five adapters are attribute-salience agents. They answer whether a
formal attribute is one of the main emotional factors in the image. The `final`
adapter performs the final emotion analysis using only the attributes selected
by the first stage.

## Pipeline

```text
Input artwork
  -> color agent: yes/no
  -> composition agent: yes/no
  -> line agent: yes/no
  -> light agent: yes/no
  -> brushstroke agent: yes/no
  -> collect attributes answered as yes
  -> final agent analyzes emotion using only those attributes
  -> output emotion / arousal / valence / explanation
```

This matches the two-stage FAB-G design:

1. Attribute salience screening.
2. Cue-constrained emotional reasoning.

## Dataset

Download the salience dataset from Hugging Face:

```bash
huggingface-cli download printblue/EmoArt-Salience \
  --repo-type dataset \
  --local-dir data/EmoArt-Salience
```

The dataset is expected to provide artwork images, binary salience labels for
the five formal attributes, and emotion labels. If the downloaded files use a
different layout, convert them into the multimodal supervised fine-tuning format
required by your training pipeline.

## LoRA Training

We recommend using LLaMA-Factory to fine-tune the LoRA adapters for
`Qwen/Qwen3-VL-8B-Instruct`.

Train one LoRA adapter for each attribute agent:

```text
fabg_color_lora
fabg_composition_lora
fabg_line_lora
fabg_light_lora
fabg_brushstroke_lora
```

Then train one additional LoRA adapter for the final emotion-analysis agent:

```text
fabg_final_lora
```

The attribute-agent training targets should be constrained `yes` / `no`
answers. The final-agent training target should be a JSON response containing:

```json
{
  "emotion": "...",
  "arousal": "...",
  "valence": "...",
  "explanation": "..."
}
```

This repository does not prescribe the exact LLaMA-Factory configuration. Use
the configuration that matches your hardware, dataset format, and Qwen3-VL
environment.

## Configure Inference Paths

After training, edit the paths at the top of `infer_fabg_qwen3vl.py`:

```python
BASE_MODEL_PATH = r"Qwen/Qwen3-VL-8B-Instruct"

ATTRIBUTE_AGENT_ADAPTER_PATHS = {
    "color": r"checkpoints/fabg_color_lora",
    "composition": r"checkpoints/fabg_composition_lora",
    "line": r"checkpoints/fabg_line_lora",
    "light": r"checkpoints/fabg_light_lora",
    "brushstroke": r"checkpoints/fabg_brushstroke_lora",
}

FINAL_AGENT_ADAPTER_PATH = r"checkpoints/fabg_final_lora"
```

Use absolute paths if your checkpoints are outside this project.

## Run Inference

Install dependencies:

```bash
pip install -r code/requirements.txt
```

Run FAB-G inference:

```bash
python code/infer_fabg_qwen3vl.py sample.png
```

Example output:

```json
{
  "image_path": "sample.png",
  "salience_mask": {
    "color": 1,
    "composition": 1,
    "line": 1,
    "light": 0,
    "brushstroke": 0
  },
  "salient_attributes": ["color", "composition", "line"],
  "emotion": "melancholy",
  "arousal": "low",
  "valence": "negative",
  "explanation": "Muted color, sparse composition, and downward linear rhythm support a restrained melancholic affect."
}
```

## Code Structure

```text
code/
  infer_fabg_qwen3vl.py
  requirements.txt
  fabg_sim/
    core.py
```

Key components:

- `AttributeAgent`: builds the yes/no salience prompt for one formal attribute.
- `EmotionAnalysisAgent`: builds the final cue-constrained emotion prompt.
- `MultiAgentFABG`: runs the five attribute agents and then the final agent.
- `Qwen3VLLoRABackend`: loads the Qwen3-VL base model and switches LoRA adapters
  with `set_adapter(...)`.

## Notes

- The base model is loaded once.
- LoRA adapters are loaded on first use and then kept in memory.
- The five attribute agents are executed sequentially.
- The final agent receives only the attributes answered as `yes`.
- Attribute agents should output `yes` or `no`; otherwise parsing may fail.
- The final agent should output valid JSON.
