from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading the language model on CPU...", flush=True)
start = perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
).to("cpu")

model.eval()

print(
    f"Model loaded in {perf_counter() - start:.2f} seconds",
    flush=True,
)

context = (
    "[1] The Sun produces energy through nuclear fusion "
    "in its core. This energy provides heat and light."
)

questions = [
    "How does the Sun produce energy?",
    "How many moons does Mars have?",
]

for question in questions:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer in English using only the supplied context. "
                "Every factual detail in your answer must be explicitly "
                "supported by the context. "
                "Do not add facts from your own knowledge. "
                "Cite the source number, such as [1]. "
                "If the context does not contain the answer, say: "
                "'There is not enough information in the context.' "
                "Keep the answer short."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    print(f"\nQuestion: {question}", flush=True)
    print("Generating...", flush=True)

    start = perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("Answer:", answer, flush=True)
    print(
        f"Generation time: {perf_counter() - start:.2f} seconds",
        flush=True,
    )
