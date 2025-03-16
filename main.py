import transformers as tr
import torch

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Load tokenizers and models
tokenizer = tr.AutoTokenizer.from_pretrained(expert_path)
amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path)

# Set models to evaluation mode
amateur_model.eval()
expert_model.eval()

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

# Preparing the prompt
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

# Contrastive decoding function
def contrastive_generation(amateur_model, expert_model, prompt, max_tokens=50) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids

    for _ in range(max_tokens):
        # Get logits from both models
        with torch.no_grad():
            amateur_logits = amateur_model(generated_ids).logits[:, -1, :]
            expert_logits = expert_model(generated_ids).logits[:, -1, :]

        # Compute contrastive scores
        contrastive_scores = expert_logits - amateur_logits

        # Apply plausibility constraint (e.g., ensuring the token is plausible under the expert model)
        expert_probs = torch.nn.functional.softmax(expert_logits, dim=-1)
        plausible_mask = expert_probs > 0.01  # Example threshold to filter out improbable tokens
        contrastive_scores = contrastive_scores * plausible_mask

        # Select the token with the highest contrastive score
        next_token_id = torch.argmax(contrastive_scores, dim=-1).unsqueeze(-1)
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    # Decode generated ids to text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# Call the contrastive generation function
generated_output = contrastive_generation(amateur_model, expert_model, user_message, max_tokens=50)
print(generated_output)
