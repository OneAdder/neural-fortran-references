from transformers import AutoModelForCausalLM, AutoTokenizer


tok = AutoTokenizer.from_pretrained('trl-internal-testing/tiny-Qwen2ForCausalLM-2.5')
model = AutoModelForCausalLM.from_pretrained('trl-internal-testing/tiny-Qwen2ForCausalLM-2.5')
# tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
# model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')


tokens = tok('In faith I do not love thee with mine', return_tensors='pt', padding='longest')
pred = model.forward(**tokens)
