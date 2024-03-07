import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# torch.set_default_device("cuda")
torch.cuda.manual_seed_all(19)
print('Device: ',torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("loading_model")
model_lora=AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", device_map={"":0},  trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
# model_lora.to('cuda')
# tokenizer.to('cuda')
print("Finished")
# inputs = tokenizer('''def print_prime(n):
#    """
#    Print all primes between 1 and n
#    """''', return_tensors="pt", return_attention_mask=False)

# outputs = model_lora.generate(**inputs, max_length=200)
# text = tokenizer.batch_decode(outputs)[0]
# print(text)
tokenizer.pad_token = tokenizer.eos_token

def print_linear_layers(module, parent_name=''):
    for name, child in module.named_children():
        # print(name,' ::: ',child)
        if isinstance(child, torch.nn.Linear):
            print(f"Layer Name: {parent_name + '.' + name if parent_name else name}\nModule: {child}\n")
        else:
            # print('rec call')
            print_linear_layers(child, parent_name=name if not parent_name else parent_name + '.' + name)
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
def replace_linear_with_lora(model, rank, alpha):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(child, rank, alpha))
        else:
            replace_linear_with_lora(child, rank, alpha)
replace_linear_with_lora(model_lora,8,16)

for param in model_lora.parameters():
    param.requires_grad = False


for name, module in model_lora.named_modules():
    if isinstance(module, LoRALayer):  # Check if it's an instance of your LoRALayer
        # Unfreeze all parameters in this module
        for param in module.parameters():
            param.requires_grad = True


def tokenize(sample):
    model_inps =  tokenizer(sample["text"], padding=True, truncation=True, max_length=512)
    return model_inps
     
print("Creating Dataset")
data = load_dataset("gsm8k", "main", split="train")
data_df = data.to_pandas()
data_df["text"] = data_df[["question", "answer"]].apply(lambda x: "question: " + x["question"] + " answer: " + x["answer"], axis=1)
data = Dataset.from_pandas(data_df)
tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)
# tokenized_data = tokenized_data.with_format("torch")
# tokenized_data.set_tensor_type("torch.cuda.FloatTensor")
# tokenized_data = tokenized_data.with_format("torch").to("cuda") 
tokenized_data
print("starting_training")
training_arguments = TrainingArguments(
        output_dir="phi-1_5-finetuned-gsm8k",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=100,
        max_steps=100,
        # device_map='auto',
        num_train_epochs=1,
        push_to_hub=False
    )
     

trainer = Trainer(
    model=model_lora,
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()