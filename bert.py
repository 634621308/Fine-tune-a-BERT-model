from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#加载数据集
dataset=load_dataset("IMDB")
print(dataset)

##数据预处理
#加载分词器
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
#对数据集进行分词
def tokenize_function(examples):
    return tokenizer(examples["text"],padding="max_length",truncation=True,max_length=128)


tokenizerd_datasets=dataset.map(tokenize_function,batched=True)

#加载预训练模型
model=BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)

#创建训练集和验证集
train_dataset=tokenizerd_datasets["train"]
eval_dataset=tokenizerd_datasets["test"]

#定义训练参数
training_args=TrainingArguments(
    output_dir="./results",#模型保存
    num_train_epochs=3,    
    eval_strategy="epoch",#每个周期结束时进行评估
    save_strategy="epoch",#每个周期结束时保存模型
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    weight_decay=0.01,#权重衰减，防止过拟合
    warmup_steps=500,#预热步数

    logging_dir="./logs",#日志保存目录
    logging_steps=10,

    load_best_model_at_end=True,#在训练结束时加载最佳模型
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train(resume_from_checkpoint="./results/checkpoint-4689")  #从指定检查点继续训练

#评估模型
eval_results=trainer.evaluate()
print(f"Evaluation results: {eval_results}")

#保存模型
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")