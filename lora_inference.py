from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

class LoraInference:
    def __init__(self, model_path, lora_path=None):
        """
        初始化模型和tokenizer
        
        Args:
            model_path (str): 基础模型路径
            lora_path (str, optional): LoRA适配器路径，如果不提供则使用基础模型
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
    
    def generate(self, prompt, max_new_tokens=200):
        """
        生成文本

        Args:
            prompt (str): 输入提示文本
            max_new_tokens (int, optional): 最大生成token数量，默认50

        Returns:
            str: 生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(0)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def cleanup(self):
        """
        清理模型和GPU缓存
        """
        self.model = None
        torch.cuda.empty_cache()

def main():
    # 示例使用
    model_path = r'D:\MODELS\deepseek-ai\DeepSeek-R1-Distill-Qwen-1_5b'
    #lora_path = r'D:\MODELS\deepseek-ai\model_2025-03-03_19-41-23-lora\checkpoint-750'
    
    # 初始化推理器
    inferencer = LoraInference(model_path)
    
    # 生成文本
    prompt = "写一篇童话故事：小猪，森林"
    result = inferencer.generate(prompt)
    print(result)
    
    # 清理资源
    #inferencer.cleanup()

if __name__ == "__main__":
    main() 