import torch
import logging
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
from config.settings import settings

logger = logging.getLogger(__name__)

class LocalLLMGenerator:
    """LLM generator optimized for AMD Ryzen AI Max 395 hardware"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_initialized = False
        
        # Hardware optimization settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimization_config = self._get_optimization_config()
        
        self._initialize_model()

    def _get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration for your specific hardware"""
        
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        return {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "max_memory": self._get_memory_mapping()
        }

    def _get_memory_mapping(self) -> Dict:
        """Optimize memory usage for 128GB RAM system"""
        if torch.cuda.is_available():
            return {0: "40GB", "cpu": "80GB"}  # Utilize your 128GB effectively
        else:
            return {"cpu": "120GB"}  CPU-only with large RAM

    def _initialize_model(self):
        """Initialize the local LLM with error handling"""
        try:
            logger.info("🚀 Initializing local LLM...")
            
            # Model selection based on availability
            model_path = self._select_model()
            
            logger.info(f"📦 Loading model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model with optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **self.optimization_config
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                **settings.LOCAL_LLM_PARAMS
            )
            
            self.is_initialized = True
            logger.info("✅ Local LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize local LLM: {str(e)}")
            self._setup_fallback()

    def _select_model(self) -> str:
        """Select the best available model for Arabic content"""
        # Priority list of models suitable for Arabic
        model_priority = [
            settings.LOCAL_LLM_PATH,  # Your specified path
            "aubmindlab/aragpt2-base",  # Arabic GPT-2
            "marefa-nlp/marefa-bert-base",  # Arabic BERT
            "UBC-NLP/MARBERT",  # Modern Arabic BERT
        ]
        
        for model_path in model_priority:
            if model_path and model_path != "path/to/your/local/model":
                return model_path
        
        # Fallback model
        return "gpt2"  # English fallback

    def _setup_fallback(self):
        """Setup fallback for when local model fails"""
        logger.warning("🔄 Setting up fallback mode - using simple pattern matching")
        self.is_initialized = False
        self.fallback_responses = {
            "legal": "هذا سؤال قانوني يتطلب تحليل القوانين الكويتية.",
            "religious": "هذا سؤال ديني يتطلب الرجوع إلى المصادر الإسلامية.",
            "historical": "هذا سؤال تاريخي عن الكويت أو المنطقة.",
            "cultural": "هذا سؤال عن الثقافة والتراث الكويتي.",
            "general": "شكرًا لسؤالك. أحتاج إلى مزيد من المعلومات للإجابة بدقة."
        }

    def generate_answer(self, question: str, context: str, query_type: str = "general") -> str:
        """Generate answer using local LLM with Arabic optimization"""
        
        if not self.is_initialized:
            return self._fallback_answer(query_type)
        
        try:
            # Build Arabic-optimized prompt
            prompt = self._build_arabic_prompt(question, context, query_type)
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=settings.LOCAL_LLM_PARAMS["max_tokens"],
                temperature=settings.LOCAL_LLM_PARAMS["temperature"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )[0]['generated_text']
            
            # Extract the answer part (remove the prompt)
            answer = response[len(prompt):].strip()
            
            # Post-process for Arabic quality
            answer = self._post_process_arabic(answer)
            
            logger.info(f"✅ Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {str(e)}")
            return self._fallback_answer(query_type)

    def _build_arabic_prompt(self, question: str, context: str, query_type: str) -> str:
        """Build Arabic-optimized prompt based on query type"""
        
        base_prompt = """
        أنت مساعد خبير متخصص في المعلومات الكويتية. استخدم المعلومات التالية للإجابة على السؤال بدقة وأمانة.

        المعلومات المتاحة:
        {context}

        السؤال: {question}

        تعليمات مهمة:
        - اذكر المصادر عندما تكون متوفرة
        - كن دقيقًا في المعلومات القانونية والدينية
        - استخدم اللغة العربية الفصحى
        - إذا لم تكن متأكدًا، قل أنك لا تعرف بدلاً من التخمين

        الإجابة:
        """.format(context=context, question=question)
        
        # Add type-specific instructions
        type_instructions = {
            "legal": "\nملاحظة: هذا سؤال قانوني - كن دقيقًا في ذكر المواد والقوانين.",
            "religious": "\nملاحظة: هذا سؤال ديني - التزم بالمصادر الإسلامية المعتمدة.",
            "historical": "\nملاحظة: هذا سؤال تاريخي - قدم التواريخ والسياق بدقة.",
            "cultural": "\nملاحظة: هذا سؤال ثقافي - ركز على التراث الكويتي الأصيل."
        }
        
        prompt = base_prompt + type_instructions.get(query_type, "")
        return prompt.strip()

    def _post_process_arabic(self, text: str) -> str:
        """Post-process Arabic text for better quality"""
        # Remove repetitive patterns
        text = re.sub(r'(.)\1{3,}', r'\1', text)  # Remove character repetition
        
        # Ensure proper Arabic punctuation
        text = text.replace('..', '۔').replace('...', '…')
        
        # Trim and clean
        text = text.strip()
        
        # Ensure the response ends with proper punctuation
        if text and text[-1] not in ['۔', '.', '!', '؟']:
            text += '۔'
        
        return text

    def _fallback_answer(self, query_type: str) -> str:
        """Provide fallback answer when LLM is not available"""
        fallback = self.fallback_responses.get(query_type, self.fallback_responses["general"])
        return fallback + " (النظام في وضع الاستعداد)"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_initialized:
            return {"status": "not_initialized", "fallback_mode": True}
        
        return {
            "status": "initialized",
            "model_path": str(self.model.config.name_or_path),
            "device": self.device,
            "max_tokens": settings.LOCAL_LLM_PARAMS["max_tokens"],
            "optimizations": ["4-bit_quantization", "memory_mapping"]
        }