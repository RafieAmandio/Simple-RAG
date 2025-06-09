"""Ready-to-Use RAG Generator dengan Smart Fallback Language Detection"""

from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import re

class SmartLanguageDetector:
    """Smart language detector dengan sophisticated fallback untuk tie cases."""
    
    def __init__(self, llm=None):
        self.llm = llm
        
    def _weighted_score_detection(self, query: str) -> Tuple[str, int, int, float]:
        """Weighted scoring dengan detailed breakdown."""
        
        # Indonesian indicators dengan weights
        id_indicators = {
            # Strong command words - sangat jelas Indonesian
            'tuliskan': 5, 'buatlah': 5, 'jelaskan': 4, 'sebutkan': 4,
            'berikan': 3, 'tampilkan': 3, 'uraikan': 3, 'bandingkan': 4,
            'hitunglah': 4, 'tentukan': 3, 'gambarkan': 3,
            
            # Question words
            'bagaimana': 4, 'mengapa': 4, 'dimana': 3, 'kapan': 3, 'siapa': 3, 
            'apa': 2, 'berapa': 3, 'mana': 2,
            
            # Common Indonesian words
            'yang': 2, 'dan': 2, 'atau': 2, 'dengan': 2, 'untuk': 2,
            'dalam': 2, 'dari': 2, 'pada': 2, 'ke': 1, 'di': 2,
            'adalah': 2, 'akan': 2, 'dapat': 2, 'bisa': 2, 'harus': 2,
            'jika': 2, 'karena': 2, 'tetapi': 2, 'namun': 2, 'sehingga': 2,
            'membuat': 2, 'menggunakan': 2, 'menunjukkan': 3, 'menampilkan': 3,
            
            # Technical terms dalam konteks Indonesian
            'sederhana': 3, 'contoh': 2, 'kode': 2, 'program': 1,
            'fungsi': 2, 'method': 1, 'class': 1, 'constructor': 1,
            
            # Indonesian-specific patterns dan suffixes
            'nya': 2, 'lah': 2, 'kah': 2,
            
            # Compound patterns yang khas Indonesian
            'cara': 3, 'kerja': 2, 'saat': 2, 'sebuah': 3,
        }
        
        # English indicators dengan weights
        en_indicators = {
            # Question words
            'how': 4, 'what': 4, 'when': 3, 'where': 3, 'why': 4, 'who': 3, 'which': 3,
            
            # Action words
            'write': 3, 'create': 3, 'explain': 3, 'show': 3, 'demonstrate': 3,
            'compare': 3, 'describe': 3, 'define': 3, 'list': 3,
            
            # Common English words
            'the': 2, 'and': 2, 'or': 2, 'to': 2, 'of': 2, 'in': 2, 'for': 2,
            'with': 2, 'that': 2, 'this': 2, 'is': 2, 'are': 2, 'was': 2, 'were': 2,
            'does': 2, 'do': 2, 'can': 2, 'will': 2, 'would': 2, 'should': 2,
            'if': 2, 'but': 2, 'because': 2, 'so': 2, 'then': 2,
            
            # Technical terms
            'simple': 2, 'example': 2, 'code': 1, 'function': 2, 'method': 1,
            'class': 1, 'constructor': 1, 'program': 1, 'work': 2, 'works': 2,
            'when': 3, 'overwritten': 3, 'polymorphism': 1
        }
        
        query_lower = query.lower()
        
        # Calculate weighted scores
        id_score = 0
        en_score = 0
        
        for word, weight in id_indicators.items():
            if word in query_lower:
                id_score += weight
                
        for word, weight in en_indicators.items():
            if word in query_lower:
                en_score += weight
        
        # Additional pattern bonuses
        if re.search(r'\b(men|ber|ter|pe|se)\w+', query_lower):  # Indonesian prefixes
            id_score += 2
        
        if re.search(r'\w+(kan|lah|kah|nya)\b', query_lower):  # Indonesian suffixes
            id_score += 2
            
        if re.search(r'\b(ing|ed|ly|tion|ness)\b', query_lower):  # English suffixes
            en_score += 1
        
        # Calculate confidence berdasarkan score difference
        total_score = id_score + en_score
        if total_score == 0:
            confidence = 0.0
        else:
            score_diff = abs(id_score - en_score)
            confidence = score_diff / total_score
        
        return query_lower, id_score, en_score, confidence
    
    def _fallback_tie_breaker(self, query: str, id_score: int, en_score: int) -> str:
        """Advanced fallback methods untuk tie cases."""
        
        print(f"🔄 TIE DETECTED! Applying fallback methods...")
        print(f"   Indonesian: {id_score}, English: {en_score}")
        
        query_lower = query.lower()
        
        # FALLBACK 1: Indonesian sentence structure patterns
        print(f"📝 Fallback 1: Sentence structure analysis")
        
        # Indonesian "cara" + verb pattern
        if re.search(r'\bcara\s+\w+', query_lower):
            print(f"   ✅ Found Indonesian 'cara + verb' pattern")
            return 'id'
        
        # Indonesian "sebuah" + noun pattern
        if re.search(r'\bsebuah\s+\w+', query_lower):
            print(f"   ✅ Found Indonesian 'sebuah + noun' pattern")
            return 'id'
        
        # Indonesian "saat" + clause pattern
        if re.search(r'\bsaat\s+\w+', query_lower):
            print(f"   ✅ Found Indonesian 'saat + clause' pattern")
            return 'id'
        
        # English "how does" pattern
        if re.search(r'\bhow\s+does\s+\w+', query_lower):
            print(f"   ✅ Found English 'how does + subject' pattern")
            return 'en'
        
        # English "when a" pattern
        if re.search(r'\bwhen\s+a\s+\w+', query_lower):
            print(f"   ✅ Found English 'when a + noun' pattern")
            return 'en'
        
        # FALLBACK 2: Context-based analysis
        print(f"📚 Fallback 2: Context analysis")
        
        # Programming context dalam Indonesian educational setting
        programming_terms = ['polymorphism', 'inheritance', 'constructor', 'method', 'class', 'override']
        tech_term_count = sum(1 for term in programming_terms if term in query_lower)
        
        if tech_term_count > 0:
            # Jika contains tech terms + Indonesian question structure
            if any(word in query_lower for word in ['bagaimana', 'cara', 'saat', 'sebuah']):
                print(f"   ✅ Tech terms + Indonesian structure = Indonesian")
                return 'id'
        
        # FALLBACK 3: Word order analysis
        print(f"🔀 Fallback 3: Word order analysis")
        
        words = query.split()
        if len(words) >= 3:
            # Indonesian sering mulai dengan question word + specific structure
            first_word = words[0].lower()
            
            if first_word in ['bagaimana', 'mengapa', 'dimana']:
                print(f"   ✅ Starts with clear Indonesian question word")
                return 'id'
            
            if first_word in ['how', 'what', 'why', 'where'] and len(words) > 5:
                # Long English questions kurang umum dalam context ini
                print(f"   🤔 Long English question, might be Indonesian user translating")
                # Additional check: apakah mengandung Indonesian markers?
                if any(marker in query_lower for marker in ['cara', 'saat', 'sebuah', 'yang']):
                    print(f"   ✅ Contains Indonesian markers = Indonesian")
                    return 'id'
        
        # FALLBACK 4: Statistical patterns berdasarkan query characteristics
        print(f"📊 Fallback 4: Statistical analysis")
        
        # Indonesian educational queries cenderung lebih panjang dan deskriptif
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        print(f"   Word count: {word_count}, Avg word length: {avg_word_length:.1f}")
        
        if word_count > 8 and avg_word_length > 5:
            print(f"   ✅ Long, descriptive query = likely Indonesian")
            return 'id'
        
        # FALLBACK 5: LLM-based classification (jika available)
        if self.llm:
            print(f"🤖 Fallback 5: LLM classification")
            try:
                llm_result = self._llm_language_classification(query)
                if llm_result != 'unknown':
                    print(f"   ✅ LLM decided: {llm_result}")
                    return llm_result
            except Exception as e:
                print(f"   ❌ LLM classification failed: {e}")
        
        # FALLBACK 6: Context-based default dengan reasoning
        print(f"🎯 Fallback 6: Contextual default")
        
        # Dalam Indonesian educational context, bias towards Indonesian
        # Tapi check dulu untuk clear English patterns
        if any(pattern in query_lower for pattern in ['how to', 'what is', 'can you', 'show me']):
            print(f"   ✅ Clear English command pattern = English")
            return 'en'
        
        # Default ke Indonesian untuk educational context
        print(f"   ✅ Educational context default = Indonesian")
        return 'id'
    
    def _llm_language_classification(self, query: str) -> str:
        """Gunakan LLM untuk classify language sebagai final fallback."""
        
        classification_prompt = f"""Classify the language of this query. Consider the context and sentence structure.

Query: "{query}"

Respond with exactly one word: "INDONESIAN" or "ENGLISH" or "UNKNOWN"

Language:"""
        
        try:
            response = self.llm.predict(classification_prompt).strip().upper()
            if "INDONESIAN" in response:
                return 'id'
            elif "ENGLISH" in response:
                return 'en'
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def detect_language_with_smart_fallback(self, query: str) -> str:
        """Main detection method dengan smart fallback untuk tie cases."""
        
        print(f"🔍 Smart language detection for: '{query}'")
        
        # Step 1: Weighted scoring
        query_lower, id_score, en_score, confidence = self._weighted_score_detection(query)
        
        print(f"   Indonesian score: {id_score}")
        print(f"   English score: {en_score}")
        print(f"   Confidence: {confidence:.2f}")
        
        # Step 2: Decision logic
        score_diff = abs(id_score - en_score)
        
        # Clear winner (difference >= 2)
        if score_diff >= 2:
            if id_score > en_score:
                detected = 'id'
                print(f"   ✅ Clear Indonesian winner: {detected}")
            else:
                detected = 'en'
                print(f"   ✅ Clear English winner: {detected}")
            return detected
        
        # Close scores (difference = 1) - gunakan confidence
        elif score_diff == 1:
            if confidence > 0.3:  # Reasonable confidence
                if id_score > en_score:
                    detected = 'id'
                    print(f"   ✅ Narrow Indonesian lead with confidence: {detected}")
                else:
                    detected = 'en'
                    print(f"   ✅ Narrow English lead with confidence: {detected}")
                return detected
            else:
                print(f"   ⚖️ Narrow lead but low confidence, using fallback...")
                return self._fallback_tie_breaker(query, id_score, en_score)
        
        # Exact tie (difference = 0)
        else:
            print(f"   ⚖️ Perfect tie, using smart fallback...")
            return self._fallback_tie_breaker(query, id_score, en_score)

class RAGGenerator:
    """RAG Generator dengan Smart Fallback Language Detection - Ready to Use"""
    
    def __init__(self, config: Dict[str, Any], chat_memory=None):
        self.config = config
        self.chat_memory = chat_memory
        
        self.llm = ChatOpenAI(
            model_name=config["llm"]["model_name"],
            temperature=0.0,
            max_tokens=config["llm"]["max_tokens"],
            seed=42,
            model_kwargs={
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        )
        
        # Initialize smart detector dengan LLM support
        self.language_detector = SmartLanguageDetector(self.llm)
        
        # Enhanced prompts dengan stronger language enforcement
        self.prompts = {
            'en': PromptTemplate(
                input_variables=["context", "question", "chat_history", "query_guidance"],
                template="""CRITICAL INSTRUCTION: You must answer ONLY in ENGLISH. Do not use any Indonesian words.

CONTEXT: This chatbot is specifically designed to help students clarify concepts from their lessons.
Answer the question using the provided learning materials.

QUESTION ANALYSIS: {query_guidance}

LANGUAGE REQUIREMENT: Your entire response must be in ENGLISH language only.

WHEN ANSWERING STUDENT'S QUESTION:
1. Answer in ENGLISH (same language as the question)
2. Focus specifically on what the question is asking for
3. Use information from the learning materials that directly addresses the question
4. If the question asks for definitions, prioritize clear explanations
5. If the question asks for comparisons, structure your answer to show differences
6. Provide specific details and examples from the learning materials
7. Provide a comprehensive answer that directly addresses what the question is asking for
8. If no relevant information is found, clearly state this.

Previous conversation:
{chat_history}

LEARNING MATERIALS:
{context}

QUESTION: {question}

Based on the learning materials above, provide a comprehensive answer IN ENGLISH ONLY that directly addresses what the question is asking for:"""
            ),
            
            'id': PromptTemplate(
                input_variables=["context", "question", "chat_history", "query_guidance"],
                template="""INSTRUKSI PENTING: Anda harus menjawab HANYA dalam BAHASA INDONESIA. Jangan gunakan kata-kata Inggris kecuali istilah teknis yang sudah umum.

KONTEKS: Chatbot ini secara khusus dirancang untuk membantu siswa memahami konsep-konsep dari pelajaran mereka.
Jawab pertanyaan menggunakan materi pembelajaran yang disediakan.

ANALISIS PERTANYAAN: {query_guidance}

PERSYARATAN BAHASA: Seluruh jawaban Anda harus dalam BAHASA INDONESIA saja.

SAAT MENJAWAB PERTANYAAN SISWA:
1. Jawab dalam BAHASA INDONESIA (bahasa yang sama dengan pertanyaan)
2. Fokus secara spesifik pada apa yang ditanyakan dalam pertanyaan
3. Gunakan informasi dari materi pembelajaran yang langsung menjawab pertanyaan
4. Jika pertanyaan meminta definisi, prioritaskan penjelasan yang jelas
5. Jika pertanyaan meminta perbandingan, susun jawaban untuk menunjukkan perbedaan
6. Berikan detail spesifik dan contoh dari materi pembelajaran
7. Berikan jawaban komprehensif yang langsung menjawab apa yang ditanyakan
8. Jika tidak ditemukan informasi yang relevan, nyatakan dengan jelas.

Percakapan sebelumnya:
{chat_history}

MATERI PEMBELAJARAN:
{context}

PERTANYAAN: {question}

Berdasarkan materi pembelajaran di atas, berikan jawaban komprehensif DALAM BAHASA INDONESIA SAJA yang secara langsung menjawab apa yang ditanyakan:"""
            )
        }
    
    def _analyze_query_intent(self, query: str, language: str) -> str:
        """Analyze query intent untuk provide specific guidance."""
        query_lower = query.lower()
        
        # Definition questions
        if any(pattern in query_lower for pattern in ['what is', 'apa itu', 'define', 'definisi']):
            if language == 'id':
                return "Pertanyaan ini meminta DEFINISI. Prioritaskan penjelasan yang jelas tentang konsep, termasuk pengertian dasar dan karakteristik utama."
            else:
                return "This question asks for a DEFINITION. Prioritize clear explanations of the concept, including basic meaning and key characteristics."
        
        # Comparison questions
        elif any(pattern in query_lower for pattern in ['compare', 'comparison', 'vs', 'versus', 'bandingkan', 'perbandingan', 'perbedaan']):
            if language == 'id':
                return "Pertanyaan ini meminta PERBANDINGAN. Struktur jawaban dengan jelas menunjukkan perbedaan dan persamaan antara konsep yang dibandingkan."
            else:
                return "This question asks for a COMPARISON. Structure the answer to clearly show differences and similarities between the concepts being compared."
        
        # Process/method questions - Enhanced untuk Indonesian
        elif any(pattern in query_lower for pattern in ['how does', 'how to', 'bagaimana', 'cara', 'process', 'proses', 'kerja']):
            if language == 'id':
                return "Pertanyaan ini meminta penjelasan PROSES atau CARA KERJA. Fokus pada langkah-langkah, mekanisme, atau cara kerja yang spesifik dengan contoh yang jelas."
            else:
                return "This question asks about a PROCESS or HOW SOMETHING WORKS. Focus on specific steps, mechanisms, or how something works with clear examples."
        
        # Example/code questions - Enhanced untuk Indonesian
        elif any(pattern in query_lower for pattern in ['example', 'contoh', 'write', 'tuliskan', 'create', 'buatlah', 'buat', 'program']):
            if language == 'id':
                return "Pertanyaan ini meminta CONTOH atau PEMBUATAN PROGRAM. Berikan contoh kode program yang lengkap dan jelas dengan penjelasan step-by-step dalam bahasa Indonesia."
            else:
                return "This question asks for EXAMPLES or PROGRAM CREATION. Provide complete and clear code examples with step-by-step explanations."
        
        # General guidance
        else:
            if language == 'id':
                return "Pertanyaan umum tentang konsep pemrograman. Berikan jawaban komprehensif yang mencakup aspek-aspek penting dari topik yang ditanyakan dalam bahasa Indonesia yang mudah dipahami."
            else:
                return "General question about programming concepts. Provide a comprehensive answer covering important aspects of the asked topic in clear English."
    
    def _format_context_enhanced(self, documents: List[Document], query: str) -> str:
        """Enhanced context formatting dengan query relevance."""
        if not documents:
            return "No learning materials found."
        
        # Extract key terms dari query untuk relevance checking
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        # Remove stop words untuk both languages
        stop_words = {
            'how', 'what', 'when', 'where', 'why', 'the', 'and', 'for', 'with',
            'bagaimana', 'apa', 'kapan', 'dimana', 'mengapa', 'yang', 'dan', 'untuk', 'dengan',
            'cara', 'saat', 'sebuah'
        }
        query_terms = query_terms - stop_words
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            
            # Check relevance ke query
            content_lower = content.lower()
            relevance_score = sum(1 for term in query_terms if term in content_lower)
            
            # Truncate berdasarkan relevance
            if relevance_score > 0:
                max_length = 1500  # More content untuk relevant docs
            else:
                max_length = 800   # Less content untuk less relevant docs
            
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            source = doc.metadata.get('source', f'Document {i}')
            source_name = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
            
            # Mark relevance untuk LLM
            relevance_marker = "[HIGHLY RELEVANT]" if relevance_score >= 2 else "[RELEVANT]" if relevance_score >= 1 else ""
            
            formatted_docs.append(f"Source {i} ({source_name}) {relevance_marker}:\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    def generate(self, query: str, retrieved_documents: List[Document]) -> str:
        """Enhanced generation dengan smart fallback language detection."""
        print(f"🎯 Generating enhanced response dengan smart fallback")
        print(f"📚 Using {len(retrieved_documents)} documents")
        
        # Gunakan smart language detection dengan fallback
        query_language = self.language_detector.detect_language_with_smart_fallback(query)
        print(f"🌐 Final language decision: {query_language}")
        
        # Analyze query intent berdasarkan detected language
        query_guidance = self._analyze_query_intent(query, query_language)
        print(f"🧠 Query intent: {query_guidance[:70]}...")
        
        # Enhanced context formatting
        context = self._format_context_enhanced(retrieved_documents, query)
        
        # Simple chat history
        chat_history = ""
        if self.chat_memory:
            try:
                chat_history = self.chat_memory.get_context_from_history(max_messages=2)
            except:
                pass
        
        # Generate dengan query-aware prompting
        if not retrieved_documents:
            print("⚠️  No documents - using fallback")
            
            fallback_lang = "English" if query_language == 'en' else "Bahasa Indonesia"
            fallback_prompt = f"""Answer this educational question in {fallback_lang} based on general knowledge:

Query guidance: {query_guidance}

Question: {query}

Provide helpful educational guidance following the query guidance above.

Answer in {fallback_lang}:"""
            
            response = self.llm.predict(fallback_prompt)
        else:
            print("✅ Using retrieved documents dengan query-aware prompting")
            
            prompt_template = self.prompts[query_language]
            formatted_prompt = prompt_template.format(
                context=context,
                question=query,
                chat_history=chat_history,
                query_guidance=query_guidance
            )
            
            response = self.llm.predict(formatted_prompt)
        
        print("✅ Enhanced response generated dengan smart fallback")
        
        # Add ke chat memory jika available
        if self.chat_memory:
            try:
                self.chat_memory.add_user_message(query)
                self.chat_memory.add_assistant_message(response, [])
            except:
                pass
        
        return response

# Test function untuk verify smart fallback
def test_generator_with_problematic_queries():
    """Test generator dengan queries yang problematic sebelumnya."""
    
    # Mock config untuk testing
    config = {
        "llm": {
            "model_name": "gpt-3.5-turbo",
            "max_tokens": 1000
        }
    }
    
    # Initialize generator
    generator = RAGGenerator(config)
    
    # Test queries yang sebelumnya bermasalah
    test_queries = [
        "Tuliskan program Java sederhana yang menunjukkan constructor chaining",
        "Bagaimana cara kerja polymorphism saat sebuah metode di-overwrite?",
        "Jelaskan perbedaan antara method overloading dan overriding",
        "What is the difference between abstract class and interface?",
        "How does inheritance work in Java programming?"
    ]
    
    print("🧪 Testing Smart Fallback Generator")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔬 Test {i}: {query}")
        print("-" * 50)
        
        # Test hanya language detection (tanpa actual generation)
        detected_lang = generator.language_detector.detect_language_with_smart_fallback(query)
        print(f"📊 Final Result: {detected_lang}")
        
        expected = 'id' if any(word in query.lower() for word in ['tuliskan', 'bagaimana', 'jelaskan']) else 'en'
        status = "✅ CORRECT" if detected_lang == expected else "❌ WRONG"
        print(f"🎯 Status: {status} (Expected: {expected})")
        
        print("-" * 50)
    
    return "Test completed"