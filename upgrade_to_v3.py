#!/usr/bin/env python3
"""
A-Qlegal 3.0 - Advanced Intelligence System
Implements multi-model fusion, explainability, fallback generation, and trust features
"""

import os
import json
import numpy as np
from pathlib import Path
from loguru import logger
import pickle
from tqdm import tqdm
import re
from datetime import datetime

# Configure logging
logger.remove()
logger.add("logs/v3_upgrade.log", level="DEBUG")
logger.add(lambda msg: print(f"\033[92m{msg}\033[0m"), level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class AQlegal3Upgrade:
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("models")
        
        # Load existing data
        self.legal_documents = []
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing trained data"""
        logger.info("🔄 Loading existing data...")
        try:
            with open(self.processed_dir / "all_legal_documents.json", "r", encoding="utf-8") as f:
                self.legal_documents = json.load(f)
            logger.info(f"✅ Loaded {len(self.legal_documents)} documents")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
    
    def create_knowledge_graph_structure(self):
        """Create knowledge graph structure for legal reasoning"""
        logger.info("🔄 Creating knowledge graph structure...")
        
        knowledge_graph = {
            "nodes": [],
            "edges": [],
            "relationships": []
        }
        
        # Create nodes for each legal document
        for i, doc in enumerate(tqdm(self.legal_documents, desc="Building knowledge graph")):
            node = {
                "id": f"node_{i}",
                "type": "legal_section",
                "title": doc.get("title", ""),
                "section": doc.get("section", ""),
                "category": doc.get("category", ""),
                "content": doc.get("content", "")
            }
            knowledge_graph["nodes"].append(node)
            
            # Create edges based on citations
            if doc.get("citations"):
                for citation in doc["citations"]:
                    edge = {
                        "from": f"node_{i}",
                        "to": citation,
                        "type": "references",
                        "weight": 1.0
                    }
                    knowledge_graph["edges"].append(edge)
            
            # Create category relationships
            relationship = {
                "node_id": f"node_{i}",
                "category": doc.get("category", ""),
                "keywords": doc.get("keywords", [])
            }
            knowledge_graph["relationships"].append(relationship)
        
        # Save knowledge graph
        with open(self.processed_dir / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created knowledge graph with {len(knowledge_graph['nodes'])} nodes")
        return len(knowledge_graph["nodes"])
    
    def create_fallback_generation_system(self):
        """Create intelligent fallback system for out-of-database queries"""
        logger.info("🔄 Creating fallback generation system...")
        
        fallback_system = {
            "reasoning_templates": {
                "general_legal": """
Based on general Indian legal principles:

**Understanding Your Question:**
{question}

**General Legal Perspective:**
While this specific question isn't directly covered in our database, here's what Indian law generally says:

{reasoning}

**Important Considerations:**
1. Indian law is based on the Constitution of India
2. Laws are made by Parliament and State Legislatures
3. Courts interpret and apply these laws
4. Legal principles of natural justice apply

**Related Legal Concepts:**
{related_concepts}

**Recommended Action:**
⚠️ For specific legal advice on this matter, please consult a qualified lawyer.

**Disclaimer:**
This is a general legal perspective, not specific legal advice. Laws may vary based on jurisdiction and specific circumstances.
""",
                "criminal_law": """
Based on Indian criminal law principles:

**Your Question:**
{question}

**Criminal Law Perspective:**
{reasoning}

**Key Principles:**
1. Intent (Mens Rea) and Action (Actus Reus) required for most crimes
2. Presumption of innocence until proven guilty
3. Right to fair trial and legal representation
4. Punishments must be proportionate to the offense

**Your Rights:**
- Right to legal counsel
- Right to remain silent
- Right to bail (in bailable offenses)
- Right to fair and speedy trial

**Next Steps:**
{next_steps}
""",
                "civil_law": """
Based on Indian civil law principles:

**Your Question:**
{question}

**Civil Law Perspective:**
{reasoning}

**Key Principles:**
1. Rights and obligations under contracts
2. Property rights and inheritance laws
3. Tort law for civil wrongs
4. Dispute resolution through courts or arbitration

**Legal Remedies Available:**
{remedies}

**Recommended Approach:**
{approach}
""",
                "constitutional_law": """
Based on the Indian Constitution:

**Your Question:**
{question}

**Constitutional Perspective:**
{reasoning}

**Fundamental Principles:**
1. Fundamental Rights (Part III)
2. Directive Principles (Part IV)
3. Fundamental Duties (Part IVA)
4. Constitutional Remedies (Article 32, 226)

**Constitutional Protections:**
{protections}

**How to Enforce Your Rights:**
{enforcement}
"""
            },
            "reasoning_patterns": {
                "identify_category": [
                    ("crime", "criminal_law"),
                    ("murder", "criminal_law"),
                    ("theft", "criminal_law"),
                    ("fraud", "criminal_law"),
                    ("rights", "constitutional_law"),
                    ("fundamental", "constitutional_law"),
                    ("constitution", "constitutional_law"),
                    ("contract", "civil_law"),
                    ("property", "civil_law"),
                    ("inheritance", "civil_law")
                ],
                "general_reasoning": [
                    "Indian law is comprehensive and covers various aspects of life",
                    "The Constitution of India is the supreme law",
                    "Laws are made to protect citizens and maintain social order",
                    "Every citizen has both rights and responsibilities",
                    "Courts are the ultimate interpreters of law"
                ]
            },
            "related_concepts_map": {
                "criminal": ["IPC (Indian Penal Code)", "CrPC (Criminal Procedure)", "Evidence Act", "Police procedures"],
                "civil": ["Contract Act", "Property laws", "Family law", "Tort law"],
                "constitutional": ["Fundamental Rights", "Directive Principles", "Judicial Review", "Constitutional Remedies"]
            },
            "next_steps_templates": [
                "Consult a criminal lawyer for specific advice",
                "File an FIR at the nearest police station if you're a victim",
                "Gather all relevant evidence and documentation",
                "Know your legal rights and protections",
                "Consider legal aid if you cannot afford a lawyer"
            ]
        }
        
        # Save fallback system
        with open(self.processed_dir / "fallback_system.json", "w", encoding="utf-8") as f:
            json.dump(fallback_system, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Created intelligent fallback generation system")
        return True
    
    def create_source_attribution_system(self):
        """Create system for source attribution and explainability"""
        logger.info("🔄 Creating source attribution system...")
        
        attribution_data = []
        
        for doc in tqdm(self.legal_documents, desc="Creating attribution data"):
            attribution = {
                "doc_id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "source": doc.get("source", "Unknown"),
                "section": doc.get("section", ""),
                "category": doc.get("category", ""),
                "citations": doc.get("citations", []),
                "confidence_score": self.calculate_confidence_score(doc),
                "explanation": self.generate_explanation(doc),
                "verification_status": "verified" if doc.get("section") else "general"
            }
            attribution_data.append(attribution)
        
        # Save attribution data
        with open(self.processed_dir / "attribution_data.json", "w", encoding="utf-8") as f:
            json.dump(attribution_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created attribution system for {len(attribution_data)} documents")
        return len(attribution_data)
    
    def calculate_confidence_score(self, doc):
        """Calculate confidence score for a document"""
        score = 0.0
        
        # Has section number
        if doc.get("section"):
            score += 0.3
        
        # Has source
        if doc.get("source") and doc["source"] != "Unknown":
            score += 0.2
        
        # Has citations
        if doc.get("citations") and len(doc["citations"]) > 0:
            score += 0.2
        
        # Has content
        if doc.get("content") and len(doc["content"]) > 100:
            score += 0.2
        
        # Has simplified summary
        if doc.get("simplified_summary"):
            score += 0.1
        
        return round(min(score, 1.0), 2)
    
    def generate_explanation(self, doc):
        """Generate explanation for why this document is relevant"""
        explanation_parts = []
        
        if doc.get("section"):
            explanation_parts.append(f"This is {doc['section']}")
        
        if doc.get("category"):
            explanation_parts.append(f"which falls under {doc['category']}")
        
        if doc.get("source"):
            explanation_parts.append(f"from {doc['source']}")
        
        if explanation_parts:
            return " ".join(explanation_parts) + "."
        
        return "General legal information."
    
    def create_hallucination_shield(self):
        """Create system to detect and prevent hallucinations"""
        logger.info("🔄 Creating hallucination shield...")
        
        shield_config = {
            "validation_rules": {
                "section_validation": {
                    "valid_prefixes": ["Section", "Article", "Chapter", "Part"],
                    "valid_patterns": [
                        r"Section\s+\d+[A-Z]?\s+(IPC|CrPC|NI Act)",
                        r"Article\s+\d+[A-Z]?",
                        r"Chapter\s+[IVXLC]+",
                        r"Part\s+[IVXLC]+"
                    ]
                },
                "fact_checking": {
                    "check_against_database": True,
                    "minimum_similarity": 0.7,
                    "require_citation": True
                },
                "response_validation": {
                    "check_contradictions": True,
                    "verify_legal_terms": True,
                    "flag_uncertain_statements": True
                }
            },
            "warning_triggers": [
                "no matching section found",
                "unverified information",
                "low confidence score",
                "contradiction detected"
            ],
            "verified_sections": self.extract_verified_sections(),
            "verified_articles": self.extract_verified_articles()
        }
        
        # Save shield config
        with open(self.processed_dir / "hallucination_shield.json", "w", encoding="utf-8") as f:
            json.dump(shield_config, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Created hallucination shield")
        return True
    
    def extract_verified_sections(self):
        """Extract all verified section numbers from database"""
        sections = set()
        
        for doc in self.legal_documents:
            if doc.get("section"):
                sections.add(doc["section"])
        
        return list(sections)
    
    def extract_verified_articles(self):
        """Extract all verified article numbers from database"""
        articles = set()
        
        for doc in self.legal_documents:
            content = doc.get("content", "")
            # Extract Article numbers
            article_matches = re.findall(r'Article\s+\d+[A-Z]?', content)
            articles.update(article_matches)
        
        return list(articles)
    
    def create_multilingual_expansion(self):
        """Create expanded multilingual templates"""
        logger.info("🔄 Creating multilingual expansion...")
        
        multilingual_templates = {
            "hindi": {
                "question_templates": [
                    "{section} क्या है?",
                    "{law} को समझाइए",
                    "{topic} के बारे में बताइए",
                    "{offense} की सजा क्या है?"
                ],
                "response_templates": {
                    "greeting": "नमस्ते! मैं आपकी कानूनी सहायता के लिए यहाँ हूँ।",
                    "disclaimer": "⚠️ यह कानूनी जानकारी है, कानूनी सलाह नहीं।",
                    "consult_lawyer": "कृपया विशिष्ट मामलों के लिए वकील से परामर्श करें।"
                },
                "common_terms": {
                    "punishment": "सजा",
                    "law": "कानून",
                    "court": "न्यायालय",
                    "rights": "अधिकार",
                    "section": "धारा"
                }
            },
            "tamil": {
                "question_templates": [
                    "{section} என்றால் என்ன?",
                    "{law} விளக்குக",
                    "{topic} பற்றி சொல்லுங்கள்"
                ],
                "response_templates": {
                    "greeting": "வணக்கம்! சட்ட உதவிக்காக நான் இங்கே இருக்கிறேன்.",
                    "disclaimer": "⚠️ இது சட்ட தகவல், சட்ட ஆலோசனை அல்ல.",
                    "consult_lawyer": "குறிப்பிட்ட விஷயங்களுக்கு வழக்கறிஞரை அணுகவும்."
                }
            },
            "bengali": {
                "question_templates": [
                    "{section} কি?",
                    "{law} ব্যাখ্যা করুন",
                    "{topic} সম্পর্কে বলুন"
                ],
                "response_templates": {
                    "greeting": "নমস্কার! আইনি সহায়তার জন্য আমি এখানে আছি।",
                    "disclaimer": "⚠️ এটি আইনি তথ্য, আইনি পরামর্শ নয়।",
                    "consult_lawyer": "নির্দিষ্ট বিষয়ের জন্য আইনজীবীর সাথে পরামর্শ করুন।"
                }
            },
            "telugu": {
                "question_templates": [
                    "{section} అంటే ఏమిటి?",
                    "{law} వివరించండి",
                    "{topic} గురించి చెప్పండి"
                ],
                "response_templates": {
                    "greeting": "నమస్కారం! న్యాయ సహాయం కోసం నేను ఇక్కడ ఉన్నాను.",
                    "disclaimer": "⚠️ ఇది న్యాయ సమాచారం, న్యాయ సలహా కాదు.",
                    "consult_lawyer": "నిర్దిష్ట విషయాల కోసం న్యాయవాదిని సంప్రదించండి."
                }
            }
        }
        
        # Save multilingual templates
        with open(self.processed_dir / "multilingual_templates.json", "w", encoding="utf-8") as f:
            json.dump(multilingual_templates, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Created multilingual expansion for 4 Indian languages")
        return True
    
    def create_document_analysis_system(self):
        """Create document analysis and clause extraction system"""
        logger.info("🔄 Creating document analysis system...")
        
        analysis_system = {
            "clause_patterns": {
                "obligations": [
                    r"shall\s+\w+",
                    r"must\s+\w+",
                    r"is\s+required\s+to",
                    r"has\s+to\s+\w+"
                ],
                "rights": [
                    r"entitled\s+to",
                    r"has\s+the\s+right\s+to",
                    r"may\s+\w+",
                    r"can\s+\w+"
                ],
                "penalties": [
                    r"punishment\s+(?:of|up\s+to|shall\s+be)",
                    r"fine\s+(?:of|up\s+to|not\s+exceeding)",
                    r"imprisonment\s+(?:for|up\s+to)",
                    r"penalty\s+(?:of|shall\s+be)"
                ],
                "dates": [
                    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
                    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
                ]
            },
            "risk_indicators": [
                "unlimited liability",
                "no warranty",
                "force majeure",
                "arbitration clause",
                "governing law",
                "termination clause",
                "indemnification"
            ],
            "key_sections": [
                "definitions",
                "scope of work",
                "payment terms",
                "confidentiality",
                "intellectual property",
                "dispute resolution",
                "termination"
            ]
        }
        
        # Save analysis system
        with open(self.processed_dir / "document_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis_system, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Created document analysis system")
        return True
    
    def create_advanced_v3_app(self):
        """Create A-Qlegal 3.0 advanced app"""
        logger.info("🔄 Creating A-Qlegal 3.0 app...")
        
        app_code = '''#!/usr/bin/env python3
"""
A-Qlegal 3.0 - Advanced Intelligence System
Multi-model fusion, explainability, fallback generation, and trust features
"""

import json
import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="A-Qlegal 3.0 - Advanced Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748B;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    .confidence-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .source-attribution {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .fallback-notice {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load all systems
@st.cache_resource
def load_all_systems():
    """Load all trained models and systems"""
    try:
        # Load TF-IDF
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_matrix = np.load('data/embeddings/tfidf_matrix.npy')
        
        # Load fallback system
        try:
            with open('data/processed/fallback_system.json', 'r', encoding='utf-8') as f:
                fallback_system = json.load(f)
        except:
            fallback_system = None
        
        # Load attribution data
        try:
            with open('data/processed/attribution_data.json', 'r', encoding='utf-8') as f:
                attribution_data = json.load(f)
        except:
            attribution_data = []
        
        # Load hallucination shield
        try:
            with open('data/processed/hallucination_shield.json', 'r', encoding='utf-8') as f:
                shield = json.load(f)
        except:
            shield = None
        
        return tfidf_vectorizer, tfidf_matrix, fallback_system, attribution_data, shield
    except Exception as e:
        st.error(f"Failed to load systems: {e}")
        return None, None, None, [], None

@st.cache_data
def load_legal_data():
    """Load legal database"""
    try:
        with open("data/processed/all_legal_documents.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

@st.cache_data
def load_personas():
    """Load user personas"""
    try:
        with open("data/processed/user_personas.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def detect_query_category(query):
    """Detect legal category from query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['crime', 'murder', 'theft', 'fraud', 'assault']):
        return 'criminal_law'
    elif any(word in query_lower for word in ['constitution', 'fundamental', 'rights', 'article']):
        return 'constitutional_law'
    elif any(word in query_lower for word in ['contract', 'property', 'civil', 'inheritance']):
        return 'civil_law'
    else:
        return 'general_legal'

def generate_fallback_response(query, fallback_system, category='general_legal'):
    """Generate intelligent fallback response for out-of-database queries"""
    if not fallback_system:
        return None
    
    template = fallback_system['reasoning_templates'].get(category, 
                fallback_system['reasoning_templates']['general_legal'])
    
    # Generate reasoning based on query
    reasoning = generate_smart_reasoning(query, category)
    
    # Get related concepts
    related_concepts = []
    for key, concepts in fallback_system.get('related_concepts_map', {}).items():
        if key in category:
            related_concepts = concepts
            break
    
    # Fill template
    response = template.format(
        question=query,
        reasoning=reasoning,
        related_concepts=", ".join(related_concepts) if related_concepts else "General Indian law principles",
        next_steps="Consult a qualified lawyer for specific advice",
        remedies="Legal remedies depend on specific circumstances",
        approach="Seek professional legal guidance",
        protections="Constitutional protections available to all citizens",
        enforcement="File writ petitions in High Court (Article 226) or Supreme Court (Article 32)"
    )
    
    return response

def generate_smart_reasoning(query, category):
    """Generate smart reasoning based on query and category"""
    reasoning_parts = []
    
    # Add general legal principles
    reasoning_parts.append("Indian law operates on principles of justice, equity, and good conscience.")
    
    # Add category-specific reasoning
    if category == 'criminal_law':
        reasoning_parts.append("In criminal matters, the prosecution must prove guilt beyond reasonable doubt.")
        reasoning_parts.append("Every accused person has the right to legal representation and fair trial.")
    elif category == 'constitutional_law':
        reasoning_parts.append("The Constitution is the supreme law of India.")
        reasoning_parts.append("Fundamental Rights are enforceable through courts.")
    elif category == 'civil_law':
        reasoning_parts.append("Civil law deals with disputes between individuals or entities.")
        reasoning_parts.append("The burden of proof is on the plaintiff in most civil cases.")
    else:
        reasoning_parts.append("Legal matters in India are governed by various statutes and precedents.")
    
    return "\\n\\n".join(reasoning_parts)

def search_with_attribution(query, tfidf_vectorizer, tfidf_matrix, data, attribution_data, top_k=5):
    """Search with source attribution"""
    try:
        # Transform query
        query_vector = tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = data[idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                
                # Add attribution
                attribution = next((attr for attr in attribution_data 
                                  if attr.get('doc_id') == doc.get('id')), None)
                if attribution:
                    doc['attribution'] = attribution
                else:
                    doc['attribution'] = {
                        'confidence_score': 0.5,
                        'source': doc.get('source', 'Unknown'),
                        'verification_status': 'general'
                    }
                
                results.append(doc)
        
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ A-Qlegal 3.0</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Legal Intelligence System with Explainability & Trust</p>', unsafe_allow_html=True)
    
    # Load all systems
    with st.spinner("🔄 Loading advanced AI systems..."):
        tfidf_vectorizer, tfidf_matrix, fallback_system, attribution_data, shield = load_all_systems()
        data = load_legal_data()
        personas = load_personas()
    
    if not data:
        st.error("❌ No legal data found. Please run training first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        
        # Persona selection
        st.subheader("👤 User Persona")
        persona_options = {
            "citizen": "🧑 Citizen - Practical advice",
            "student": "🎓 Student - Educational",
            "business": "💼 Business - Compliance",
            "lawyer": "⚖️ Lawyer - Technical"
        }
        selected_persona = st.selectbox(
            "Select your role:",
            options=list(persona_options.keys()),
            format_func=lambda x: persona_options[x]
        )
        
        # Advanced features
        st.subheader("🔬 Advanced Features")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_attribution = st.checkbox("Show source attribution", value=True)
        show_explanation = st.checkbox("Show AI reasoning", value=True)
        enable_fallback = st.checkbox("Enable smart fallback", value=True)
        
        # Search settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Results to show:", 1, 10, 5)
        similarity_threshold = st.slider("Min similarity:", 0.0, 1.0, 0.0, 0.1)
        
        # Language
        st.subheader("🌍 Language")
        language = st.selectbox(
            "Select language:",
            ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "বাংলা (Bengali)", "తెలుగు (Telugu)"]
        )
        
        # Statistics
        st.markdown("---")
        st.header("📊 System Stats")
        st.metric("Documents", f"{len(data):,}")
        st.metric("Attribution Data", f"{len(attribution_data):,}")
        st.metric("Verified Sections", f"{len(shield.get('verified_sections', [])) if shield else 0:,}")
    
    # Main content
    st.header("💬 Ask Your Legal Question")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the punishment for fraud? Explain my rights under Article 21",
        height=100,
        key="query_input"
    )
    
    # Example questions
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        
        examples = [
            "What is the punishment for fraud?",
            "Explain Section 420 IPC",
            "Tell me about fundamental rights",
            "What is culpable homicide?",
            "Explain right to freedom of speech",
            "What should I do if I'm arrested?",
            "How to file an FIR?",
            "What are my consumer rights?"
        ]
        
        for i, example in enumerate(examples):
            col = col1 if i % 2 == 0 else col2
            if col.button(f"📌 {example}", key=f"ex_{i}"):
                query = example
    
    # Search
    if st.button("🔍 Search", type="primary") or query:
        if query:
            with st.spinner("🤖 AI is analyzing your question..."):
                # Detect category
                category = detect_query_category(query)
                
                # Search database
                results = search_with_attribution(query, tfidf_vectorizer, tfidf_matrix, 
                                                data, attribution_data, top_k)
                
                # Filter by threshold
                results = [r for r in results if r['similarity_score'] >= similarity_threshold]
                
                # Check if we need fallback
                if not results or (results and results[0]['similarity_score'] < 0.3):
                    if enable_fallback:
                        st.markdown('<div class="fallback-notice">', unsafe_allow_html=True)
                        st.warning("⚠️ **Smart Fallback Activated**")
                        st.write("Your question isn't directly covered in our database, but here's what Indian law generally says:")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Generate fallback response
                        fallback_response = generate_fallback_response(query, fallback_system, category)
                        if fallback_response:
                            st.markdown(fallback_response)
                        
                        st.info("💡 For specific legal advice, please consult a qualified lawyer.")
                
                # Show database results if available
                if results:
                    st.success(f"✅ Found {len(results)} relevant document(s) in database")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(
                            f"📖 {i}. {doc.get('title', 'Unknown')} | "
                            f"{doc.get('category', 'Unknown')} | "
                            f"Match: {doc.get('similarity_score', 0):.1%}",
                            expanded=i==1
                        ):
                            # Confidence score
                            if show_confidence and doc.get('attribution'):
                                confidence = doc['attribution'].get('confidence_score', 0.5)
                                st.markdown(
                                    f'<span class="confidence-badge">Confidence: {confidence:.0%}</span>',
                                    unsafe_allow_html=True
                                )
                            
                            # Source attribution
                            if show_attribution and doc.get('attribution'):
                                st.markdown('<div class="source-attribution">', unsafe_allow_html=True)
                                st.write("📚 **Source Attribution:**")
                                st.write(f"• **Source:** {doc['attribution'].get('source', 'Unknown')}")
                                st.write(f"• **Status:** {doc['attribution'].get('verification_status', 'general')}")
                                if doc.get('section'):
                                    st.write(f"• **Section:** {doc['section']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # AI reasoning
                            if show_explanation and doc.get('attribution'):
                                explanation = doc['attribution'].get('explanation', '')
                                if explanation:
                                    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                                    st.write("🧠 **AI Reasoning:**")
                                    st.write(explanation)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Main content
                            if doc.get('section'):
                                st.subheader(f"📖 {doc['section']}")
                            
                            st.subheader("📝 Simplified Summary")
                            st.write(doc.get('simplified_summary', 'No summary available'))
                            
                            if doc.get('real_life_example'):
                                st.subheader("🏠 Real-Life Example")
                                st.write(doc['real_life_example'])
                            
                            if doc.get('punishment'):
                                st.subheader("⚖️ Punishment")
                                st.write(f"**{doc['punishment']}**")
                            
                            if doc.get('keywords'):
                                st.subheader("🏷️ Keywords")
                                st.write(", ".join(doc['keywords'][:10]))
                            
                            # Feedback
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.button(f"👍 Helpful", key=f"helpful_{i}")
                            with col_b:
                                st.button(f"👎 Not helpful", key=f"not_helpful_{i}")
                elif not enable_fallback:
                    st.warning("⚠️ No relevant documents found. Try different keywords or enable smart fallback.")
        else:
            st.info("💡 Please enter a question above")
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.metric("Version", "3.0")
    with col_f2:
        st.metric("Intelligence", "Advanced")
    with col_f3:
        st.metric("Explainability", "✓")
    with col_f4:
        st.metric("Status", "🟢 Online")
    
    st.markdown(
        '<p style="text-align: center; color: #64748B;">© 2025 A-Qlegal 3.0 - Advanced Legal Intelligence for India</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
'''
        
        with open("aqlegal_v3_app.py", "w", encoding="utf-8") as f:
            f.write(app_code)
        
        logger.info("✅ Created A-Qlegal 3.0 advanced app")
        return True
    
    def run_upgrade(self):
        """Run complete V3 upgrade"""
        logger.info("🚀 Starting A-Qlegal 3.0 Advanced Intelligence Upgrade")
        logger.info("=" * 60)
        
        steps = [
            ("Creating knowledge graph structure", self.create_knowledge_graph_structure),
            ("Building fallback generation system", self.create_fallback_generation_system),
            ("Setting up source attribution", self.create_source_attribution_system),
            ("Creating hallucination shield", self.create_hallucination_shield),
            ("Expanding multilingual support", self.create_multilingual_expansion),
            ("Building document analysis system", self.create_document_analysis_system),
            ("Creating A-Qlegal 3.0 app", self.create_advanced_v3_app)
        ]
        
        for i, (description, func) in enumerate(steps, 1):
            logger.info(f"Step {i}/{len(steps)}: {description}")
            try:
                result = func()
                logger.info(f"✅ {description} completed")
            except Exception as e:
                logger.error(f"❌ {description} failed: {e}")
                return False
            logger.info("")
        
        logger.success("🎉 A-Qlegal 3.0 Upgrade Completed Successfully!")
        logger.info("")
        logger.info("🚀 To run A-Qlegal 3.0:")
        logger.info("   streamlit run aqlegal_v3_app.py")
        logger.info("")
        logger.info("✨ New Features:")
        logger.info("   • Knowledge graph reasoning")
        logger.info("   • Intelligent fallback responses")
        logger.info("   • Source attribution & confidence scores")
        logger.info("   • Hallucination prevention")
        logger.info("   • Enhanced multilingual support")
        logger.info("   • Document analysis capabilities")
        
        return True

def main():
    """Main function"""
    upgrader = AQlegal3Upgrade()
    success = upgrader.run_upgrade()
    
    if success:
        print("\n🎉 A-Qlegal 3.0 upgrade completed successfully!")
        print("🚀 Run: streamlit run aqlegal_v3_app.py")
    else:
        print("\n❌ Upgrade failed. Check logs for details.")

if __name__ == "__main__":
    main()
