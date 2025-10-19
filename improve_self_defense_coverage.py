#!/usr/bin/env python3
"""
Improve Self-Defense Coverage in A-Qlegal System
Add comprehensive self-defense laws and related legal concepts
"""

import json
from pathlib import Path

def add_self_defense_documents():
    """Add comprehensive self-defense legal documents"""
    
    # Load existing enhanced dataset
    enhanced_file = Path("data/enhanced/enhanced_legal_documents.json")
    if enhanced_file.exists():
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    else:
        documents = []
    
    # Self-defense related documents
    self_defense_docs = [
        {
            "id": f"self_defense_{i+1}",
            "title": "Right to Private Defence - Section 96 IPC",
            "section": "Section 96 IPC",
            "category": "Criminal Law",
            "text": "Nothing is an offence which is done in the exercise of the right of private defence.",
            "simplified_summary": "You have the right to defend yourself and others from harm without it being considered a crime.",
            "real_life_example": "If someone attacks you with a weapon, you can use force to protect yourself without being charged with assault.",
            "punishment": "No punishment - it's a legal right",
            "keywords": ["self defense", "private defence", "right to defend", "protection", "Section 96"],
            "source": "Indian Penal Code"
        },
        {
            "id": f"self_defense_{i+2}",
            "title": "Right to Defend Body and Property - Section 97 IPC",
            "section": "Section 97 IPC", 
            "category": "Criminal Law",
            "text": "Every person has a right, subject to the restrictions contained in Section 99, to defend his own body, and the body of any other person, against any offence affecting the human body; and the property, whether movable or immovable, of himself or of any other person, against any act which is an offence falling under the definition of theft, robbery, mischief or criminal trespass.",
            "simplified_summary": "You can defend yourself, others, and property from criminal attacks including theft, robbery, and trespassing.",
            "real_life_example": "If someone tries to steal your car, you can use reasonable force to stop them and protect your property.",
            "punishment": "No punishment - it's a legal right",
            "keywords": ["body defence", "property defence", "theft", "robbery", "trespassing", "Section 97"],
            "source": "Indian Penal Code"
        },
        {
            "id": f"self_defense_{i+3}",
            "title": "Limitations on Right of Private Defence - Section 99 IPC",
            "section": "Section 99 IPC",
            "category": "Criminal Law", 
            "text": "There is no right of private defence in cases in which there is time to have recourse to the protection of the public authorities. The right of private defence in no case extends to the inflicting of more harm than it is necessary to inflict for the purpose of defence.",
            "simplified_summary": "You cannot claim self-defense if you had time to call police, or if you used more force than necessary.",
            "real_life_example": "If someone insults you and you have time to walk away or call police, you cannot attack them and claim self-defense.",
            "punishment": "May face criminal charges for excessive force",
            "keywords": ["limitations", "excessive force", "public authorities", "necessary force", "Section 99"],
            "source": "Indian Penal Code"
        },
        {
            "id": f"self_defense_{i+4}",
            "title": "When Right of Private Defence Extends to Causing Death - Section 100 IPC",
            "section": "Section 100 IPC",
            "category": "Criminal Law",
            "text": "The right of private defence of the body extends, under the restrictions mentioned in the last preceding section, to the voluntary causing of death or of any other harm to the assailant, if the offence which occasions the exercise of the right be of any of the descriptions hereinafter enumerated: (1) Such an assault as may reasonably cause the apprehension that death will otherwise be the consequence of such assault; (2) Such an assault as may reasonably cause the apprehension that grievous hurt will otherwise be the consequence of such assault; (3) An assault with the intention of committing rape; (4) An assault with the intention of gratifying unnatural lust; (5) An assault with the intention of kidnapping or abducting; (6) An assault with the intention of wrongfully confining a person, under circumstances which may reasonably cause him to apprehend that he will be unable to have recourse to the public authorities for his release.",
            "simplified_summary": "You can use deadly force in self-defense when facing serious threats like murder, rape, kidnapping, or grievous hurt.",
            "real_life_example": "If someone attacks you with a knife intending to kill you, you can use deadly force to stop them and it won't be considered murder.",
            "punishment": "No punishment - justified self-defense",
            "keywords": ["deadly force", "murder", "rape", "kidnapping", "grievous hurt", "Section 100"],
            "source": "Indian Penal Code"
        },
        {
            "id": f"self_defense_{i+5}",
            "title": "Self-Defense Against Murder - Section 300 IPC Exception 1",
            "section": "Section 300 IPC Exception 1",
            "category": "Criminal Law",
            "text": "Culpable homicide is not murder if the offender, whilst deprived of the power of self-control by grave and sudden provocation, causes the death of the person who gave the provocation or causes the death of any other person by mistake or accident in the course of the matter to which the provocation relates.",
            "simplified_summary": "If you kill someone in self-defense due to sudden provocation, it may not be considered murder.",
            "real_life_example": "If someone suddenly attacks your family and you kill them in the heat of the moment to protect your loved ones, it may not be murder.",
            "punishment": "May be reduced to culpable homicide not amounting to murder",
            "keywords": ["murder", "provocation", "self control", "culpable homicide", "Section 300"],
            "source": "Indian Penal Code"
        },
        {
            "id": f"self_defense_{i+6}",
            "title": "Burden of Proof in Self-Defense Cases",
            "section": "General Legal Principle",
            "category": "Criminal Law",
            "text": "In cases of self-defense, the burden of proof lies on the accused to establish that they acted in self-defense. The prosecution must prove the case beyond reasonable doubt, but the accused must show that their actions were justified under the right of private defence.",
            "simplified_summary": "If you claim self-defense, you must prove that your actions were justified and necessary for protection.",
            "real_life_example": "If you're charged with assault after a fight, you need to provide evidence (witnesses, injuries, threats) that you were defending yourself.",
            "punishment": "Must prove self-defense to avoid conviction",
            "keywords": ["burden of proof", "evidence", "justification", "witnesses", "defense"],
            "source": "Criminal Procedure Code"
        },
        {
            "id": f"self_defense_{i+7}",
            "title": "Self-Defense and Proportional Force",
            "section": "Section 99 IPC",
            "category": "Criminal Law",
            "text": "The right of private defence in no case extends to the inflicting of more harm than it is necessary to inflict for the purpose of defence. The harm inflicted must be proportional to the threat faced.",
            "simplified_summary": "You can only use as much force as necessary to stop the threat - no more, no less.",
            "real_life_example": "If someone pushes you, you cannot pull out a knife and stab them. You can only push them back or restrain them.",
            "punishment": "Excessive force may result in criminal charges",
            "keywords": ["proportional force", "necessary force", "excessive force", "threat level"],
            "source": "Indian Penal Code"
        },
        {
            "id": f"self_defense_{i+8}",
            "title": "Self-Defense in Property Protection",
            "section": "Section 97 IPC",
            "category": "Criminal Law",
            "text": "Every person has a right to defend his property, whether movable or immovable, against theft, robbery, mischief, or criminal trespass. This right extends to protecting property from immediate threat of damage or theft.",
            "simplified_summary": "You can use force to protect your property from theft, damage, or trespassing.",
            "real_life_example": "If someone tries to break into your house, you can use reasonable force to stop them and protect your home.",
            "punishment": "No punishment for reasonable property defense",
            "keywords": ["property protection", "theft", "robbery", "trespassing", "home defense"],
            "source": "Indian Penal Code"
        }
    ]
    
    # Add the new documents
    documents.extend(self_defense_docs)
    
    # Save updated dataset
    with open(enhanced_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added {len(self_defense_docs)} self-defense documents")
    print(f"📊 Total documents now: {len(documents)}")
    
    return len(documents)

def retrain_models():
    """Retrain TF-IDF models with new data"""
    import numpy as np
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Load updated data
    with open("data/enhanced/enhanced_legal_documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    # Prepare texts for TF-IDF
    texts = []
    for doc in documents:
        # Combine title, text, and keywords for better search
        combined_text = f"{doc.get('title', '')} {doc.get('text', '')} {' '.join(doc.get('keywords', []))}"
        texts.append(combined_text)
    
    # Train TF-IDF vectorizer
    print("🔄 Training TF-IDF vectorizer with enhanced data...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Save models
    with open('models/enhanced_tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    np.save('data/enhanced/enhanced_tfidf_matrix.npy', tfidf_matrix.toarray())
    
    print("✅ Models retrained and saved")
    print(f"📊 Matrix shape: {tfidf_matrix.shape}")

def main():
    """Main function to improve self-defense coverage"""
    print("🛡️ Improving Self-Defense Coverage in A-Qlegal System")
    print("=" * 60)
    
    # Add self-defense documents
    total_docs = add_self_defense_documents()
    
    # Retrain models
    retrain_models()
    
    print("\n✅ Self-Defense Coverage Improved!")
    print(f"📚 Total documents: {total_docs}")
    print("🔄 Models retrained with new data")
    print("\n🚀 The system should now better handle self-defense queries!")

if __name__ == "__main__":
    main()
