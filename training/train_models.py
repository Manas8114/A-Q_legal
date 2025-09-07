"""
Training script for Legal QA System models
"""
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.main import LegalQASystem
from loguru import logger

def create_sample_data():
    """Create sample data for training"""
    logger.info("Creating sample data...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample Constitution data
    constitution_data = [
        {
            "question": "What is the fundamental right to equality under Article 14?",
            "answer": "Article 14 guarantees equality before law and equal protection of laws to all persons within the territory of India. It prohibits discrimination on grounds of religion, race, caste, sex, or place of birth.",
            "context": "Article 14 of the Indian Constitution states that the State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India. This fundamental right ensures that all citizens are treated equally by the law and prohibits arbitrary discrimination.",
            "article": "Article 14",
            "category": "interpretive"
        },
        {
            "question": "What are the fundamental rights guaranteed under Part III of the Constitution?",
            "answer": "Part III of the Constitution guarantees six fundamental rights: Right to Equality (Articles 14-18), Right to Freedom (Articles 19-22), Right against Exploitation (Articles 23-24), Right to Freedom of Religion (Articles 25-28), Cultural and Educational Rights (Articles 29-30), and Right to Constitutional Remedies (Article 32).",
            "context": "Part III of the Indian Constitution contains the fundamental rights which are essential for the development of human personality and are considered basic to the democratic way of life. These rights are justiciable and can be enforced through the courts.",
            "article": "Part III",
            "category": "fact"
        }
    ]
    
    # Sample CrPC data
    crpc_data = [
        {
            "question": "What is the procedure for filing an FIR under CrPC?",
            "answer": "An FIR (First Information Report) can be filed by any person who has knowledge of a cognizable offence. The person can go to the nearest police station and provide information orally or in writing. The police officer must record the information and read it back to the informant.",
            "context": "Section 154 of the CrPC deals with the information in cognizable cases. It states that every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction.",
            "section": "Section 154",
            "category": "procedure"
        }
    ]
    
    # Sample IPC data
    ipc_data = [
        {
            "question": "What is the punishment for theft under IPC?",
            "answer": "Theft is punishable under Section 379 of the IPC with imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
            "context": "Section 379 of the Indian Penal Code defines theft as whoever, intending to take dishonestly any moveable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft.",
            "section": "Section 379",
            "category": "fact"
        }
    ]
    
    # Save data
    with open(data_dir / "constitution_qa.json", "w") as f:
        json.dump(constitution_data, f, indent=2)
    
    with open(data_dir / "crpc_qa.json", "w") as f:
        json.dump(crpc_data, f, indent=2)
    
    with open(data_dir / "ipc_qa.json", "w") as f:
        json.dump(ipc_data, f, indent=2)
    
    logger.info("Sample data created successfully")

def main():
    """Main training function"""
    logger.info("Starting Legal QA System training...")
    
    # Create sample data if it doesn't exist
    if not Path("data/constitution_qa.json").exists():
        create_sample_data()
    
    # Initialize system
    system = LegalQASystem()
    
    # Dataset paths
    dataset_paths = {
        'constitution': 'data/constitution_qa.json',
        'crpc': 'data/crpc_qa.json',
        'ipc': 'data/ipc_qa.json'
    }
    
    # Check which datasets exist
    available_datasets = {}
    for name, path in dataset_paths.items():
        if Path(path).exists():
            available_datasets[name] = path
            logger.info(f"Found dataset: {name}")
        else:
            logger.warning(f"Dataset not found: {name}")
    
    if not available_datasets:
        logger.error("No datasets found. Please create sample data first.")
        return
    
    # Initialize system with available datasets
    try:
        system.initialize_system(available_datasets)
        logger.info("System initialized successfully")
        
        # Get system status
        status = system.get_system_status()
        logger.info(f"System status: {status}")
        
        # Save the trained system
        system.save_system("trained_legal_qa_system")
        logger.info("Trained system saved successfully")
        
        # Test the system
        logger.info("Testing the system...")
        test_questions = [
            "What is the fundamental right to equality?",
            "How to file an FIR?",
            "What is the punishment for theft?"
        ]
        
        for question in test_questions:
            try:
                result = system.ask_question(question)
                logger.info(f"Q: {question}")
                logger.info(f"A: {result['answer'][:100]}...")
                logger.info(f"Confidence: {result['confidence']:.3f}")
            except Exception as e:
                logger.error(f"Error testing question '{question}': {e}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()