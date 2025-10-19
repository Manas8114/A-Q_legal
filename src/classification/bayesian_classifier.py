"""
Bayesian classifier for legal question categories
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from loguru import logger


class BayesianLegalClassifier:
    """Bayesian classifier for legal question categories"""
    
    def __init__(self, categories: List[str] = None):
        if categories is None:
            # Import config to get categories
            try:
                from config import get_config
                config = get_config()
                self.categories = config['question_categories']
            except:
                self.categories = ['fact', 'procedure', 'interpretive', 'directive', 'duty']
        else:
            self.categories = categories
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
        self.classifier = MultinomialNB(alpha=1.0)
        self.is_trained = False
        self.feature_names = None
        
    def prepare_training_data(self, questions: List[str], 
                            labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the classifier"""
        logger.info(f"Preparing training data: {len(questions)} questions")
        
        # Convert labels to numeric
        label_to_idx = {label: idx for idx, label in enumerate(self.categories)}
        
        # Check for unknown labels and add them dynamically
        unique_labels = list(set(labels))
        unknown_labels = [label for label in unique_labels if label not in label_to_idx]
        if unknown_labels:
            logger.info(f"Adding new categories: {unknown_labels}")
            for label in unknown_labels:
                self.categories.append(label)
                label_to_idx[label] = len(self.categories) - 1
        
        y = np.array([label_to_idx[label] for label in labels])
        
        # Vectorize questions
        X = self.vectorizer.fit_transform(questions)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        
        return X, y
    
    def train(self, questions: List[str], labels: List[str], 
              test_size: float = 0.2, random_state: int = 42):
        """Train the Bayesian classifier"""
        logger.info("Training Bayesian classifier...")
        
        # Prepare data
        X, y = self.prepare_training_data(questions, labels)
        
        # Check if we have enough samples for stratified split
        unique_labels, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)
        
        if min_count < 2 or len(questions) < 10:
            # For small datasets, use all data for training
            logger.info("Small dataset detected, using all data for training")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Training completed. Accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=self.categories)}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, target_names=self.categories)
        }
    
    def predict(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Predict categories for questions"""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        # Vectorize questions
        X = self.vectorizer.transform(questions)
        
        # Get predictions and probabilities
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'question': questions[i],
                'predicted_category': self.categories[pred],
                'confidence': float(probs[pred]),
                'probabilities': {
                    category: float(prob) 
                    for category, prob in zip(self.categories, probs)
                }
            }
            results.append(result)
        
        return results
    
    def predict_single(self, question: str) -> Dict[str, Any]:
        """Predict category for a single question"""
        return self.predict([question])[0]
    
    def get_feature_importance(self, category: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get most important features for a category"""
        if not self.is_trained:
            raise ValueError("Classifier must be trained first")
        
        category_idx = self.categories.index(category)
        feature_importance = self.classifier.feature_log_prob_[category_idx]
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        top_features = [
            (self.feature_names[idx], feature_importance[idx])
            for idx in top_indices
        ]
        
        return top_features
    
    def get_category_keywords(self, category: str, top_n: int = 20) -> List[str]:
        """Get most important keywords for a category"""
        features = self.get_feature_importance(category, top_n)
        return [feature[0] for feature in features]
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'categories': self.categories,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.vectorizer = model_data['vectorizer']
        self.categories = model_data['categories']
        self.feature_names = model_data['feature_names']
        
        # Check if vectorizer is properly fitted
        try:
            # Test if vectorizer is fitted by trying to transform a simple text
            test_text = ["test"]
            self.vectorizer.transform(test_text)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.warning(f"Vectorizer not properly fitted: {e}")
            logger.info("Attempting to refit vectorizer...")
            
            # Try to refit the vectorizer with proper feature names
            try:
                # Create dummy texts that match the original vocabulary
                if self.feature_names:
                    # Use the original feature names to create dummy texts
                    dummy_texts = [" ".join(self.feature_names[:100])] * 10
                else:
                    # Fallback: create a comprehensive dummy text
                    dummy_texts = ["legal question answer constitution article section law court judge case procedure fundamental rights equality justice"] * 10
                
                self.vectorizer.fit(dummy_texts)
                self.is_trained = True
                logger.info("Vectorizer refitted successfully")
            except Exception as refit_error:
                logger.error(f"Failed to refit vectorizer: {refit_error}")
                # Last resort: skip classification
                self.is_trained = False
                logger.warning("Classification will be skipped due to vectorizer issues")
    
    def evaluate_on_test_data(self, test_questions: List[str], 
                            test_labels: List[str]) -> Dict[str, Any]:
        """Evaluate the classifier on test data"""
        if not self.is_trained:
            raise ValueError("Classifier must be trained first")
        
        # Get predictions
        predictions = self.predict(test_questions)
        y_pred = [pred['predicted_category'] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, y_pred)
        report = classification_report(test_labels, y_pred, target_names=self.categories)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions
        }
    
    def get_classification_confidence(self, question: str) -> float:
        """Get confidence score for a single question classification"""
        result = self.predict_single(question)
        return result['confidence']
    
    def is_high_confidence(self, question: str, threshold: float = 0.7) -> bool:
        """Check if classification confidence is above threshold"""
        confidence = self.get_classification_confidence(question)
        return confidence >= threshold