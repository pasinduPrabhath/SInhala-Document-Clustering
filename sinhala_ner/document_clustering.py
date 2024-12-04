import numpy as np
from gensim.models import Word2Vec
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
import requests
import logging
from typing import List, Dict, Any
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)

class LLaMaClusterNamer:
    def __init__(self, api_key: str, model: str = "llama-3.1-sonar-small-128k-online"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai/chat/completions"

    def _construct_naming_prompt(self, cluster_sentences: List[str]) -> str:
        sentences_text = "\n".join([f"- {sent}" for sent in cluster_sentences])
        return f"""
        You are an expert in Sinhala text analysis. Generate ONE concise cluster name.

        Top Cluster Sentences (Most Representative):
        {sentences_text}

        STRICT Rules:
        1. Return EXACTLY ONE name in format: "සිංහල නම (English Translation)"
        2. Maximum 3 words per language
        3. NO lists, bullets, or multiple names
        4. NO additional text or explanations
        5. MUST be descriptive of the main theme
        6. DO NOT include sentence numbers or bullet points
        7. PRIORITIZE the above sentences when determining the cluster name

        Example good outputs:
        "ආර්ථික පුවත් (Economic News)"
        "ක්‍රීඩා වාර්තා (Sports Reports)"
        "දේශපාලන සිදුවීම් (Political Events)"

        Generate cluster name:
        """

    def generate_cluster_name(self, cluster_sentences: List[str]) -> str:
        try:
            if not cluster_sentences:
                return "හිස් කාණ්ඩය (Empty Cluster)"

            prompt = self._construct_naming_prompt(cluster_sentences)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a Sinhala text analysis expert. Generate concise, accurate cluster names."},
                    {"role": "user", "content": prompt}
                ]
            }

            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()

            # Extract and clean the response
            cluster_name = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            # Remove any extra quotes
            cluster_name = cluster_name.replace('"', '').replace('"', '').strip()
            
            # Validate format (should contain both Sinhala and English)
            if '(' in cluster_name and ')' in cluster_name:
                return cluster_name
            else:
                return "වර්ගීකරණ නොකළ කාණ්ඩය (Uncategorized Cluster)"

        except Exception as e:
            logger.error(f"Error in generate_cluster_name: {e}")
            return "Technical Error"

class DocumentClusterer:
    def __init__(self, word2vec_model_path: str, api_key: str, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        self.cluster_namer = LLaMaClusterNamer(api_key)

    def elbow_method(self, doc_vectors, max_clusters=8):
        distortions = []
        silhouette_scores = []
        db_scores = []
        fpc_scores = []
        k_values = range(2, max_clusters + 1)

        for k in k_values:
            # Temporarily change n_clusters
            self.n_clusters = k
            
            # Perform clustering
            scaled_vectors = StandardScaler().fit_transform(doc_vectors)
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                scaled_vectors.T,
                c=k,
                m=1.6,
                error=0.005,
                maxiter=1000,
                seed=42
            )
            
            # Calculate metrics
            hard_labels = u.argmax(axis=0)
            center = np.array([doc_vectors[hard_labels == i].mean(axis=0) for i in range(k)])
            
            # Calculate distortion
            distortion = sum(np.min(np.linalg.norm(doc_vectors - center[:, np.newaxis], axis=2), axis=0))
            distortions.append(distortion)
            
            # Other metrics
            silhouette = silhouette_score(doc_vectors, hard_labels)
            db_score = davies_bouldin_score(doc_vectors, hard_labels)
            fpc = np.trace(u.dot(u.T)) / float(u.shape[1])
            
            silhouette_scores.append(silhouette)
            db_scores.append(db_score)
            fpc_scores.append(fpc)

        # Find optimal k using combined metrics
        normalized_scores = {
            'distortion': (distortions - np.min(distortions)) / (np.max(distortions) - np.min(distortions)),
            'silhouette': silhouette_scores,
            'db_index': (db_scores - np.min(db_scores)) / (np.max(db_scores) - np.min(db_scores)),
            'fpc': fpc_scores
        }

        # Combined score (weighted average)
        combined_scores = []
        for i in range(len(k_values)):
            score = (
                -0.25 * normalized_scores['distortion'][i] +  # Negative because lower is better
                0.3 * normalized_scores['silhouette'][i] +    # Higher is better
                -0.25 * normalized_scores['db_index'][i] +    # Negative because lower is better
                0.2 * normalized_scores['fpc'][i]             # Higher is better
            )
            combined_scores.append(score)

        optimal_k = k_values[np.argmax(combined_scores)]
        return optimal_k, {
            'k_values': k_values,
            'distortions': distortions,
            'silhouette_scores': silhouette_scores,
            'db_scores': db_scores,
            'fpc_scores': fpc_scores
        }

    def find_optimal_clusters(self, ner_results: List[Dict]) -> Dict[str, Any]:
        doc_vectors = self.create_document_vectors(ner_results)
        optimal_k, metrics = self.elbow_method(doc_vectors)
        
        # Update n_clusters with optimal value
        self.n_clusters = optimal_k
        print(f"Using n_clusters = {self.n_clusters} for clustering")
        # Perform clustering with optimal k
        return self.cluster_documents(ner_results)

    def create_document_vectors(self, ner_results: List[Dict]) -> np.ndarray:
        def sentence_vector(sentence_data):
            sentence = sentence_data['sentence']
            entities = sentence_data.get('detected_entities', [])
            
            words = sentence.lower().split()
            word_vectors = [self.word2vec_model.wv[word] 
                          for word in words 
                          if word in self.word2vec_model.wv]
            
            entity_weights = {
                'PERSON': 1.5,
                'ORGANIZATION': 1.3,
                'LOCATION': 1.2,
                'DATE': 1.1,
                'TIME': 1.1,
                'EVENT': 1.6
            }
            
            entity_vectors = []
            for entity in entities:
                entity_type = entity.get('type', '')
                entity_text = entity.get('text', '').lower().split()
                entity_vec = [self.word2vec_model.wv[word] 
                            for word in entity_text 
                            if word in self.word2vec_model.wv]
                if entity_vec:
                    weight = entity_weights.get(entity_type, 1.0)
                    weighted_vec = np.mean(entity_vec, axis=0) * weight
                    entity_vectors.append(weighted_vec)
            
            all_vectors = word_vectors + entity_vectors
            return np.mean(all_vectors, axis=0) if all_vectors else np.zeros(self.word2vec_model.vector_size)
        
        return np.array([sentence_vector(doc) for doc in ner_results])

    def cluster_documents(self, ner_results: List[Dict]) -> Dict[str, Any]:
        doc_vectors = self.create_document_vectors(ner_results)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(doc_vectors)
        
        # Fuzzy C-Means Clustering
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            scaled_vectors.T,
            c=self.n_clusters,
            m=1.6,
            error=0.005,
            maxiter=1000,
            seed=42
        )
        
        # Organize results by clusters
        clusters = []
        for cluster_idx in range(self.n_clusters):
            cluster_sentences = []
            
            for doc_idx, doc in enumerate(ner_results):
                membership_prob = u[cluster_idx, doc_idx]
                if membership_prob > 0.3:
                    cluster_sentences.append({
                        'sentence': doc['sentence'],
                        'membership_probability': float(membership_prob)
                    })
            
            if cluster_sentences:
                cluster_name = self.cluster_namer.generate_cluster_name(
                    [sent['sentence'] for sent in cluster_sentences]
                )
                
                clusters.append({
                    'cluster_name': cluster_name,
                    'sentences': sorted(
                        cluster_sentences,
                        key=lambda x: x['membership_probability'],
                        reverse=True
                    )
                })
        
        return {
            'total_clusters': self.n_clusters,
            'clusters': clusters
        }