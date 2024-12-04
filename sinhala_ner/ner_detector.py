import os
import json
import re
import requests
import logging
import spacy
from typing import List, Dict, Any

class SinhalaNERDetector:
    def __init__(self, api_key: str, model: str = "llama-3.1-sonar-small-128k-online"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.nlp = spacy.load('spacy_model')

    def _construct_ner_prompt(self, text: str) -> str:
        return f""" 
        You are an expert in Sinhala Named Entity Recognition. Carefully analyze the following Sinhala text and extract named entities.
        
        Entity Classification Guidelines:
        1. LOCATION: Cities, countries, regions, geographical areas
        2. PERSON: Individual names, personal identifiers
        3. ORGANIZATION: Companies, institutions, governmental bodies
        4. DATE: Specific dates, periods, years
        5. TIME: Specific time periods, hours, minutes
        6. EVENT: Events with following sub-types:
           - SPORTSEVENT: Sports matches, tournaments, competitions
           - POLITICALEVENT: Political gatherings, elections
           - ECONOMICEVENT: Economic summits, financial events
           - CULTURALEVENT: Cultural festivals, celebrations
        
        Text: {text}
        
        Respond STRICTLY in this JSON format:
        {{
            "entities": [
                {{
                    "text": "entity_name",
                    "type": "EVENT",
                    "sub_type": "SPORTSEVENT",  // Only for EVENT type
                    "start_index": 0,
                    "end_index": 10
                }}
            ]
        }}
        Note: For EVENT type entities, always specify the sub_type as one of: SPORTSEVENT, POLITICALEVENT, ECONOMICEVENT, CULTURALEVENT
        """

    def detect_entities(self, documents: List[str]) -> List[Dict[str, Any]]:
        results = []
        for doc in documents:
            try:
                # SpaCy Entities
                spacy_entities = self._extract_spacy_entities(doc)
                
                # Llama Entities
                llama_entities = self._detect_llama_entities(doc)
                
                # Combine Entities
                combined_entities = self._merge_entities(spacy_entities, llama_entities)
                
                results.append({
                    "sentence": doc,
                    "detected_entities": combined_entities
                })
            except Exception as e:
                logging.error(f"NER detection error: {e}")
                results.append({
                    "sentence": doc,
                    "detected_entities": []
                })
        return results

    def _extract_spacy_entities(self, sentence: str) -> List[Dict[str, Any]]:
        doc = self.nlp(sentence)
        return [
            {
                "text": ent.text,
                "type": ent.label_,
                "start_index": ent.start_char,
                "end_index": ent.end_char
            } for ent in doc.ents
        ]

    def _detect_llama_entities(self, doc: str) -> List[Dict[str, Any]]:
        prompt = self._construct_ner_prompt(doc)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Extract named entities"},
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response_data = response.json()
        
        return self._parse_ner_response(response_data)

    def _parse_ner_response(self, response_data):
        try:
            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    parsed_content = json.loads(json_match.group(0))
                else:
                    return []
            
            entities = parsed_content.get('entities', [])
            return entities
        except Exception as e:
            logging.error(f"Parsing error: {e}")
            return []

    def _merge_entities(self, spacy_entities, llama_entities):
        unique_entities = []
        seen_entities = set()
        
        for entity_list in [spacy_entities, llama_entities]:
            for entity in entity_list:
                if not entity or 'text' not in entity or 'type' not in entity:
                    continue
                
                entity_text = entity.get('text', '').strip().lower()
                entity_type = entity.get('type', '').upper()
                
                if not entity_text or not entity_type:
                    continue
                
                entity_key = (entity_text, entity_type)
                if entity_key not in seen_entities:
                    unique_entities.append(entity)
                    seen_entities.add(entity_key)
        
        return unique_entities