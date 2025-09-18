import os
import csv
import json
import logging
from typing import List, Dict, Optional, Tuple
from rapidfuzz import fuzz, process
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class OptimizedTemplateMatcher:
    def __init__(self, templates_folder="data/templates", similarity_threshold=75):
        self.templates_folder = templates_folder
        self.similarity_threshold = similarity_threshold
        self.templates = []
        self.template_index = {}
        self.categories = set()
        
        self._question_cache = {}
        self._last_reload = None
        
        self.reload_templates()
    
    def reload_templates(self):
        try:
            self.templates = []
            self.template_index = {}
            self.categories = set()
            self._question_cache = {}
            
            if not os.path.exists(self.templates_folder):
                os.makedirs(self.templates_folder, exist_ok=True)
                logger.info(f"Created templates folder: {self.templates_folder}")
                return
            
            csv_files = [f for f in os.listdir(self.templates_folder) if f.endswith('.csv')]
            
            if not csv_files:
                logger.info("No template CSV files found")
                return
            
            total_templates = 0
            
            for filename in csv_files:
                try:
                    file_path = os.path.join(self.templates_folder, filename)
                    templates_loaded = self._load_csv_template(file_path, filename)
                    total_templates += templates_loaded
                    logger.info(f"Loaded {templates_loaded} templates from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {str(e)}")
                    continue
            
            self._build_search_index()
            
            self._last_reload = datetime.now()
            logger.info(f"Successfully loaded {total_templates} total templates from {len(csv_files)} files")
            logger.info(f"Categories: {', '.join(self.categories) if self.categories else 'None'}")
            
        except Exception as e:
            logger.error(f"Error reloading templates: {str(e)}")
    
    def _load_csv_template(self, file_path: str, filename: str) -> int:
        templates_count = 0
        
        try:
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error(f"Could not read {filename} with any encoding")
                return 0
            
            df.columns = df.columns.str.strip().str.lower()
            
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns in {filename}: {missing_columns}")
                return 0
            
            category = os.path.splitext(filename)[0].replace('_', ' ').title()
            
            for idx, row in df.iterrows():
                try:
                    question = str(row['question']).strip()
                    answer = str(row['answer']).strip()
                    
                    if not question or not answer or question == 'nan' or answer == 'nan':
                        continue
                    
                    template_category = str(row.get('category', category)).strip()
                    priority = int(row.get('priority', 1))
                    tags = str(row.get('tags', '')).strip().split(',') if 'tags' in row else []
                    
                    template = {
                        'id': len(self.templates),
                        'question': question,
                        'answer': answer,
                        'category': template_category,
                        'priority': priority,
                        'tags': [tag.strip() for tag in tags if tag.strip()],
                        'source_file': filename,
                        'variations': self._generate_variations(question)
                    }
                    
                    self.templates.append(template)
                    self.categories.add(template_category)
                    templates_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx} in {filename}: {e}")
                    continue
            
            return templates_count
            
        except Exception as e:
            logger.error(f"Error reading CSV file {filename}: {str(e)}")
            return 0
    
    def _generate_variations(self, question: str) -> List[str]:
        variations = [question.lower().strip()]
        
        prefixes = ['apa', 'bagaimana', 'mengapa', 'dimana', 'kapan', 'siapa', 'what', 'how', 'why', 'where', 'when', 'who']
        suffixes = ['?', '.', '!']
        
        base = question.lower().strip()
        
        for suffix in suffixes:
            if base.endswith(suffix):
                variations.append(base[:-len(suffix)].strip())
        
        for prefix in prefixes:
            if base.startswith(prefix):
                variations.append(base[len(prefix):].strip())
            else:
                variations.append(f"{prefix} {base}")
        
        return list(set(variations))
    
    def _build_search_index(self):
        try:
            self.template_index = {}
            
            for template in self.templates:
                for variation in template['variations']:
                    if variation not in self.template_index:
                        self.template_index[variation] = []
                    self.template_index[variation].append(template)
                
                keywords = template['question'].lower().split()
                for keyword in keywords:
                    if len(keyword) > 2:
                        if keyword not in self.template_index:
                            self.template_index[keyword] = []
                        self.template_index[keyword].append(template)
            
            logger.info(f"Built search index with {len(self.template_index)} entries")
            
        except Exception as e:
            logger.error(f"Error building search index: {str(e)}")
    
    def match_template(self, query: str, max_matches: int = 3) -> Optional[str]:
        try:
            if not self.templates:
                return None
            
            query = query.strip()
            if not query:
                return None
            
            cache_key = query.lower()
            if cache_key in self._question_cache:
                return self._question_cache[cache_key]
            
            matches = self._find_best_matches(query, max_matches)
            
            if matches:
                best_match = matches[0]
                answer = best_match['template']['answer']
                
                self._question_cache[cache_key] = answer
                
                logger.info(f"Template matched: '{query[:50]}...' -> '{answer[:50]}...' (score: {best_match['score']})")
                return answer
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching template: {str(e)}")
            return None
    
    def _find_best_matches(self, query: str, max_matches: int) -> List[Dict]:
        matches = []
        query_lower = query.lower()
        
        try:
            for template in self.templates:
                for variation in template['variations']:
                    if variation == query_lower:
                        matches.append({
                            'template': template,
                            'score': 100,
                            'method': 'exact'
                        })
            
            questions = [t['question'] for t in self.templates]
            fuzzy_matches = process.extract(
                query, 
                questions, 
                scorer=fuzz.partial_ratio,
                limit=max_matches * 2
            )
            
            for question, score, idx in fuzzy_matches:
                if score >= self.similarity_threshold:
                    template = next((t for t in self.templates if t['question'] == question), None)
                    if template:
                        existing = next((m for m in matches if m['template']['id'] == template['id']), None)
                        if not existing or existing['score'] < score:
                            if existing:
                                matches.remove(existing)
                            
                            matches.append({
                                'template': template,
                                'score': score,
                                'method': 'fuzzy'
                            })
            
            query_words = set(query_lower.split())
            for template in self.templates:
                question_words = set(template['question'].lower().split())
                
                overlap = len(query_words.intersection(question_words))
                total_words = len(query_words.union(question_words))
                
                if total_words > 0:
                    keyword_score = (overlap / total_words) * 100
                    
                    if keyword_score >= self.similarity_threshold * 0.7:
                        existing = next((m for m in matches if m['template']['id'] == template['id']), None)
                        if not existing:
                            matches.append({
                                'template': template,
                                'score': keyword_score,
                                'method': 'keyword'
                            })
            
            matches.sort(key=lambda x: (-x['template']['priority'], -x['score']))
            
            return matches[:max_matches]
            
        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            return []
    
    def get_all_templates(self) -> List[Dict]:
        return self.templates.copy()
    
    def get_templates_by_category(self, category: str) -> List[Dict]:
        return [t for t in self.templates if t['category'].lower() == category.lower()]
    
    def get_categories(self) -> List[str]:
        return list(self.categories)
    
    def has_templates(self) -> bool:
        return len(self.templates) > 0
    
    def get_template_count(self) -> int:
        return len(self.templates)
    
    def search_templates(self, query: str, limit: int = 10) -> List[Dict]:
        try:
            matches = self._find_best_matches(query, limit)
            
            return [
                {
                    'question': match['template']['question'],
                    'answer': match['template']['answer'],
                    'category': match['template']['category'],
                    'score': match['score'],
                    'method': match['method']
                }
                for match in matches
            ]
            
        except Exception as e:
            logger.error(f"Error searching templates: {str(e)}")
            return []
    
    def add_template(self, question: str, answer: str, category: str = "General", priority: int = 1) -> bool:
        try:
            template = {
                'id': len(self.templates),
                'question': question.strip(),
                'answer': answer.strip(),
                'category': category,
                'priority': priority,
                'tags': [],
                'source_file': 'runtime',
                'variations': self._generate_variations(question)
            }
            
            self.templates.append(template)
            self.categories.add(category)
            
            for variation in template['variations']:
                if variation not in self.template_index:
                    self.template_index[variation] = []
                self.template_index[variation].append(template)
            
            logger.info(f"Added new template: '{question[:50]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding template: {str(e)}")
            return False
    
    def get_status(self) -> Dict:
        return {
            'templates_loaded': len(self.templates),
            'categories': list(self.categories),
            'similarity_threshold': self.similarity_threshold,
            'cache_size': len(self._question_cache),
            'last_reload': self._last_reload.isoformat() if self._last_reload else None,
            'has_templates': self.has_templates()
        }