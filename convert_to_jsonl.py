#!/usr/bin/env python3
"""
Convert policies.json to policies.jsonl format
Each line = one Q&A pair or policy section
"""
import json

def convert_policies_to_jsonl(input_file='policies.json', output_file='policies.jsonl'):
    """Convert policies.json to JSONL format"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        policies = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for section, content in policies.items():
            if isinstance(content, list):
                # Q&A pairs from policy_parser
                for item in content:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        record = {
                            "section": section,
                            "question": item['question'],
                            "answer": item['answer']
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + '\n')
            elif isinstance(content, str) and content.strip():
                # Plain text policy - treat as single Q&A
                record = {
                    "section": section,
                    "question": f"What is the {section} policy?",
                    "answer": content.strip()
                }
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {input_file} to {output_file}")

if __name__ == '__main__':
    convert_policies_to_jsonl()
