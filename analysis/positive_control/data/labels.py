import re,json
NAME=r"(Belmopan|Belize City)(?!\w)"
def classify(text):
 text=text.strip();labels=[];evidence=[];reason=[]
 p=re.match(r'^(?:(?:the city of |the capital city of |the capital city |the city )|(?:(?:also )?known as |(?:also )?called ))?'+NAME,text,re.I)
 if p:
  tail=text[p.end():];clause=text[:text.find('.')+1] if '.' in text else text
  if re.match(r'^\s*\([A-Z][A-Z\s]*\)',tail):reason.append('entity identity ambiguous: parenthesized organization-like acronym')
  elif re.search(r'\b(?:not|but|used to|mentioned|referenced)\b',clause,re.I):reason.append('negated, qualified or metatextual first clause: uncertainty retained')
  elif re.match(r'^\s+(?:was|is|used to)\b',tail,re.I):reason.append('entity followed by its own predicate; identity answer not automatically asserted')
  elif re.match(r'^(?:$|[.,;:\n]|\s+(?:which|itself)\b)',tail,re.I):
   labels.append('correct' if p.group(1).casefold()=='belmopan' else 'wrong');evidence.append(text[:p.end()]);reason.append('complete named identity in cloze answer slot')
  else:reason.append('unresolved continuation after name')
 explicit=r"(?:the )?(?:capital(?: city)? of Belize|Belize['’]s capital(?: city)?) is\s+"+NAME
 for m in re.finditer(explicit,text,re.I):
  before=text[max(0,m.start()-24):m.start()]
  if re.search(r'\b(?:not|former|formerly|previous|past)\b',before,re.I):continue
  labels.append('correct' if m.group(1).casefold()=='belmopan' else 'wrong');evidence.append(m.group())
 if 'wrong' in labels and 'correct' in labels:label='mixed_target_assertions'
 elif 'wrong' in labels:label='incorrect_target_assertion'
 elif 'correct' in labels:label='correct_target_assertion'
 elif re.match(r'^(?:also )?(?:mentioned|referenced)\b',text,re.I):label='nonresponsive_metatext'
 elif re.match(r'^(?:(?:primarily )?located in|situated in|in)\s+Belmopan\b',text,re.I):label='partial_locative_answer'
 else:label='unverified_target'
 return dict(target_label=label,target_evidence=json.dumps(evidence,ensure_ascii=False),audit_reason='; '.join(reason) or 'no supported complete affirmative target identity extracted',full_response_factuality='unverified',reviewer_type='machine-assisted rule verification, not human review')
