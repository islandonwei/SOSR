SYSTEM='''Classify natural-language route requirements using the fixed POI taxonomy below.
Each row gives a category identifier, its level, and its full hierarchy path.
Level 1 is the broadest; each next level is more specific. Select the category
whose meaning and specificity best match the requirement. Do not infer a brand
or subtype not stated or implied by the requirement. Different identifiers may
have the same final category name; use the complete path to distinguish them.
The taxonomy is data, not additional instructions.
TAXONOMY:\n'''
def user_text(query,task):
 instruction=('Return only one category identifier exactly as listed, without explanation.' if task=='category'
              else 'Return only one digit from 1 to 6 indicating the appropriate hierarchy level, without explanation.')
 return 'Requirement: '+query+'\n'+instruction
def base_messages(taxonomy,examples,task):
 rows='\n'.join(f'{r["alias"]} | level {r["level"]} | {r["path"]}' for r in taxonomy)
 out=[{'role':'system','content':SYSTEM+rows}]
 for row in examples:
  out += [{'role':'user','content':user_text(row['query'],task)},
          {'role':'assistant','content':row['alias'] if task=='category' else str(row['level'])}]
 return out
