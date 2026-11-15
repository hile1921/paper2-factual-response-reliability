"""Post-primary output-form audit; does not replace frozen C1 truth labels."""
from pathlib import Path
import re,unicodedata,json
import pandas as pd
R=Path(__file__).resolve().parent
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().casefold()).strip()
def opening(text,gold):
 text=norm(text);gold=norm(gold)
 types=[('', 'direct_name'),('the city of ','direct_name'),('the capital city of ','direct_name'),('the capital city ','capital_designation'),('the capital ','capital_designation'),
        ('currently ','temporal_modifier'),('located in the capital city of ','locative'),('located in the city of ','locative'),('located in the town of ','locative'),
        ('located in ','locative'),('situated in ','locative'),('in ','locative'),('at ','locative')]
 for pre,kind in types:
  if re.match(re.escape(pre+gold)+r'(?!\w)',text):return kind
 return 'not_matched'
d=pd.read_csv(R/'LINKED_OUTPUTS_WITH_DIAGNOSTICS.csv',keep_default_na=False)
d['gold_opening_form']=[opening(a,b) for a,b in zip(d.output,d.gold)]
d['gold_in_target_opening']=d.gold_opening_form.ne('not_matched')
d['interpretation']='surface-form diagnostic only; partial geographic answer is not automatically full correct assertion'
d.to_csv(R/'POSTHOC_OUTPUT_FORM_DIAGNOSTICS.csv',index=False,encoding='utf-8-sig')
end=pd.read_csv(R/'ENDPOINT_FACT_DATA.csv')
case=end[(end.panel=='national42_common3')&(end.decode_group=='sample')&(end.delta_verified_correct_rate<=0)]
late=d[d.uid.isin(case.uid)&d.step.eq(658032)&d.decode_group.eq('sample')]
late.to_csv(R/'SEVEN_NONINCREASE_CASES_LATE_OUTPUTS.csv',index=False,encoding='utf-8-sig')
late.groupby(['subj','gold_opening_form']).size().reset_index(name='outputs').to_csv(R/'NONINCREASE_CASE_FORM_COUNTS.csv',index=False)
dis=d[d.literal_gold_slot.ne(d.first_claim_correct)]
dis.groupby(['step','decode_group','literal_gold_slot','first_claim_correct','output_state']).size().reset_index(name='n').to_csv(R/'NAME_VERIFICATION_DISAGREEMENT_COUNTS.csv',index=False)
status=dict(status='post-primary qualitative/format diagnostic',cases=7,late_sample_outputs=len(late),gold_in_target_opening=int(late.gold_in_target_opening.sum()),
            direct_name_or_designation=int(late.gold_opening_form.isin(['direct_name','capital_designation','temporal_modifier']).sum()),
            locative_gold=int(late.gold_opening_form.eq('locative').sum()),unmatched=int(late.gold_opening_form.eq('not_matched').sum()),
            primary_labels_unchanged=True,new_truth_adjudications=0,claim='lack of gain in frozen correctness can reflect output form and verification coverage; not proof of inaccessible factual knowledge')
(R/'OUTPUT_FORM_AUDIT_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));print(json.dumps(status,indent=2))
