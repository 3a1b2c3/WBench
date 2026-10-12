"""Render capability badges and README from the reviewed variant registry.
Usage: python render.py REGISTRY HOMEPAGE README
The homepage must already contain the capability filter integration.
"""
import re,json,html,pathlib,sys

def render(registry,homepage,readme):
 data=json.loads(registry.read_text()); models=data['models']; s=homepage.read_text(); found=set()
 labels={'realtime':{'yes':'Realtime','no':'Not realtime','unknown':'Realtime ?'},'audio':{'yes':'Audio','no':'No audio','unknown':'Audio ?'}}
 def rowedit(match):
  row=match.group(0);name=re.search(r'class="model-icon"[^>]*>([^<]+)',row).group(1).strip();tags=re.findall(r'<span class="[^"]*\bmodel-tag\b[^"]*">(.*?)</span>',row)
  key=name+' | '+', '.join(t for t in tags if t!='realtime');found.add(key);m=models[key]
  row=re.sub(r' data-(?:realtime|audio)="[^"]+"','',row)
  row=row.replace('<tr ',f'<tr data-realtime="{m["realtime"]["value"]}" data-audio="{m["audio"]["value"]}" ',1)
  row=re.sub(r'<span class="model-tag">realtime</span>','',row)
  row=re.sub(r'<div class="capability-badges">.*?</div>','',row)
  badges='<div class="capability-badges">'
  for axis in labels:
   c=m[axis];title=axis.title()+': '+({'yes':'Yes','no':'No','unknown':'Unknown'}[c['value']])+'. '+c['note']
   badges+=f'<span class="capability-badge cap-{c["value"]}" title="{html.escape(title,quote=True)}">{labels[axis][c["value"]]}</span>'
  badges+='</div>'
  return row.replace('<span class="creator">',badges+'<span class="creator">',1)
 s=re.sub(r'<tr (?:data-[^>]+)>.*?</tr>',rowedit,s)
 assert found==set(models),(found^set(models))
 homepage.write_text(s)
 text=readme.read_text(); start='<!-- capabilities:start -->';end='<!-- capabilities:end -->'
 lines=[start,'','**Model capabilities** (reviewed '+data['reviewed_at']+'). Labels apply to the listed variant and published system configuration. They do not change scores or ranks.','','- **Realtime — Yes:** documented realtime or near-realtime streaming / interaction under the stated control interface. **No:** documented offline-only or slower-than-realtime configuration. Hardware and latency details are retained in the evidence registry.','- **Audio — Yes:** jointly generated audio-video or viewpoint-following spatial sound. **No:** documented video-only generation. Input audio, added soundtracks and silent evaluation files do not establish this capability.','- **Unknown:** evidence is missing or cannot be matched to the evaluated variant; it does **not** mean No. WBench does not currently score latency or audio quality.','','| Model | Variant | Type | Realtime | Audio | Evidence |','|:---|:---|:---|:---:|:---:|:---|']
 for m in models.values():
  urls=list(dict.fromkeys(m[a]['sources'][0] for a in ['realtime','audio'] if m[a]['sources']))
  ev=' · '.join(f'[Source {i+1}]({u})' for i,u in enumerate(urls)) or 'Pending confirmation'
  vals=[{'yes':'Yes','no':'No','unknown':'Unknown'}[m[a]['value']] for a in ['realtime','audio']]
  lines.append('| '+' | '.join([m['name'],m['variant'] or '—',m['type'],*vals,ev])+' |')
 lines+=['','[Capability evidence and configuration notes](assets/model-capabilities.json). Corrections should identify the exact variant and link to an official model card, paper, or inference documentation.','',end]
 block='\n'.join(lines)
 if start in text:text=re.sub(re.escape(start)+r'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
 else:text=text.replace('## 🚀 Quick Start',block+'\n\n## 🚀 Quick Start',1)
 readme.write_text(text)

if __name__=='__main__':render(*(pathlib.Path(x) for x in sys.argv[1:]))
